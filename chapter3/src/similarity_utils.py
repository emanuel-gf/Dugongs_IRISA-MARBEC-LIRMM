"""
similarity_utils.py
=====================

Computes cosine similarity between embedding vectors looked up by ID --
works identically for full-tile embeddings (keyed by sample_id) or object
embeddings (keyed by detection_id), since both are just "some ID -> some
1024-dim vector" once loaded.

Usage
-----
    from similarity_utils import build_embedding_lookup, calculate_cosine_similarity

    # Build the lookup ONCE (cheap, O(N)) -- reuse across many similarity calls
    lookup = build_embedding_lookup(data["sample_ids"], data["embeddings"])

    # Single anchor vs single target
    sim = calculate_cosine_similarity("id_a", "id_b", lookup)

    # Single anchor vs a list of targets
    sims = calculate_cosine_similarity("id_a", ["id_b", "id_c", "id_d"], lookup)
"""

import numpy as np


def build_embedding_lookup(ids: np.ndarray, embeddings: np.ndarray) -> dict:
    """
    Builds an {id: embedding_vector} dict in a single O(N) pass -- build
    this ONCE per embedding source (full-tile or object) and reuse it for
    every subsequent similarity call, rather than re-scanning the raw
    arrays inside calculate_cosine_similarity on every call.

    Parameters
    ----------
    ids        : np.ndarray (N,) -- sample_ids or detection_ids
    embeddings : np.ndarray (N, dim)

    Returns
    -------
    dict {id: np.ndarray (dim,)}
    """
    assert len(ids) == len(embeddings), (
        f"ids ({len(ids)}) and embeddings ({len(embeddings)}) must be the same length."
    )
    return {str(i): emb for i, emb in zip(ids, embeddings)}


def calculate_cosine_similarity(id_emb_anchor: str, id_emb_targets, lookup: dict):
    """
    Computes cosine similarity between one anchor embedding and either a
    single target embedding or a list of target embeddings, looked up by ID.

    Works for either full-tile embeddings (lookup keyed by sample_id) or
    object embeddings (lookup keyed by detection_id) -- the function itself
    doesn't care which, since both are just ID -> vector.

    Parameters
    ----------
    id_emb_anchor : str -- ID of the anchor embedding
    id_emb_targets : str or list[str] -- ID of a single target, OR a list
                      of target IDs to compare the anchor against
    lookup        : dict {id: embedding_vector}, as built by
                      build_embedding_lookup()

    Returns
    -------
    If id_emb_targets is a single str  -> float (cosine similarity, [-1, 1])
    If id_emb_targets is a list        -> dict {target_id: float}, same
                                            order as the input list
    """
    if id_emb_anchor not in lookup:
        raise KeyError(f"Anchor id '{id_emb_anchor}' not found in lookup.")

    anchor_vec = lookup[id_emb_anchor]
    anchor_norm = anchor_vec / (np.linalg.norm(anchor_vec) + 1e-12)

    is_single = isinstance(id_emb_targets, str)
    target_ids = [id_emb_targets] if is_single else list(id_emb_targets)

    missing = [tid for tid in target_ids if tid not in lookup]
    if missing:
        raise KeyError(f"Target id(s) not found in lookup: {missing}")

    results = {}
    for tid in target_ids:
        target_vec = lookup[tid]
        target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-12)
        results[tid] = float(np.dot(anchor_norm, target_norm))

    return results[id_emb_targets] if is_single else results