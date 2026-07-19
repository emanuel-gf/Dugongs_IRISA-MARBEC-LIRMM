"""
build_results_dataframe.py
==========================
Turn a FOLDER of RT-DETR tile-level prediction JSONs into ONE tidy DataFrame:
one row per evaluated run, keyed by (method, partition, seed). Every plot and
table in the thesis is then a groupby/pivot off this frame -- the JSONs are
never touched again.

Pipeline per JSON
-----------------
    reconstruct predictions -> FiftyOne field   (reconstruct_tile_predictions)
    evaluate_detections (COCO, IoU=0.5)         -> COCODetectionResults
    metrics: mAP_50, mAP_50_95, mAR, mAR_50,
             F1-optimal point (P/R/F1/TP/FP/FN) -> one dict
    parse method/partition/seed from filename   -> id columns
    -> append row

Filename convention
-------------------
    NNW_p{pct}_{method}_seed{n}_rtdetr_{date}_{time}_test_predictions.json
      NNW_p5_random_seed2_rtdetr_0715_1334_test_predictions.json
      NNW_p25_centroid_seed2_rtdetr_0715_1657_test_predictions.json
    The 'NNW' prefix (fine-tuned on West Papua) is ignored. Multi-word methods
    like 'centroid_uniqueness' are parsed correctly (everything between the
    partition token and '_seed').

Metric note
-----------
    mAP_50_95 = res.mAP()  (FiftyOne COCO average over IoU .50:.95)
    mAP_50    = AP@0.5 computed from the confidence-swept P-R curve at IoU 0.5.
                Single object task -> this IS mAP@0.5 and matches the thesis
                AP = integral p(r) dr definition, without relying on FiftyOne
                internals that vary across versions.

Requires
--------
    reconstruct_tile_predictions.add_tile_predictions  (companion module)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from reconstruct_tile_predictions import add_tile_predictions, derive_field_name


# ── filename parsing ──────────────────────────────────────────────────────────

_RUN_RE = re.compile(r"p(\d+)_(.+?)_seed(\d+)_rtdetr", re.IGNORECASE)


def parse_run_name(path, split_on="_rtdetr"):
    """
    Parse '(method, partition, seed)' from a prediction-JSON filename.

    Returns
    -------
    dict {method, partition, partition_pct, seed, field_name} or None if the
    name doesn't match the expected pattern.
    """
    stem = Path(path).stem
    m = _RUN_RE.search(stem)
    if not m:
        return None
    pct = int(m.group(1))
    return {
        "method":        m.group(2).lower(),   # 'centroid', 'centroid_uniqueness', 'ball', 'random'
        "partition":     pct / 100.0,          # 0.05, 0.25, ... (numeric, for plotting)
        "partition_pct": pct,                  # 5, 25, ...       (integer label)
        "seed":          int(m.group(3)),
        "field_name":    derive_field_name(stem, split_on=split_on),
    }


# ── metric helpers ────────────────────────────────────────────────────────────

def _ap_from_pr(recall, precision):
    """All-points AP (area under P-R curve). Single-class -> mAP@iou."""
    r = np.asarray(recall, dtype=float)
    p = np.asarray(precision, dtype=float)
    order = np.argsort(r)
    r, p = r[order], p[order]
    mrec = np.concatenate([[0.0], r, [1.0]])
    mpre = np.concatenate([[p[0] if len(p) else 0.0], p, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _sweep_metrics(res, iou=0.5, n_thresholds=200):
    """
    Confidence sweep at a fixed IoU over a COCODetectionResults object.
    Returns the F1-optimal operating point plus AP@iou / max-recall@iou.
    Positive labels inferred from res.ytrue (handles 'dugong' + 'calf').
    """
    ytrue = np.array(res.ytrue)
    ypred = np.array(res.ypred)
    confs = np.array([v if v is not None else 0.0 for v in res.confs], dtype=float)
    ious  = np.array([v if v is not None else 0.0 for v in res.ious],  dtype=float)

    pos = set(np.unique(ytrue)) - {"(none)"}
    is_pos = lambda a: np.isin(a, list(pos))

    n_gt = int((ypred == "(none)").sum() + (is_pos(ytrue) & is_pos(ypred)).sum())
    pred_mask = ypred != "(none)"

    ts = np.linspace(0.01, 0.99, n_thresholds)
    P = np.zeros(n_thresholds); R = np.zeros(n_thresholds); F = np.zeros(n_thresholds)
    TP = np.zeros(n_thresholds, int); FP = np.zeros(n_thresholds, int); FN = np.zeros(n_thresholds, int)

    for i, t in enumerate(ts):
        mask = pred_mask & (confs >= t)
        tp = int((is_pos(ytrue[mask]) & (ious[mask] >= iou)).sum())
        fp = int((ytrue[mask] == "(none)").sum())
        fn = n_gt - tp
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        P[i], R[i] = prec, rec
        F[i] = 2 * prec * rec / (prec + rec + 1e-9)
        TP[i], FP[i], FN[i] = tp, fp, fn

    b = int(np.argmax(F))
    return {
        "n_gt": n_gt,
        "positive_label": str(sorted(pos)),
        "mAP_50": _ap_from_pr(R, P),
        "mAR_50": float(R.max()),
        "f1_threshold": float(ts[b]),
        "f1": float(F[b]),
        "precision": float(P[b]),
        "recall": float(R[b]),
        "tp": int(TP[b]), "fp": int(FP[b]), "fn": int(FN[b]),
    }


# ── the ingest loop ───────────────────────────────────────────────────────────

def build_results_dataframe(
    dataset,
    json_dir,
    gt_field="ground_truth",
    view_tag_fmt="test_{seed}",
    eval_view=None,
    iou=0.5,
    n_thresholds=200,
    pattern="*_test_predictions.json",
    reconstruct=True,
    confidence_threshold=0.0,
    classwise=False,
    store_eval_key=False,
    cache_path=None,
    verbose=True,
):
    """
    Build the tidy per-run results DataFrame from a folder of prediction JSONs.

    Parameters
    ----------
    dataset          : FiftyOne dataset holding the tiles (with GT + test tags)
    json_dir         : folder of prediction JSONs, OR a list of JSON paths
    gt_field         : ground-truth detections field
    view_tag_fmt     : per-seed test view is dataset.match_tags(fmt.format(seed=seed)).
                       Ignored if eval_view is given.
    eval_view        : a single fixed view to evaluate ALL runs on (use this if
                       the test set is shared across seeds instead of tagged
                       per-seed).
    iou              : IoU threshold for matching / the sweep (default 0.5)
    n_thresholds     : confidence sweep resolution (bump to 500 for a smoother AP)
    pattern          : glob for discovering JSONs in json_dir
    reconstruct      : if True, write each run's predictions to a FiftyOne field
                       first (via add_tile_predictions). Set False if the fields
                       already exist.
    confidence_threshold : pre-eval confidence filter when reconstructing
    classwise        : passed to evaluate_detections (False = cross-class matches
                       allowed, i.e. dugong/calf treated as one object class)
    store_eval_key   : if True, store per-sample TP/FP/FN under the field name
                       (useful for viewing mistakes in the App; off by default to
                       keep the dataset light across many runs)
    cache_path       : if given, write the DataFrame here (.parquet preferred,
                       falls back to .csv)
    verbose          : print progress

    Returns
    -------
    df : pd.DataFrame, one row per run, columns:
         method, partition, partition_pct, seed, pred_field,
         mAP_50, mAP_50_95, mAR, mAR_50,
         f1, precision, recall, f1_threshold, tp, fp, fn,
         n_gt, iou, positive_label, status
    """
    def log(msg, level="info"):
        if verbose:
            pre = {"info": "  ", "ok": "OK  ", "warn": "WARN", "err": "ERR "}.get(level, "  ")
            print(f"[{pre}] {msg}")

    # discover files
    if isinstance(json_dir, (str, Path)):
        paths = sorted(Path(json_dir).glob(pattern))
    else:
        paths = [Path(p) for p in json_dir]
    if not paths:
        raise FileNotFoundError(f"No JSONs matched {pattern} in {json_dir}")
    log(f"Found {len(paths)} prediction JSONs.")

    rows = []
    for i, jp in enumerate(paths):
        ids = parse_run_name(jp)
        tag = None if eval_view is not None else view_tag_fmt.format(seed=ids["seed"]) if ids else "?"
        print(f"\n[{i+1}/{len(paths)}] {jp.name}")

        if ids is None:
            log(f"could not parse method/partition/seed -- skipped.", "warn")
            rows.append({"pred_field": jp.stem, "status": "unparsed_name"})
            continue

        field = ids["field_name"]
        try:
            # 1) reconstruct predictions into a FiftyOne field
            if reconstruct:
                add_tile_predictions(
                    dataset, jp, field_name=field,
                    confidence_threshold=confidence_threshold, verbose=False,
                )

            # 2) pick the evaluation view
            view = eval_view if eval_view is not None else dataset.match_tags(tag)
            n_total = len(view)
            if n_total == 0:
                raise ValueError(f"eval view '{tag}' is empty.")

            # GT-presence sanity check (catches the ground_truth vs ground_truthv2 trap)
            n_gt_tiles = len(view.exists(gt_field))
            if n_gt_tiles < n_total:
                log(f"{n_total - n_gt_tiles}/{n_total} tiles in '{tag}' have no "
                    f"'{gt_field}' -- their predictions will all count as FP. "
                    f"Check the GT field.", "warn")

            # 3) evaluate (COCO -> mAP@[.50:.95]) and sweep (-> AP@0.5, F1-opt)
            res = view.evaluate_detections(
                field, gt_field=gt_field, method="coco", iou=iou,
                classwise=classwise, compute_mAP=True,
                eval_key=(field if store_eval_key else None),
            )
            m = _sweep_metrics(res, iou=iou, n_thresholds=n_thresholds)

            rows.append({
                "method": ids["method"],
                "partition": ids["partition"],
                "partition_pct": ids["partition_pct"],
                "seed": ids["seed"],
                "pred_field": field,
                "mAP_50": m["mAP_50"],
                "mAP_50_95": float(res.mAP()),
                "mAR": float(res.mAR()),
                "mAR_50": m["mAR_50"],
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_threshold": m["f1_threshold"],
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "n_gt": m["n_gt"],
                "iou": iou,
                "positive_label": m["positive_label"],
                "status": "ok",
            })
            log(f"{ids['method']:<20} p{ids['partition_pct']:<3} seed{ids['seed']}  "
                f"mAP50={m['mAP_50']:.3f}  mAP50-95={res.mAP():.3f}  "
                f"F1={m['f1']:.3f}", "ok")

        except Exception as exc:
            log(f"failed: {exc}", "err")
            rows.append({
                "method": ids.get("method"), "partition": ids.get("partition"),
                "seed": ids.get("seed"), "pred_field": field,
                "status": f"error: {exc}",
            })

    df = pd.DataFrame(rows)

    # ── completeness check ────────────────────────────────────────────────────
    _report_completeness(df, verbose)

    # ── cache ─────────────────────────────────────────────────────────────────
    if cache_path:
        _cache(df, cache_path, log)

    return df


# ── completeness + caching ────────────────────────────────────────────────────

def _report_completeness(df, verbose=True):
    if not verbose or df.empty:
        return
    ok = df[df["status"] == "ok"] if "status" in df else df
    bad = df[df["status"] != "ok"] if "status" in df else df.iloc[0:0]

    print("\n" + "=" * 60)
    print(f"  Ingest complete: {len(ok)} ok, {len(bad)} failed, {len(df)} total")
    print("=" * 60)

    if len(ok):
        # runs per (method, partition): should equal the number of seeds everywhere
        grid = ok.groupby(["method", "partition"]).size().unstack(fill_value=0)
        print("\n  runs per method x partition (each cell should = #seeds):")
        print(grid.to_string())

        dup = ok.groupby(["method", "partition", "seed"]).size()
        dup = dup[dup > 1]
        if len(dup):
            print("\n  WARNING duplicate (method,partition,seed) rows:")
            print(dup.to_string())

    if len(bad):
        print("\n  failed / skipped runs:")
        for _, r in bad.iterrows():
            print(f"    {r.get('pred_field','?'):<32} {r.get('status')}")


def _cache(df, cache_path, log):
    p = Path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if p.suffix != ".parquet":
            p = p.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        log(f"cached -> {p}", "ok")
    except Exception as exc:
        p = p.with_suffix(".csv")
        df.to_csv(p, index=False)
        log(f"parquet failed ({exc}); wrote CSV -> {p}", "warn")