"""
classification_pipeline.py
============================

Binary classification pipeline (dugong presence/absence) on pooled DINOv3
embeddings.

This module does NOT assemble train/test splits -- that's left entirely to
the caller (build X_train/y_train/X_test/y_test/test_ids yourself, however
you like). It only provides:

  - evaluate_predictions()    : shared metric computation, used identically
                                  across every model type, returns a tidy
                                  per-sample results DataFrame plus a list
                                  of misclassified sample_ids for tracing
                                  back into FiftyOne.
  - fit_logistic_regression() : sklearn LogisticRegression wrapper.
  - fit_random_forest()       : sklearn RandomForestClassifier wrapper.
  - DugongMLP / fit_mlp()     : shallow expand-then-contract MLP (torch),
                                  class-weighted BCE loss, early stopping
                                  on a validation split carved from train.

Usage
-----
    from classification_pipeline import fit_logistic_regression, fit_random_forest, fit_mlp

    result = fit_logistic_regression(X_train, y_train, X_test, y_test, test_ids=test_ids)
    print(result["metrics"])
    result["results_df"]          # tidy per-sample DataFrame
    result["failed_ids"]          # sample_ids the model got wrong

    # Visualise failures directly in the FiftyOne App:
    session.view = dataset.select(result["failed_ids"])
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ── Shared evaluation ──────────────────────────────────────────────────────────

def evaluate_predictions(
    y_test, y_pred, y_proba=None, test_ids=None,
    model_name="model",
):
    """
    Computes a standard metric set, used identically across every model type
    so results are directly comparable. Also builds a tidy per-sample
    results DataFrame and the list of sample_ids the model got wrong, so
    failures can be loaded directly into the FiftyOne App.

    Parameters
    ----------
    y_test   : array-like (n_test,) int {0,1} -- ground truth
    y_pred   : array-like (n_test,) int {0,1} -- predicted class
    y_proba  : array-like (n_test,) float or None -- predicted P(class=1),
                used for ROC-AUC if provided, also stored in results_df
    test_ids : array-like (n_test,) str or None -- sample_ids, SAME ROW
                ORDER as y_test/y_pred. Required to get failed_ids back;
                if omitted, results_df still works but failed_ids will be
                positional integer indices instead of sample_ids.
    model_name : str -- label for printing and stored in results_df["model"]

    Returns
    -------
    dict with keys:
        metrics      : dict of scalar metrics (+ confusion_matrix as nested list)
        results_df   : pd.DataFrame, one row per test sample, columns:
                        sample_id, y_true, y_pred, y_proba (if given),
                        correct (bool), model
        failed_ids   : list of sample_ids (or positional indices) where
                        y_pred != y_true -- pass straight to
                        dataset.select(failed_ids) / session.view=...
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except ValueError:
            metrics["roc_auc"] = None   # e.g. only one class present in y_test

    print(f"\n=== {model_name} ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Confusion matrix [[TN,FP],[FN,TP]]:\n{np.array(metrics['confusion_matrix'])}")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"],
                                zero_division=0))

    # ── Tidy per-sample results DataFrame ────────────────────────────────────
    correct = y_pred == y_test
    n = len(y_test)

    if test_ids is None:
        print("  NOTE: test_ids not provided -- results_df/failed_ids will use "
              "positional integer indices instead of real sample_ids.")
        ids_col = np.arange(n)
    else:
        ids_col = np.asarray(test_ids)
        assert len(ids_col) == n, (
            f"test_ids length ({len(ids_col)}) doesn't match y_test length ({n})"
        )

    df_data = {
        "sample_id": ids_col,
        "y_true":    y_test,
        "y_pred":    y_pred,
        "correct":   correct,
        "model":     model_name,
    }
    if y_proba is not None:
        df_data["y_proba"] = np.asarray(y_proba)

    results_df = pd.DataFrame(df_data)

    failed_ids = ids_col[~correct].tolist()
    print(f"  Failed predictions: {len(failed_ids)} / {n}")

    return {
        "metrics": metrics,
        "results_df": results_df,
        "failed_ids": failed_ids,
    }


# ── Model 1: Logistic regression ────────────────────────────────────────────────

def fit_logistic_regression(
    X_train, y_train, X_test, y_test,
    test_ids=None,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight="balanced",
    random_state: int = 42,
):
    """
    Fits a logistic regression classifier and evaluates on the test set.

    class_weight="balanced" is the default since presence/absence tiles are
    very unlikely to be perfectly balanced (negatives typically outnumber
    positives in tiled aerial imagery) -- this reweights the loss inversely
    proportional to class frequency rather than letting the majority class
    dominate.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : your own split arrays
    test_ids      : array-like or None -- sample_ids in the SAME ROW ORDER
                     as X_test/y_test. Needed to get real sample_ids back
                     in results_df/failed_ids (see evaluate_predictions).
    C             : float -- inverse regularisation strength
    max_iter      : int   -- max solver iterations
    class_weight  : "balanced" | dict | None
    random_state  : int

    Returns
    -------
    dict with keys: model, y_pred, y_proba, metrics, results_df, failed_ids
    """
    model = LogisticRegression(
        C=C, max_iter=max_iter, class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]   # P(class = 1 / positive)

    eval_out = evaluate_predictions(
        y_test, y_pred, y_proba, test_ids=test_ids,
        model_name="Logistic Regression",
    )

    return {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        **eval_out,   # metrics, results_df, failed_ids
    }


# ── Model 2: Random forest ──────────────────────────────────────────────────────

def fit_random_forest(
    X_train, y_train, X_test, y_test,
    test_ids=None,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    class_weight="balanced",
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Fits a random forest classifier and evaluates on the test set.

    Same class_weight="balanced" reasoning as logistic regression -- avoids
    the majority (negative) class dominating the learned splits.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : your own split arrays
    test_ids          : array-like or None -- sample_ids, SAME ROW ORDER as
                         X_test/y_test (see evaluate_predictions)
    n_estimators       : int   -- number of trees (default 300)
    max_depth          : int or None -- max tree depth (None = unlimited,
                          sklearn default behaviour)
    min_samples_leaf   : int   -- minimum samples per leaf (higher = more
                          regularised, less overfitting on a small train set)
    class_weight       : "balanced" | dict | None
    n_jobs             : int   -- parallel jobs for tree building (-1 = all cores)
    random_state        : int

    Returns
    -------
    dict with keys: model, y_pred, y_proba, metrics, results_df, failed_ids,
                     feature_importances (np.ndarray, len = embedding dim)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    eval_out = evaluate_predictions(
        y_test, y_pred, y_proba, test_ids=test_ids,
        model_name="Random Forest",
    )

    return {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_importances": model.feature_importances_,
        **eval_out,   # metrics, results_df, failed_ids
    }


# ── Model 3: Shallow MLP (expand-then-contract) ─────────────────────────────────

class DugongMLP(nn.Module):
    """
    Shallow expand-then-contract MLP:
        1024 -> 1536 -> 768 -> 256 -> 1 (logit)

    ReLU activations + dropout on every hidden layer (regularisation against
    overfitting -- with only a few thousand training rows and a 1024-dim
    input, this network has enough capacity to memorise the training set
    without it). Outputs a single raw logit (no sigmoid applied internally)
    -- use BCEWithLogitsLoss for training and torch.sigmoid(logits) to get
    probabilities at inference time.
    """
    def __init__(self, input_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1536),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1536, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),   # raw logit, no activation
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # (batch,) instead of (batch, 1)


def find_best_threshold(y_true, y_proba, metric: str = "f1", n_steps: int = 199):
    """
    Sweeps decision thresholds in (0, 1) and returns the one that maximises
    the given metric on the provided (y_true, y_proba) pair.

    IMPORTANT: call this on a VALIDATION split, never on the test set --
    tuning the threshold against test labels would leak test information
    into model selection and bias the reported test metrics optimistically.

    Parameters
    ----------
    y_true   : array-like (n,) int {0,1}
    y_proba  : array-like (n,) float -- predicted P(class=1)
    metric   : "f1" | "precision" | "recall" | "accuracy"
    n_steps  : int -- number of threshold candidates to try, evenly spaced
                in (0, 1) (default 199 -> step size 0.005)

    Returns
    -------
    best_threshold : float
    best_score     : float
    """
    metric_fns = {
        "f1":        f1_score,
        "precision": precision_score,
        "recall":    recall_score,
        "accuracy":  accuracy_score,
    }
    if metric not in metric_fns:
        raise ValueError(f"metric must be one of {list(metric_fns)}, got '{metric}'")
    score_fn = metric_fns[metric]

    y_true  = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    thresholds = np.linspace(0.005, 0.995, n_steps)
    best_threshold, best_score = 0.5, -1.0

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        score = score_fn(y_true, y_pred_t, zero_division=0) if metric != "accuracy" \
                else score_fn(y_true, y_pred_t)
        if score > best_score:
            best_score = score
            best_threshold = t

    return float(best_threshold), float(best_score)


def fit_mlp(
    X_train, y_train, X_test, y_test,
    test_ids=None,
    model_fn=None,
    val_size: float = 0.15,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    tune_threshold: bool = True,
    threshold_metric: str = "f1",
    device: str | None = None,
    random_state: int = 42,
):
    """
    Trains a binary classifier MLP with class-weighted BCEWithLogitsLoss and
    early stopping on a validation split carved from X_train, then evaluates
    on X_test using the same evaluate_predictions() used by the other models.

    Class weighting: pos_weight = n_negative / n_positive (computed on the
    TRAIN split only, same imbalance-correction logic as sklearn's
    class_weight="balanced" used for Logistic Regression / Random Forest).

    Early stopping: validation loss is checked every epoch; training stops
    if it hasn't improved for `patience` consecutive epochs, and the
    model weights from the BEST validation-loss epoch are restored before
    evaluating on X_test (not simply the weights at the final epoch).

    Threshold tuning (tune_threshold=True, default): a class-weighted BCE
    loss systematically shifts where the "natural" decision boundary sits
    in probability space, so a fixed 0.5 cutoff is often miscalibrated --
    typically producing high recall but poor precision (lots of false
    positives). After training, the best decision threshold is found by
    sweeping against the VALIDATION set (never the test set, to avoid
    leaking test information into model selection) and reused to compute
    y_pred on the test set.

    Parameters
    ----------
    X_train, y_train, X_test, y_test : your own split arrays
    test_ids       : array-like or None -- sample_ids, SAME ROW ORDER as
                      X_test/y_test (see evaluate_predictions)
    model_fn       : callable or None -- a zero-arg constructor returning an
                      nn.Module outputting a single raw logit per sample, e.g.
                          model_fn=lambda: DugongMLP(input_dim=X_train.shape[1])
                      Defaults to DugongMLP(input_dim=X_train.shape[1]) if
                      not provided.
    val_size       : float -- fraction of X_train carved out for validation
                      / early stopping / threshold tuning (default 0.15)
    batch_size     : int
    max_epochs     : int -- upper bound; early stopping will usually exit sooner
    patience       : int -- epochs with no val-loss improvement before stopping
    lr             : float -- Adam learning rate
    weight_decay   : float -- Adam L2 regularisation (default 1e-3, raised
                      from 1e-4 to fight overfitting on a small training set)
    tune_threshold : bool -- if True (default), pick the decision threshold
                      that maximises threshold_metric on the validation set,
                      instead of using a fixed 0.5 cutoff
    threshold_metric : "f1" | "precision" | "recall" | "accuracy"
    device         : "cuda" | "cpu" | None (auto-detect)
    random_state   : int -- used for the train/val split and torch seeding

    Returns
    -------
    dict with keys: model, y_pred, y_proba, metrics, results_df, failed_ids,
                     train_history, decision_threshold (the cutoff actually
                     used to produce y_pred -- 0.5 if tune_threshold=False)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(random_state)

    if model_fn is None:
        model_fn = lambda: DugongMLP(input_dim=X_train.shape[1])

    # ── Train/val split (stratified, since positives are rare) ──────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=val_size,
        stratify=y_train, random_state=random_state, shuffle=True,
    )
    print(f"  Train: {X_tr.shape[0]} (pos={y_tr.sum()})  "
          f"Val: {X_val.shape[0]} (pos={y_val.sum()})  device={device}")

    # ── Class-weighted loss (mirrors class_weight='balanced') ────────────────
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int(len(y_tr) - y_tr.sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    print(f"  pos_weight (n_neg/n_pos) = {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Tensors / loaders ──────────────────────────────────────────────────
    def _to_tensor(X, y=None):
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        if y is None:
            return X_t
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)
        return X_t, y_t

    X_tr_t,  y_tr_t  = _to_tensor(X_tr, y_tr)
    X_val_t, y_val_t = _to_tensor(X_val, y_val)
    X_test_t          = _to_tensor(X_test)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size, shuffle=False)

    # ── Model / optimiser ─────────────────────────────────────────────────
    model = model_fn().to(device)
    print(f"  Model: {model.__class__.__name__}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_loss   = float("inf")
    best_state_dict = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)
        train_loss = train_loss_sum / train_n

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_loss_sum / val_n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:>3}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  best={best_val_loss:.4f}  "
                  f"no_improve={epochs_no_improve}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(no val improvement for {patience} epochs).")
            break

    # ── Restore best-val-loss weights before evaluating on test ─────────────
    model.load_state_dict(best_state_dict)
    print(f"  Restored weights from best val_loss={best_val_loss:.4f}")

    # ── Threshold tuning on the VALIDATION set (never on test) ──────────────
    model.eval()
    if tune_threshold:
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_proba  = torch.sigmoid(val_logits).cpu().numpy()
        decision_threshold, best_val_score = find_best_threshold(
            y_val, val_proba, metric=threshold_metric,
        )
        print(f"  Tuned threshold (max {threshold_metric} on val) = "
              f"{decision_threshold:.3f}  (val {threshold_metric}={best_val_score:.4f})")
    else:
        decision_threshold = 0.5
        print(f"  Using fixed decision_threshold = 0.5 (tune_threshold=False)")

    # ── Inference on X_test, using the (possibly tuned) threshold ───────────
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_proba = torch.sigmoid(logits).cpu().numpy()
    y_pred = (y_proba >= decision_threshold).astype(int)

    eval_out = evaluate_predictions(
        y_test, y_pred, y_proba, test_ids=test_ids,
        model_name="MLP",
    )

    return {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "decision_threshold": decision_threshold,
        "train_history": history,
        **eval_out,   # metrics, results_df, failed_ids
    }