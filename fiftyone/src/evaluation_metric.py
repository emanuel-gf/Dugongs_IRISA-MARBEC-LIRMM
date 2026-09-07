import numpy as np
import pandas as pd

def sweep_thresholds(res, iou_thresh=0.5, n_thresholds=200):
    """Sweep confidence thresholds and return metrics at F1-optimal point."""
    ytrue = np.array(res.ytrue)
    ypred = np.array(res.ypred)
    confs = np.array([v if v is not None else 0.0 for v in res.confs], dtype=float)
    ious  = np.array([v if v is not None else 0.0 for v in res.ious],  dtype=float)

    n_gt      = int((ypred == "(none)").sum() +
                    ((ytrue == "Dugong") & (ypred == "Dugong")).sum())
    pred_mask = ypred != "(none)"
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    best = {"f1": -1}
    for thresh in thresholds:
        mask = pred_mask & (confs >= thresh)
        tp   = int(((ytrue[mask] == "Dugong") & (ious[mask] >= iou_thresh)).sum())
        fp   = int((ytrue[mask] == "(none)").sum())
        fn   = int(n_gt - tp)

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)

        if f1 > best["f1"]:
            best = dict(
                threshold=thresh, tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1,
                n_gt=n_gt,
            )
    return best


def extract_all_metrics(results_dict: dict,
                        seed:int) -> pd.DataFrame:
    """
    Args:
        result_dict: eval_key and results as values (COCODetectionResults)

    Returns:
        DataFrame with one row per (seed, model)
    """
    rows = []
    for eval_key, res in results_dict.items():
            best = sweep_thresholds(res)
            rows.append({
                "seed":         seed,
                "eval_key":     eval_key,
                "mAP":          res.mAP(),
                "mAR":          res.mAR(),
                "threshold":    best["threshold"],
                "f1":           best["f1"],
                "precision":    best["precision"],
                "recall":       best["recall"],
                "tp":           best["tp"],
                "fp":           best["fp"],
                "fn":           best["fn"],
                "n_gt":         best["n_gt"],
            })
    print(f"Seed {seed} done — {len(results_dict)} models processed")

    return pd.DataFrame(rows)



## HELPERS
def create_eval_key_name(ds_field_name):
    splits = ds_field_name.split('_')

    if 'ACLR' in splits:
        part = splits[3]
        seed = splits[4]
        raw = splits[-1]
        return f"ACLR_p{part}_{seed}_{raw}"
    else:
        part = splits[2]
        seed = splits[3]
        raw = splits[-1]
        return f"p{part}_{seed}_{raw}"


def filter_ops(all_op, 
               nms_or_raw=None, 
               seed_term=None):
    def match(op):
        tokens = op.split('_')

        if nms_or_raw and nms_or_raw not in tokens:
            return False

        if seed_term and seed_term not in tokens:
            return False

        return True

    return [op for op in all_op if match(op)]


def select_evaluation_list(
    all_op,
    nms_or_raw='raw',
    baseline_field='baseline',
    seed_term=None,  
):
    """
    Create a dict with pred_field and eval_key.
    """
    nms_or_raw = nms_or_raw.strip().lower()

    evaluation_list = filter_ops(
        all_op,
        nms_or_raw=nms_or_raw,
        seed_term=seed_term
    )

    models_ft = [
        {
            "pred_field": r,
            "eval_key": create_eval_key_name(r),
        }
        for r in evaluation_list
    ]

    if nms_or_raw not in {"raw", "nms"}:
        raise ValueError(f"Invalid nms_or_raw value: {nms_or_raw}")

    models_ft.append({
        "pred_field": baseline_field,
        "eval_key": f"baseline_{seed_term}_{nms_or_raw}",
    })


    return models_ft