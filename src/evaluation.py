import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, y_score=None):
    """Return common binary classification metrics as a dictionary."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)

    return metrics


def create_classification_report(y_true, y_pred):
    """Create a readable classification report."""
    return classification_report(y_true, y_pred, zero_division=0)


def create_confusion_matrix(y_true, y_pred):
    """Create a confusion matrix for binary classification."""
    return confusion_matrix(y_true, y_pred)


def recall_at_precision(y_true, y_score, target_precision=0.90):
    """Find threshold where precision >= target_precision, return (recall, threshold).

    If no threshold achieves target precision, return (0.0, max threshold).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precisions[:-1] >= target_precision)[0]
    if len(valid) == 0:
        return 0.0, float(thresholds[-1])
    idx = valid[np.argmax(recalls[:-1][valid])]
    return float(recalls[idx]), float(thresholds[idx])


def tune_threshold_f1(y_true, y_score):
    """Find threshold that maximises F1 on the PR curve.

    Returns (threshold, f1_at_threshold).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0,
    )
    idx = np.argmax(f1_scores)
    return float(thresholds[idx]), float(f1_scores[idx])


def evaluate_model_full(
    model_name,
    member,
    split,
    imbalance_strategy,
    y_true,
    y_score,
    threshold,
):
    """Return the locked group-comparison results schema as a dict.

    Caller passes y_score (probability or decision function output) and threshold;
    y_pred is computed internally as (y_score >= threshold).
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    recall_at_90, _ = recall_at_precision(y_true, y_score, 0.90)
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "member": member,
        "model": model_name,
        "split": split,
        "imbalance_strategy": imbalance_strategy,
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "recall_at_90p": float(recall_at_90),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "fpr": fpr,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
