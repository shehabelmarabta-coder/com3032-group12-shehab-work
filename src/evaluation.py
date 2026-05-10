import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
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
