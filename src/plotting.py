import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["Genuine", "Fraud"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_roc_curve(y_true, y_score, title="ROC Curve"):
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(title)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    fig.tight_layout()
    return fig, ax


def plot_precision_recall_curve(y_true, y_score, title="Precision-Recall Curve"):
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
