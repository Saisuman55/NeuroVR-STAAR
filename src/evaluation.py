"""Comprehensive evaluation utilities for 2D classification and 3D segmentation in NeuroVR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from src.classifier_2d import CLASS_NAMES


def evaluate_classification(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str] = CLASS_NAMES,
    save_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run model evaluation, computing accuracy, per-class metrics, and confusion matrix."""
    model.eval()
    model.to(device)

    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    labels = list(range(len(class_names)))
    names = list(class_names)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics_summary = {
        "accuracy": round(acc, 6),
        "total_samples": int(len(y_true)),
        "macro_avg": {
            "precision": round(report_dict["macro avg"]["precision"], 6),
            "recall": round(report_dict["macro avg"]["recall"], 6),
            "f1-score": round(report_dict["macro avg"]["f1-score"], 6),
        },
        "weighted_avg": {
            "precision": round(report_dict["weighted avg"]["precision"], 6),
            "recall": round(report_dict["weighted avg"]["recall"], 6),
            "f1-score": round(report_dict["weighted avg"]["f1-score"], 6),
        },
        "per_class": {
            name: {
                "precision": round(report_dict[name]["precision"], 6),
                "recall": round(report_dict[name]["recall"], 6),
                "f1-score": round(report_dict[name]["f1-score"], 6),
                "support": int(report_dict[name]["support"]),
            }
            for name in names if name in report_dict
        },
        "confusion_matrix": cm.tolist(),
    }

    if save_dir is not None:
        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        with open(out_path / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)

        with open(out_path / "classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        plot_confusion_matrix(cm, names, out_path / "confusion_matrix.png")

    return metrics_summary


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> Path:
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ticks = np.arange(len(class_names))
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=class_names,
        yticklabels=class_names,
        title="Classification Confusion Matrix",
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def dice_score(prediction: Tensor, target: Tensor, smooth: float = 1e-6) -> float:
    """Compute binary Dice coefficient for segmentation tensors."""
    p_flat = prediction.float().reshape(-1)
    t_flat = target.float().reshape(-1)

    p_sum = p_flat.sum().item()
    t_sum = t_flat.sum().item()

    # Both empty masks score 1.0
    if p_sum == 0 and t_sum == 0:
        return 1.0

    intersection = (p_flat * t_flat).sum().item()
    score = (2.0 * intersection + smooth) / (p_sum + t_sum + smooth)
    return float(score)


def iou_score(prediction: Tensor, target: Tensor, smooth: float = 1e-6) -> float:
    """Compute Intersection over Union (Jaccard Index) for segmentation tensors."""
    p_flat = prediction.float().reshape(-1)
    t_flat = target.float().reshape(-1)

    p_sum = p_flat.sum().item()
    t_sum = t_flat.sum().item()

    if p_sum == 0 and t_sum == 0:
        return 1.0

    intersection = (p_flat * t_flat).sum().item()
    union = p_sum + t_sum - intersection
    score = (intersection + smooth) / (union + smooth) if union > 0 else 0.0
    return float(score)


def segmentation_precision_recall(
    prediction: Tensor, target: Tensor, smooth: float = 1e-6
) -> tuple[float, float]:
    """Compute precision and recall for segmentation masks."""
    p_bool = prediction.bool().reshape(-1)
    t_bool = target.bool().reshape(-1)

    p_sum = p_bool.sum().item()
    t_sum = t_bool.sum().item()

    if p_sum == 0 and t_sum == 0:
        return 1.0, 1.0

    true_pos = (p_bool & t_bool).sum().item()
    precision = true_pos / p_sum if p_sum > 0 else 0.0
    recall = true_pos / t_sum if t_sum > 0 else 0.0
    return float(precision), float(recall)
