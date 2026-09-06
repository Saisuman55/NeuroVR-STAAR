"""Model registry and safe inference integration for NeuroVR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from PIL import Image
import torch
from torch import nn

from src.classifier_2d import (
    CLASS_NAMES,
    create_classifier_2d,
    get_transforms,
)


def get_classification_checkpoint_path(config: dict[str, Any]) -> Path | None:
    """Find the configured classification checkpoint.
    
    Returns a Path only if explicitly configured in config['classification']['checkpoint_path']
    (or if set to 'auto', searching known checkpoint directories).
    """
    configured = config.get("classification", {}).get("checkpoint_path")
    if configured == "auto":
        candidates = [
            "checkpoints/classification/best.pt",
            "checkpoints/classification/last.pt",
            "models/classifier/best.pt",
            "models/classifier/last.pt",
        ]
        for cand in candidates:
            p = Path(cand)
            if p.is_file():
                return p
        return None

    if configured:
        p = Path(configured)
        return p if p.is_file() else None

    return None


def get_segmentation_checkpoint_path(config: dict[str, Any]) -> Path | None:
    """Find the configured 3D segmentation checkpoint."""
    configured = (
        config.get("three_d_segmentation", {}).get("checkpoint_path")
        or config.get("segmentation", {}).get("checkpoint_path")
    )
    if configured == "auto":
        candidates = [
            "checkpoints/segmentation/best.pt",
            "checkpoints/segmentation/last.pt",
            "models/segmenter/best.pt",
        ]
        for cand in candidates:
            p = Path(cand)
            if p.is_file():
                return p
        return None

    if configured:
        p = Path(configured)
        return p if p.is_file() else None

    return None


def get_classification_model_status(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect whether a trained 2D classification model is available."""
    ckpt_path = get_classification_checkpoint_path(config)
    if ckpt_path is None:
        return {
            "status": "requires_checkpoint",
            "message": "2D EfficientNet-B4 inference is supported by the application architecture and model registry. A compatible trained checkpoint must be mounted or configured before predictions can be generated.",
            "ui_message": "Classification model not loaded. Connect a trained EfficientNet-B4 checkpoint to enable analysis.",
            "checkpoint": None,
            "architecture": config.get("classification", {}).get("model", "EfficientNet-B4"),
            "classes": list(CLASS_NAMES),
        }

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return {
            "status": "available",
            "message": "Trained classification model is available for inference.",
            "checkpoint": str(ckpt_path),
            "architecture": ckpt.get("model_architecture", config.get("classification", {}).get("model", "EfficientNet-B4")),
            "epoch": ckpt.get("epoch"),
            "classes": ckpt.get("class_names", list(CLASS_NAMES)),
            "image_size": ckpt.get("image_size"),
            "val_metrics": ckpt.get("val_metrics"),
            "timestamp": ckpt.get("timestamp"),
        }
    except Exception as err:
        return {
            "status": "requires_checkpoint",
            "message": f"Corrupt or unreadable checkpoint: {err}",
            "checkpoint": str(ckpt_path),
        }


def get_segmentation_model_status(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect whether a trained 3D segmentation model is available."""
    ckpt_path = get_segmentation_checkpoint_path(config)
    # Check if BraTS data is present
    top_manifest = Path(config["paths"]["segmentation_dataset"]) / "brats_manifest.json"
    legacy_manifest = Path(config["paths"]["segmentation_dataset"]) / "brats" / "dataset_manifest.json"
    brats_imported = top_manifest.is_file() or legacy_manifest.is_file()

    if ckpt_path is None:
        return {
            "status": "not_imported",
            "message": "3D U-Net segmentation architecture is implemented as a baseline. BraTS-style benchmark data and/or the required trained segmentation checkpoint are not currently imported into this environment.",
            "ui_message": "3D segmentation is unavailable until the required dataset and trained model resources are configured.",
            "checkpoint": None,
            "dataset_status": "imported" if brats_imported else "not_imported",
            "architecture": config.get("three_d_segmentation", {}).get("architecture", "3d_unet"),
        }

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return {
            "status": "available",
            "checkpoint": str(ckpt_path),
            "architecture": ckpt.get("architecture", "3d_unet"),
            "epoch": ckpt.get("epoch"),
        }
    except Exception as err:
        return {
            "status": "unavailable",
            "reason": f"Corrupt or unreadable checkpoint: {err}",
            "checkpoint": str(ckpt_path),
        }


def get_model_status(config: dict[str, Any]) -> dict[str, Any]:
    """Return central status report for all models in NeuroVR."""
    return {
        "classification": get_classification_model_status(config),
        "segmentation": get_segmentation_model_status(config),
    }


def load_classification_model_for_inference(
    config: dict[str, Any],
    device: torch.device | None = None,
) -> tuple[nn.Module | None, dict[str, Any] | None]:
    """Load a trained classification model if a valid checkpoint is present."""
    ckpt_path = get_classification_checkpoint_path(config)
    if ckpt_path is None:
        return None, None

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        arch = ckpt.get("model_architecture", config.get("classification", {}).get("model", "EfficientNet-B4"))
        classes = ckpt.get("class_names", list(CLASS_NAMES))
        model = create_classifier_2d(model_name=arch, num_classes=len(classes))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        if device is not None:
            model.to(device)
        return model, ckpt
    except Exception:
        return None, None


def predict_2d_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
    class_names: Sequence[str] = CLASS_NAMES,
    image_size: tuple[int, int] = (380, 380),
) -> dict[str, Any]:
    """Perform honest inference on a single 2D brain MRI scan using model.eval() and torch.no_grad()."""
    with Image.open(image_path) as img:
        rgb_image = img.convert("RGB")

    transform = get_transforms(image_size=image_size, is_training=False)
    tensor = transform(rgb_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(probabilities.argmax())
    predicted_class = class_names[pred_idx]
    confidence = float(probabilities[pred_idx])

    probs_dict = {
        name: round(float(probabilities[i]), 4)
        for i, name in enumerate(class_names)
    }

    return {
        "status": "predicted",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": probs_dict,
    }
