"""Model evaluation CLI script for NeuroVR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.classifier_2d import (
    CLASS_NAMES,
    BrainTumor2DDataset,
    create_classifier_2d,
    get_transforms,
    prepare_classification_splits,
)
from src.evaluation import evaluate_classification
from src.model_registry import get_model_status
from src.utils import load_config, select_device, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model checkpoints on validation or test sets (NeuroVR)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to central configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: checkpoints/classification/best.pt)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/classification",
        help="Directory to save evaluation reports",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    ckpt_path = Path(args.checkpoint or "checkpoints/classification/best.pt")
    if not ckpt_path.is_file():
        print("=" * 60)
        print("Model checkpoint unavailable — evaluation cannot be performed.")
        print(f"Checkpoint not found at: {ckpt_path}")
        print("Current model availability status:")
        print(json.dumps(get_model_status(config), indent=2))
        print("=" * 60)
        return 0

    device = select_device(config)
    logger.info("Loading checkpoint from %s on device %s...", ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location=device)

    arch = ckpt.get("model_architecture", "EfficientNet-B4")
    classes = ckpt.get("class_names", list(CLASS_NAMES))
    image_size = tuple(ckpt.get("image_size", (380, 380)))

    model = create_classifier_2d(model_name=arch, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load data split
    data_root = Path(config["paths"]["classification_dataset"])
    seed = int(config.get("project", {}).get("random_seed", 42))
    splits = prepare_classification_splits(data_root, seed=seed, classes=classes)
    eval_samples = splits[args.split]

    transform = get_transforms(image_size=image_size, is_training=False)
    dataset = BrainTumor2DDataset(eval_samples, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Evaluating on %s split (%d samples)...", args.split, len(dataset))
    results = evaluate_classification(
        model=model,
        data_loader=loader,
        device=device,
        class_names=classes,
        save_dir=args.output_dir,
    )

    print("=" * 60)
    print("NeuroVR Classification Evaluation Results:")
    print(f"Split: {args.split}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {results['macro_avg']['f1-score']:.4f}")
    print(f"Weighted F1: {results['weighted_avg']['f1-score']:.4f}")
    print("Per-class Metrics:")
    for c, met in results["per_class"].items():
        print(f"  {c:12s} - Precision: {met['precision']:.4f}, Recall: {met['recall']:.4f}, F1: {met['f1-score']:.4f}, Support: {met['support']}")
    print("=" * 60)
    print(f"Reports and confusion matrix saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
