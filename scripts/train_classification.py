"""2D Brain Tumor MRI Classification Training Script for NeuroVR."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.classifier_2d import (
    CLASS_NAMES,
    BrainTumor2DDataset,
    create_classifier_2d,
    get_transforms,
    prepare_classification_splits,
)
from src.evaluation import evaluate_classification
from src.training import Trainer2D
from src.utils import load_config, select_device, set_random_seed, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train 2D Brain Tumor MRI Classifier (NeuroVR)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to central configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model architecture (e.g. EfficientNet-B4, resnet18, simple_cnn)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute a quick 1-epoch smoke test with limited batches",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (mps, cpu, cuda)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/classification",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/classification",
        help="Directory to save evaluation reports and training history",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    # 1. Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = select_device(config)
    logger.info("Selected compute device: %s", device)

    # 2. Random seed for deterministic reproducibility
    seed = int(config.get("project", {}).get("random_seed", 42))
    set_random_seed(seed)

    # 3. Parameters
    clf_config = config.get("classification", {})
    model_name = args.model_name or clf_config.get("model", "EfficientNet-B4")
    epochs = args.epochs or (1 if args.smoke_test else int(clf_config.get("epochs", 10)))
    batch_size = args.batch_size or (4 if args.smoke_test else int(clf_config.get("batch_size", 16)))
    lr = args.lr or float(clf_config.get("optimizer", {}).get("learning_rate", 0.0001))
    weight_decay = float(clf_config.get("optimizer", {}).get("weight_decay", 0.0001))
    image_size = tuple(clf_config.get("input_size", (380, 380)))
    val_split = float(clf_config.get("validation_split", 0.2))

    # For smoke tests, use smaller image size for rapid execution
    if args.smoke_test and model_name.lower() in {"efficientnet-b4", "efficientnet_b4"}:
        image_size = (224, 224)

    logger.info(
        "Training configuration: model=%s, epochs=%d, batch_size=%d, lr=%g, image_size=%s, smoke_test=%s",
        model_name, epochs, batch_size, lr, image_size, args.smoke_test,
    )

    # 4. Prepare data splits
    data_root = Path(config["paths"]["classification_dataset"])
    splits = prepare_classification_splits(
        data_root,
        validation_split=val_split,
        seed=seed,
        classes=CLASS_NAMES,
    )

    train_transform = get_transforms(image_size=image_size, is_training=not args.smoke_test, augmentation_config=clf_config.get("augmentation"))
    eval_transform = get_transforms(image_size=image_size, is_training=False)

    train_ds = BrainTumor2DDataset(splits["train"], transform=train_transform)
    val_ds = BrainTumor2DDataset(splits["val"], transform=eval_transform)
    test_ds = BrainTumor2DDataset(splits["test"], transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info(
        "Dataset sizes: Train=%d, Validation=%d, Test=%d (Total=%d)",
        len(train_ds), len(val_ds), len(test_ds), len(train_ds) + len(val_ds) + len(test_ds),
    )

    # 5. Build model
    model = create_classifier_2d(
        model_name=model_name,
        num_classes=len(CLASS_NAMES),
        dropout=float(clf_config.get("dropout", 0.2)),
        pretrained=False,
    )

    # 6. Loss and optimizer
    label_smoothing = float(clf_config.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 7. Trainer
    trainer = Trainer2D(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        report_dir=args.report_dir,
        model_name=model_name,
        image_size=image_size,
    )

    max_batches = 10 if args.smoke_test else None
    start_time = time.time()
    history = trainer.train(epochs=epochs, max_batches_per_epoch=max_batches, verbose=True)
    duration = time.time() - start_time

    logger.info("Training complete in %.2f seconds.", duration)

    # 8. Evaluate on test split with the saved best model
    best_ckpt_path = Path(args.checkpoint_dir) / "best.pt"
    if best_ckpt_path.is_file():
        logger.info("Evaluating best model from %s on test set...", best_ckpt_path)
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_eval = evaluate_classification(
            model=model,
            data_loader=test_loader,
            device=device,
            class_names=CLASS_NAMES,
            save_dir=args.report_dir,
        )
        logger.info("Test Accuracy: %.4f (over %d test images)", test_eval["accuracy"], test_eval["total_samples"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
