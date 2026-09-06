"""2D Brain Tumor MRI classification models, dataset wrappers, and transforms."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import torchvision.models as tv_models
import torchvision.transforms as transforms


CLASS_NAMES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS: dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BrainTumor2DDataset(Dataset):
    """PyTorch Dataset wrapping real 2D brain MRI images."""

    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)
        return tensor, label

    def get_sample_path(self, index: int) -> Path:
        return self.samples[index][0]


def get_transforms(
    image_size: tuple[int, int] | list[int] = (380, 380),
    is_training: bool = False,
    augmentation_config: dict[str, Any] | None = None,
) -> transforms.Compose:
    """Construct deterministic validation transforms or augmented training transforms."""
    size = tuple(image_size)
    ops: list[Any] = [transforms.Resize(size)]

    if is_training and augmentation_config and augmentation_config.get("enabled", True):
        if augmentation_config.get("horizontal_flip", True):
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        degrees = float(augmentation_config.get("rotation_degrees", 15))
        if degrees > 0:
            ops.append(transforms.RandomRotation(degrees=degrees))
        bc = augmentation_config.get("brightness_contrast", {})
        if bc and bc.get("enabled", False):
            b_lim = float(bc.get("brightness_limit", 0.1))
            c_lim = float(bc.get("contrast_limit", 0.1))
            ops.append(transforms.ColorJitter(brightness=b_lim, contrast=c_lim))

    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def discover_dataset_files(
    dataset_root: str | Path,
    classes: Sequence[str] = CLASS_NAMES,
) -> dict[str, dict[str, list[Path]]]:
    """Discover real image files split by directory ('Training', 'Testing') and class."""
    root = Path(dataset_root)
    discovered: dict[str, dict[str, list[Path]]] = {
        "Training": {c: [] for c in classes},
        "Testing": {c: [] for c in classes},
    }
    for split in ("Training", "Testing"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for class_name in classes:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            for item in sorted(class_dir.iterdir()):
                if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    discovered[split][class_name].append(item)
    return discovered


def prepare_classification_splits(
    dataset_root: str | Path,
    validation_split: float = 0.2,
    seed: int = 42,
    classes: Sequence[str] = CLASS_NAMES,
) -> dict[str, list[tuple[Path, int]]]:
    """Split the Training directory into train/val subsets deterministically, leaving Testing untouched.

    Returns:
        dict with keys 'train', 'val', 'test', each containing a list of (Path, class_index) tuples.
    """
    discovered = discover_dataset_files(dataset_root, classes)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []

    # Stratified split per class from Training/
    for class_name in classes:
        train_files = list(discovered["Training"][class_name])
        # Sort first for deterministic ordering before shuffle
        train_files.sort()
        rng.shuffle(train_files)

        val_count = int(round(len(train_files) * validation_split))
        val_subset = train_files[:val_count]
        train_subset = train_files[val_count:]

        idx = class_to_idx[class_name]
        for p in train_subset:
            train_samples.append((p, idx))
        for p in val_subset:
            val_samples.append((p, idx))

        for p in discovered["Testing"][class_name]:
            test_samples.append((p, idx))

    # Verify zero data leakage across splits
    train_paths = {p for p, _ in train_samples}
    val_paths = {p for p, _ in val_samples}
    test_paths = {p for p, _ in test_samples}

    overlap_train_val = train_paths.intersection(val_paths)
    overlap_train_test = train_paths.intersection(test_paths)
    overlap_val_test = val_paths.intersection(test_paths)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Data leakage detected: train-val overlap={len(overlap_train_val)}, "
            f"train-test overlap={len(overlap_train_test)}, "
            f"val-test overlap={len(overlap_val_test)}"
        )

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


class SimpleTestCNN(nn.Module):
    """A lightweight 2D CNN architecture for fast unit testing."""

    def __init__(self, num_classes: int = 4, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        feats = self.features(x)
        flattened = torch.flatten(feats, 1)
        return self.classifier(flattened)


def create_classifier_2d(
    model_name: str = "EfficientNet-B4",
    num_classes: int = 4,
    dropout: float = 0.2,
    pretrained: bool = False,
) -> nn.Module:
    """Build the configured 2D classification neural network.

    Args:
        model_name: "EfficientNet-B4", "resnet18", "resnet34", "resnet50", or "simple_cnn"
        num_classes: Number of target categories (default: 4)
        dropout: Dropout rate before classification head
        pretrained: Whether to download/load ImageNet pretrained weights (default: False for reproducibility)
    """
    canonical = model_name.strip().lower().replace("-", "_")

    if canonical in {"efficientnet_b4", "efficientnet"}:
        weights = "DEFAULT" if pretrained else None
        model = tv_models.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=False),
            nn.Linear(in_features, num_classes),
        )
        return model

    if canonical == "resnet18":
        weights = "DEFAULT" if pretrained else None
        model = tv_models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if canonical == "resnet34":
        weights = "DEFAULT" if pretrained else None
        model = tv_models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if canonical == "simple_cnn":
        return SimpleTestCNN(num_classes=num_classes)

    raise ValueError(f"Unsupported classification model architecture: {model_name}")
