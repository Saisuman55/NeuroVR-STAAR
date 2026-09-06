"""Dataset inspection, pairing, and reproducible split utilities for NeuroVR."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
from PIL import Image, UnidentifiedImageError


SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class ImageSample:
    """A classification image and its inspection result."""

    path: Path
    class_name: str | None
    dimensions: tuple[int, int] | None
    extension: str
    valid: bool
    error: str | None = None


@dataclass
class ClassificationReport:
    """Inspection results for a classification directory."""

    dataset_path: Path
    status: str
    expected_classes: tuple[str, ...]
    samples: list[ImageSample] = field(default_factory=list)
    total_files: int = 0
    valid_images: int = 0
    invalid_images: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    image_size_statistics: dict[str, Any] = field(default_factory=dict)
    file_extension_counts: dict[str, int] = field(default_factory=dict)
    missing_classes: list[str] = field(default_factory=list)
    unexpected_classes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SegmentationPair:
    """An image-mask pair, including missing or invalid members."""

    key: str
    image_path: Path | None
    mask_path: Path | None
    image_dimensions: tuple[int, int] | None
    mask_dimensions: tuple[int, int] | None
    valid: bool
    mask_is_binary: bool | None
    mask_values: tuple[int, ...] = ()
    mask_pixels: int = 0
    foreground_pixels: int = 0
    error: str | None = None


@dataclass
class SegmentationReport:
    """Inspection results for a paired segmentation directory."""

    dataset_path: Path
    status: str
    mask_suffix: str
    pairs: list[SegmentationPair] = field(default_factory=list)
    total_images: int = 0
    total_masks: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    images_without_masks: list[str] = field(default_factory=list)
    masks_without_images: list[str] = field(default_factory=list)
    duplicate_conflicts: list[str] = field(default_factory=list)
    empty_mask_count: int = 0
    non_empty_mask_count: int = 0
    total_mask_pixels: int = 0
    foreground_pixels: int = 0
    background_pixels: int = 0
    foreground_percentage: float = 0.0


T = TypeVar("T")


def _image_dimensions(path: Path) -> tuple[int, int]:
    """Open an image safely and return width and height."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(str(error) or "unreadable image") from error


def _all_files(path: Path) -> list[Path]:
    """Return deterministic recursive file paths below a directory."""
    return sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix())


def inspect_classification_dataset(
    dataset_path: str | Path,
    expected_classes: Sequence[str],
) -> ClassificationReport:
    """Inspect classification images without assuming a fixed directory layout."""
    path = Path(dataset_path)
    expected = tuple(expected_classes)
    report = ClassificationReport(path, "dataset_not_present", expected)
    if not path.is_dir():
        return report

    files = _all_files(path)
    report.total_files = len(files)
    report.file_extension_counts = dict(Counter(file.suffix.lower() for file in files))
    class_directories = {directory.name for directory in path.iterdir() if directory.is_dir()}
    report.missing_classes = sorted(set(expected) - class_directories)
    report.unexpected_classes = sorted(class_directories - set(expected))
    if not files:
        report.status = "dataset_empty"
        report.class_counts = {class_name: 0 for class_name in expected}
        return report

    size_values: list[tuple[int, int]] = []
    for file in files:
        relative_parts = file.relative_to(path).parts
        class_name = relative_parts[0] if len(relative_parts) > 1 else None
        extension = file.suffix.lower()
        if extension not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        try:
            dimensions = _image_dimensions(file)
            sample = ImageSample(file, class_name, dimensions, extension, True)
            size_values.append(dimensions)
            report.valid_images += 1
            if class_name is not None:
                report.class_counts[class_name] = report.class_counts.get(class_name, 0) + 1
        except ValueError as error:
            sample = ImageSample(file, class_name, None, extension, False, str(error))
            report.invalid_images += 1
        report.samples.append(sample)

    report.class_counts = {
        class_name: report.class_counts.get(class_name, 0)
        for class_name in sorted(set(expected) | set(report.class_counts) | set(report.unexpected_classes))
    }
    if size_values:
        widths, heights = zip(*size_values)
        report.image_size_statistics = {
            "count": len(size_values),
            "width_min": min(widths),
            "width_max": max(widths),
            "height_min": min(heights),
            "height_max": max(heights),
            "unique_sizes": sorted(set(size_values)),
        }
    report.status = "inspected"
    return report


def _pair_key(path: Path, root: Path, mask_suffix: str) -> str:
    """Build a relative pairing key without an image or mask suffix."""
    stem = path.stem
    if stem.endswith(mask_suffix):
        stem = stem[: -len(mask_suffix)]
    return (path.relative_to(root).parent / stem).as_posix()


def inspect_segmentation_dataset(
    dataset_path: str | Path,
    mask_suffix: str = "_mask",
) -> SegmentationReport:
    """Inspect deterministic MRI/mask pairs and basic mask statistics."""
    path = Path(dataset_path)
    report = SegmentationReport(path, "dataset_not_present", mask_suffix)
    if not path.is_dir():
        return report

    files = [file for file in _all_files(path) if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    image_files: dict[str, list[Path]] = defaultdict(list)
    mask_files: dict[str, list[Path]] = defaultdict(list)
    for file in files:
        key = _pair_key(file, path, mask_suffix)
        (mask_files if file.stem.endswith(mask_suffix) else image_files)[key].append(file)

    report.total_images = sum(len(files) for files in image_files.values())
    report.total_masks = sum(len(files) for files in mask_files.values())
    keys = sorted(set(image_files) | set(mask_files))
    for key in keys:
        images = image_files.get(key, [])
        masks = mask_files.get(key, [])
        if len(images) > 1 or len(masks) > 1:
            report.duplicate_conflicts.append(key)
        image_path = images[0] if images else None
        mask_path = masks[0] if masks else None
        if image_path is None:
            report.masks_without_images.append(key)
        if mask_path is None:
            report.images_without_masks.append(key)
        pair = _inspect_pair(key, image_path, mask_path)
        report.pairs.append(pair)
        if pair.valid:
            report.valid_pairs += 1
            report.total_mask_pixels += pair.mask_pixels
            report.foreground_pixels += pair.foreground_pixels
            report.background_pixels += pair.mask_pixels - pair.foreground_pixels
            if pair.foreground_pixels:
                report.non_empty_mask_count += 1
            else:
                report.empty_mask_count += 1
        else:
            report.invalid_pairs += 1

    if report.total_mask_pixels:
        report.foreground_percentage = 100 * report.foreground_pixels / report.total_mask_pixels
    report.status = "dataset_empty" if not files else "inspected"
    return report


def _inspect_pair(key: str, image_path: Path | None, mask_path: Path | None) -> SegmentationPair:
    """Validate one image-mask pair and summarize its mask."""
    image_dimensions = mask_dimensions = None
    values: tuple[int, ...] = ()
    mask_pixels = foreground_pixels = 0
    errors: list[str] = []
    if image_path is not None:
        try:
            image_dimensions = _image_dimensions(image_path)
        except ValueError as error:
            errors.append(f"image: {error}")
    else:
        errors.append("image missing")
    mask_is_binary: bool | None = None
    if mask_path is not None:
        try:
            mask_dimensions = _image_dimensions(mask_path)
            with Image.open(mask_path) as mask_image:
                mask_array = np.asarray(mask_image.convert("L"))
            values = tuple(int(value) for value in np.unique(mask_array))
            mask_is_binary = set(values).issubset({0, 1, 255})
            mask_pixels = int(mask_array.size)
            foreground_pixels = int(np.count_nonzero(mask_array))
            if not mask_is_binary:
                errors.append(f"mask has unexpected values: {values}")
        except ValueError as error:
            errors.append(f"mask: {error}")
    else:
        errors.append("mask missing")
    if image_dimensions and mask_dimensions and image_dimensions != mask_dimensions:
        errors.append("image and mask dimensions differ")
    valid = not errors
    return SegmentationPair(
        key, image_path, mask_path, image_dimensions, mask_dimensions, valid,
        mask_is_binary, values, mask_pixels, foreground_pixels, "; ".join(errors) or None,
    )


@dataclass(frozen=True)
class DatasetSplits(Generic[T]):
    """Deterministic train, validation, and test partitions."""

    train: tuple[T, ...]
    validation: tuple[T, ...]
    test: tuple[T, ...]

    def summary(self) -> dict[str, int]:
        """Return split counts."""
        return {"train": len(self.train), "validation": len(self.validation), "test": len(self.test)}


def split_dataset(
    samples: Sequence[T],
    seed: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    labels: Sequence[str] | None = None,
) -> DatasetSplits[T]:
    """Split samples reproducibly, optionally distributing labels across splits."""
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must leave a positive test split")
    if labels is not None and len(labels) != len(samples):
        raise ValueError("labels must have the same length as samples")
    rng = random.Random(seed)
    indexed_groups: dict[str, list[int]] = defaultdict(list)
    if labels is None:
        indexed_groups["all"] = list(range(len(samples)))
    else:
        for index, label in enumerate(labels):
            indexed_groups[str(label)].append(index)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for group in sorted(indexed_groups):
        indices = indexed_groups[group]
        rng.shuffle(indices)
        train_end = round(len(indices) * train_ratio)
        validation_end = train_end + round(len(indices) * validation_ratio)
        train_indices.extend(indices[:train_end])
        validation_indices.extend(indices[train_end:validation_end])
        test_indices.extend(indices[validation_end:])
    for indices in (train_indices, validation_indices, test_indices):
        indices.sort()
    return DatasetSplits(
        tuple(samples[index] for index in train_indices),
        tuple(samples[index] for index in validation_indices),
        tuple(samples[index] for index in test_indices),
    )


def dataset_summary(
    classification: ClassificationReport,
    segmentation: SegmentationReport,
    splits: DatasetSplits[Any] | None = None,
    seed: int | None = None,
) -> str:
    """Create a human-readable summary of inspected data and optional splits."""
    lines = [
        "Classification:",
        f"  path: {classification.dataset_path}",
        f"  status: {classification.status}",
        f"  classes: {', '.join(classification.expected_classes)}",
        f"  total samples: {classification.valid_images}",
        f"  class counts: {classification.class_counts}",
        f"  valid/invalid files: {classification.valid_images}/{classification.invalid_images}",
        "Segmentation:",
        f"  path: {segmentation.dataset_path}",
        f"  status: {segmentation.status}",
        f"  image-mask pairs: {segmentation.valid_pairs}/{len(segmentation.pairs)} valid",
        f"  empty/non-empty masks: {segmentation.empty_mask_count}/{segmentation.non_empty_mask_count}",
    ]
    if not classification.dataset_path.is_dir() and not segmentation.dataset_path.is_dir():
        lines.append("Dataset not present - inspection deferred until dataset is supplied.")
    if splits is not None:
        lines.append(f"Splits: {splits.summary()}")
    if seed is not None:
        lines.append(f"Random seed: {seed}")
    return "\n".join(lines)