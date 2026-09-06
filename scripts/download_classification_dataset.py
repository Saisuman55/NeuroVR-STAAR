"""Download and validate the four-class Kaggle classification dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

EXPECTED_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
DATASET_ID = "masoudnickparvar/brain-tumor-mri-dataset"
SPLIT_NAMES = {"training", "testing", "train", "test", "valid", "validation"}


def _kaggle_executable() -> str | None:
    """Return the kaggle executable path, checking user-local Python bin dirs."""
    if shutil.which("kaggle"):
        return "kaggle"
    # Check common user-local install locations (e.g. pip install --user)
    for candidate in [
        Path.home() / "Library" / "Python" / "3.9" / "bin" / "kaggle",
        Path.home() / "Library" / "Python" / "3.10" / "bin" / "kaggle",
        Path.home() / "Library" / "Python" / "3.11" / "bin" / "kaggle",
        Path.home() / ".local" / "bin" / "kaggle",
    ]:
        if candidate.is_file():
            return str(candidate)
    return None


def credentials_available() -> bool:
    """Return whether standard Kaggle credentials are configured."""
    credential_file = Path.home() / ".kaggle" / "kaggle.json"
    return credential_file.is_file() or bool(
        os.getenv("KAGGLE_USERNAME") and (os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_TOKEN"))
    )


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting path traversal entries."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        zip_file.extractall(destination)


def discover_split_class_files(root: Path) -> dict[str, dict[str, list[Path]]]:
    """Find image files grouped by split (Training/Testing) and class.

    Handles two archive layouts:
      1. split/class/files  (e.g. Training/glioma/*.jpg)
      2. class/files        (flat, no split — assigned to Training)
    """
    result: dict[str, dict[str, list[Path]]] = {
        "Training": {c: [] for c in EXPECTED_CLASSES},
        "Testing": {c: [] for c in EXPECTED_CLASSES},
    }
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = [p.lower() for p in file.relative_to(root).parts[:-1]]
        split_part = next((p for p in parts if p in SPLIT_NAMES), None)
        class_part = next((p for p in parts if p in EXPECTED_CLASSES), None)
        if class_part is None:
            continue
        if split_part in {"testing", "test"}:
            split_key = "Testing"
        else:
            split_key = "Training"
        result[split_key][class_part].append(file)
    return result


def inspect_files(root: Path) -> dict[str, Any]:
    """Calculate actual image, format, corruption, and duplicate-name facts."""
    files = [f for f in sorted(root.rglob("*")) if f.is_file()]
    supported = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS]
    corrupted: list[str] = []
    dimensions: Counter[str] = Counter()
    names: Counter[str] = Counter()
    for file in supported:
        names[file.name] += 1
        try:
            with Image.open(file) as image:
                image.verify()
            with Image.open(file) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
        except (OSError, UnidentifiedImageError):
            corrupted.append(file.as_posix())
    splits = discover_split_class_files(root)
    images_per_class: dict[str, int] = {c: 0 for c in EXPECTED_CLASSES}
    split_counts: dict[str, int] = {}
    for split_name, classes in splits.items():
        count = sum(len(v) for v in classes.values())
        split_counts[split_name] = count
        for class_name, class_files in classes.items():
            images_per_class[class_name] = images_per_class.get(class_name, 0) + len(class_files)
    return {
        "total_files": len(files),
        "total_images": len(supported),
        "images_per_class": images_per_class,
        "split_counts": split_counts,
        "format_distribution": dict(Counter(f.suffix.lower() for f in supported)),
        "dimensions": dict(dimensions),
        "corrupted_files": corrupted,
        "duplicate_filenames": sorted(name for name, count in names.items() if count > 1),
        "expected_classes": list(EXPECTED_CLASSES),
    }


def create_manifest(destination: Path, source: Path, report: dict[str, Any]) -> None:
    """Write actual discovery information to the classification manifest."""
    manifest = {
        "dataset": "Brain Tumor MRI Dataset",
        "source": "Kaggle dataset dsv/2645886",
        "source_url": "https://www.kaggle.com/dsv/2645886",
        "source_directory": str(source),
        "local_directory": str(destination),
        "discovered_by_neurovr": report,
    }
    (destination / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Download, safely extract, validate, and stage the dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data/classification"))
    args = parser.parse_args()

    kaggle_exe = _kaggle_executable()
    if kaggle_exe is None:
        print("Kaggle CLI is not installed.")
        print("Install it:  pip install kaggle")
        print("Then configure credentials: https://www.kaggle.com/docs/api")
        return 2

    if not credentials_available():
        print("Kaggle credentials are missing.")
        print("Create ~/.kaggle/kaggle.json with your username and key.")
        print("Instructions: https://www.kaggle.com/docs/api")
        return 2

    args.destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="neurovr_kaggle_") as temporary:
        temporary_path = Path(temporary)
        command = [kaggle_exe, "datasets", "download", "-d", DATASET_ID, "-p", str(temporary_path)]
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"Kaggle download failed with exit code {result.returncode}.")
            return result.returncode

        archives = sorted(temporary_path.glob("*.zip"))
        if not archives:
            print("Kaggle download completed without an archive.")
            return 1

        extracted = temporary_path / "extracted"
        extracted.mkdir()
        safe_extract(archives[0], extracted)

        splits = discover_split_class_files(extracted)
        missing = [c for c in EXPECTED_CLASSES if not any(splits[s][c] for s in splits)]
        if missing:
            print(f"Expected classes missing from archive: {', '.join(missing)}")
            return 1

        for split_name, classes in splits.items():
            for class_name, files in classes.items():
                if not files:
                    continue
                class_destination = args.destination / split_name / class_name
                class_destination.mkdir(parents=True, exist_ok=True)
                for source in files:
                    target = class_destination / source.name
                    if target.exists():
                        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
                    shutil.copy2(source, target)

        final_report = inspect_files(args.destination)
        create_manifest(args.destination, extracted, final_report)
        print(json.dumps(final_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
