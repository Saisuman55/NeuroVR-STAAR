"""Generate actual lightweight dataset reports for NeuroVR.

Usage (from project root):
  .venv/bin/python scripts/dataset_report.py
  .venv/bin/python scripts/dataset_report.py --classification
  .venv/bin/python scripts/dataset_report.py --segmentation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when the script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image, UnidentifiedImageError

from src.utils import load_config


def classification_report(root: Path) -> dict[str, Any]:
    """Inspect actual supported classification images recursively."""
    records: list[dict[str, Any]] = []
    for file in (sorted(root.rglob("*")) if root.is_dir() else []):
        if not file.is_file() or file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parts = file.relative_to(root).parts
        split = (
            parts[0]
            if len(parts) > 1 and parts[0].lower() in {"training", "testing", "train", "test"}
            else "unspecified"
        )
        class_name = parts[-2] if len(parts) > 1 else "unspecified"
        record: dict[str, Any] = {
            "path": str(file.relative_to(root)),
            "split": split,
            "class": class_name,
            "valid": False,
        }
        try:
            with Image.open(file) as image:
                image.verify()
            with Image.open(file) as image:
                record.update({
                    "valid": True,
                    "dimensions": [image.width, image.height],
                    "mode": image.mode,
                })
        except (OSError, UnidentifiedImageError) as error:
            record["error"] = str(error)
        records.append(record)

    valid = [r for r in records if r["valid"]]
    dim_counter: Counter = Counter(
        tuple(r["dimensions"]) for r in valid
    )
    return {
        "path": str(root),
        "exists": root.is_dir(),
        "total_images": len(records),
        "valid_images": len(valid),
        "invalid_images": len(records) - len(valid),
        "class_counts": dict(Counter(r["class"] for r in valid)),
        "split_counts": dict(Counter(r["split"] for r in valid)),
        "dimensions": {f"{w}x{h}": c for (w, h), c in dim_counter.items()},
        "records": records,
    }


def segmentation_report(brats_root: Path) -> dict[str, Any]:
    """Read the BraTS manifest and report actual subject/label/shape statistics.

    Falls back gracefully when no data has been imported yet.
    """
    # Prefer the top-level brats_manifest.json written by the importer.
    top_manifest = brats_root.parent / "brats_manifest.json"
    manifest_path = top_manifest if top_manifest.is_file() else brats_root / "dataset_manifest.json"
    splits_path = brats_root / "splits.json"

    if not manifest_path.is_file():
        return {
            "path": str(brats_root),
            "exists": brats_root.is_dir(),
            "status": "not_imported",
            "message": "BraTS data not imported.",
            "discovered_subjects": 0,
            "valid_subjects": 0,
            "invalid_subjects": 0,
            "modality_availability": {},
            "shape_distribution": {},
            "spacing_distribution": {},
            "segmentation_label_distribution": {},
            "subjects_with_tumor": 0,
            "split_counts": {},
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_data = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.is_file() else {}

    return {
        "path": str(brats_root),
        "exists": brats_root.is_dir(),
        "status": "ready" if manifest.get("valid_subjects", 0) > 0 else "not_imported",
        "discovered_subjects": manifest.get("discovered_subjects", 0),
        "valid_subjects": manifest.get("valid_subjects", 0),
        "invalid_subjects": manifest.get("invalid_subjects", 0),
        "modality_availability": manifest.get("modality_availability", {}),
        "shape_distribution": manifest.get("shape_distribution", {}),
        "spacing_distribution": manifest.get("spacing_distribution", {}),
        "segmentation_label_distribution": manifest.get("segmentation_label_distribution", {}),
        "subjects_with_tumor": manifest.get("subjects_with_tumor", 0),
        "split_counts": {
            k: len(v) for k, v in split_data.items() if k.endswith("_subjects")
        },
    }


def generate_report(config: dict[str, Any]) -> dict[str, Any]:
    """Generate a combined report from files currently present on disk."""
    brats_root = Path(config["paths"]["segmentation_dataset"]) / "brats"
    return {
        "classification": classification_report(
            Path(config["paths"]["classification_dataset"])
        ),
        "segmentation": segmentation_report(brats_root),
        "generated_by": "NeuroVR",
        "statistics_source": "filesystem inspection; no values fabricated",
    }


def main() -> int:
    """Write JSON and text reports to outputs/."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification", action="store_true",
        help="Report only the 2D classification dataset.",
    )
    parser.add_argument(
        "--segmentation", action="store_true",
        help="Report only the 3D segmentation dataset.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Reserved for future expensive volume statistics.",
    )
    args = parser.parse_args()
    config = load_config()
    output = Path(config["output"]["directory"])
    output.mkdir(parents=True, exist_ok=True)
    brats_root = Path(config["paths"]["segmentation_dataset"]) / "brats"

    if args.classification and not args.segmentation:
        report = {
            "classification": classification_report(
                Path(config["paths"]["classification_dataset"])
            ),
            "generated_by": "NeuroVR",
            "statistics_source": "filesystem inspection; no values fabricated",
        }
        c = report["classification"]
        summary = [
            "NeuroVR classification dataset report",
            f"Classification images: {c['valid_images']} valid / {c['total_images']} discovered",
            f"Invalid images: {c['invalid_images']}",
            f"Class counts: {c['class_counts']}",
            f"Split counts: {c['split_counts']}",
        ]

    elif args.segmentation and not args.classification:
        seg = segmentation_report(brats_root)
        report = {
            "segmentation": seg,
            "generated_by": "NeuroVR",
            "statistics_source": "filesystem inspection; no values fabricated",
        }
        if seg["valid_subjects"] == 0:
            summary = [
                "NeuroVR segmentation dataset report",
                "BraTS data not imported.",
                "3D subjects: 0",
            ]
        else:
            summary = [
                "NeuroVR segmentation dataset report",
                f"Discovered subjects: {seg['discovered_subjects']}",
                f"Valid subjects: {seg['valid_subjects']}",
                f"Invalid subjects: {seg['invalid_subjects']}",
                f"Subjects with tumor: {seg['subjects_with_tumor']}",
                f"Modality availability: {seg['modality_availability']}",
                f"Split counts: {seg['split_counts']}",
            ]

    else:
        report = generate_report(config)
        c = report["classification"]
        seg = report["segmentation"]
        summary = [
            "NeuroVR dataset report",
            f"Classification images: {c['valid_images']} valid / {c['total_images']} discovered",
            f"3D subjects: {seg['valid_subjects']} valid / {seg['discovered_subjects']} discovered",
        ]

    (output / "dataset_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "dataset_report.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
