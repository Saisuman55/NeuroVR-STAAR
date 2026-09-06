"""Import a user-supplied, legally obtained BraTS-style dataset.

Official access route:
  BraTS 2021 — https://www.med.upenn.edu/cbica/brats2021/
  Registration, authentication, and the applicable release terms are required.
  This script does not bypass any access control, CAPTCHA, or license restriction.

Usage (from project root):
  .venv/bin/python scripts/import_brats_dataset.py --source /path/to/brats
  .venv/bin/python scripts/import_brats_dataset.py --source /path/to/brats --validate-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when the script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.volume_loader import inspect_nifti, load_nifti
from src.utils import load_config

# ---------------------------------------------------------------------------
# Modality detection
# t1ce MUST be checked before t1 to avoid substring false-positives.
# ---------------------------------------------------------------------------
MODALITY_ALIASES: dict[str, tuple[str, ...]] = {
    "flair": ("flair",),
    "t1ce":  ("t1ce", "t1gd", "t1_gd"),
    "t1":    ("t1",),
    "t2":    ("t2",),
    "seg":   ("seg", "mask", "label"),
}
REQUIRED_MODALITIES = ("flair", "t1", "t1ce", "t2", "seg")


def detect_modality(filename: str) -> str | None:
    """Detect one modality from a filename using token-aware matching.

    t1ce is checked before t1 so filenames containing 't1ce' or 't1gd' are
    never incorrectly matched as plain 't1'.  Matching is token-based: the
    stem is split on underscores and dots so that 't1' only matches when it
    appears as a complete token, not as a prefix of 't1ce'.
    """
    stem = filename.lower()
    # Strip known NIfTI extensions before tokenising.
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    tokens = set(stem.replace("-", "_").split("_"))
    for modality, aliases in MODALITY_ALIASES.items():
        if any(alias in tokens for alias in aliases):
            return modality
    return None


# ---------------------------------------------------------------------------
# Subject discovery
# ---------------------------------------------------------------------------

def discover_subjects(source: Path) -> dict[str, dict[str, Path]]:
    """Discover NIfTI modalities grouped by their source subject directory.

    Raises ValueError for duplicate modalities or duplicate subject IDs so
    that ambiguous data is never silently discarded.
    """
    subjects: dict[str, dict[str, Path]] = {}
    for subject_dir in sorted(item for item in source.rglob("*") if item.is_dir()):
        nifti_files = [
            f for f in sorted(subject_dir.iterdir())
            if f.is_file() and (
                f.name.lower().endswith(".nii") or f.name.lower().endswith(".nii.gz")
            )
        ]
        modalities: dict[str, Path] = {}
        for f in nifti_files:
            modality = detect_modality(f.name)
            if modality is not None:
                if modality in modalities:
                    raise ValueError(
                        f"Duplicate {modality} files in subject directory {subject_dir.name}"
                    )
                modalities[f] = modality  # type: ignore[assignment]
                modalities[modality] = f  # canonical key
        # Remove the Path-keyed entries we used as scratch
        modalities = {k: v for k, v in modalities.items() if isinstance(k, str)}
        if modalities:
            subject_id = subject_dir.name
            if subject_id in subjects:
                raise ValueError(f"Duplicate subject directory: {subject_id}")
            subjects[subject_id] = modalities
    return subjects


def _analyse_segmentation(path: Path) -> dict[str, Any]:
    """Read a segmentation volume and return actual label statistics.

    The original file is never modified.  Labels are reported as found;
    NeuroVR's binary conversion (tumor = all non-zero labels) is documented
    separately and applied only at training time.
    """
    try:
        vol = load_nifti(path)
        data = np.asarray(vol.data)
        unique, counts = np.unique(data, return_counts=True)
        label_counts = {int(label): int(count) for label, count in zip(unique, counts)}
        background = int(label_counts.get(0, 0))
        tumor_voxels = int(sum(c for lbl, c in label_counts.items() if lbl != 0))
        return {
            "readable": True,
            "shape": list(vol.shape),
            "dtype": vol.dtype,
            "unique_labels": [int(v) for v in unique],
            "label_counts": label_counts,
            "background_voxels": background,
            "tumor_voxels": tumor_voxels,
            "has_tumor": tumor_voxels > 0,
        }
    except Exception as exc:
        return {"readable": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Per-subject validation
# ---------------------------------------------------------------------------

def validate_subject(subject_id: str, modalities: dict[str, Path]) -> dict[str, Any]:
    """Validate all available modalities and cross-volume spatial consistency."""
    record: dict[str, Any] = {
        "subject_id": subject_id,
        "modalities": {},
        "missing_modalities": [],
        "errors": [],
        "valid": False,
    }
    valid_volumes: dict[str, Any] = {}

    for modality, path in modalities.items():
        inspection = inspect_nifti(path)
        record["modalities"][modality] = {
            "source": str(path),
            "valid": inspection.valid,
            "shape": list(inspection.shape) if inspection.shape else None,
            "spacing": list(inspection.spacing) if inspection.spacing else None,
            "dtype": inspection.dtype,
            "messages": list(inspection.messages),
        }
        if inspection.valid:
            valid_volumes[modality] = load_nifti(path)

    record["missing_modalities"] = [
        m for m in REQUIRED_MODALITIES if m not in modalities
    ]

    # Cross-modality spatial consistency (shape + affine).
    if valid_volumes:
        reference = valid_volumes.get("flair") or next(iter(valid_volumes.values()))
        record["shape"] = list(reference.shape)
        record["spacing"] = list(reference.spacing)
        for modality, volume in valid_volumes.items():
            if volume.shape != reference.shape:
                record["errors"].append(
                    f"{modality} shape {volume.shape} differs from reference {reference.shape}"
                )
            if not np.allclose(volume.affine, reference.affine, atol=1e-4):
                record["errors"].append(
                    f"{modality} affine differs from reference"
                )

    # Segmentation label analysis (non-destructive).
    if "seg" in modalities:
        record["segmentation_analysis"] = _analyse_segmentation(modalities["seg"])

    record["valid"] = (
        not record["missing_modalities"]
        and not record["errors"]
        and len(valid_volumes) == len(REQUIRED_MODALITIES)
    )
    return record


# ---------------------------------------------------------------------------
# Dataset import
# ---------------------------------------------------------------------------

def import_dataset(source: Path, destination: Path) -> dict[str, Any]:
    """Validate and copy a BraTS-style dataset into standardised subject folders."""
    if not source.is_dir():
        raise FileNotFoundError(f"BraTS source directory not found: {source}")

    subjects = discover_subjects(source)
    records = [
        validate_subject(sid, mods)
        for sid, mods in sorted(subjects.items())
    ]

    destination.mkdir(parents=True, exist_ok=True)
    for record in records:
        if not record["valid"]:
            continue
        subject_id = record["subject_id"]
        subject_dest = destination / subject_id
        if subject_dest.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing subject: {subject_dest}"
            )
        subject_dest.mkdir()
        for modality, details in record["modalities"].items():
            src_path = Path(details["source"])
            # Preserve the original .nii or .nii.gz extension.
            suffix = ".nii.gz" if src_path.name.lower().endswith(".nii.gz") else ".nii"
            shutil.copy2(src_path, subject_dest / f"{subject_id}_{modality}{suffix}")

    valid_count = sum(r["valid"] for r in records)
    return {
        "source": str(source),
        "destination": str(destination),
        "subjects": records,
        "subject_count": len(records),
        "valid_subject_count": valid_count,
    }


# ---------------------------------------------------------------------------
# Subject-level splits (no slice-level leakage)
# ---------------------------------------------------------------------------

def _split_subjects(subjects: list[str], seed: int) -> dict[str, Any]:
    """Create deterministic subject-level splits with verified no-overlap.

    All volumes from one subject remain in exactly one partition.  Splitting
    individual slices across partitions would constitute data leakage and is
    explicitly prohibited by the NeuroVR data policy.
    """
    import random

    shuffled = subjects[:]
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * 0.70)
    val_end   = train_end + round(len(shuffled) * 0.15)
    result = {
        "train_subjects":      sorted(shuffled[:train_end]),
        "validation_subjects": sorted(shuffled[train_end:val_end]),
        "test_subjects":       sorted(shuffled[val_end:]),
        "seed": seed,
        "split_policy": "subject-level; no slice-level leakage",
    }
    sets = [set(result[k]) for k in ("train_subjects", "validation_subjects", "test_subjects")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("Subject leakage detected while creating splits — aborting.")
    return result


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _build_manifest(report: dict[str, Any]) -> dict[str, Any]:
    """Build the dataset-level manifest from an import report."""
    records = report["subjects"]
    valid_records = [r for r in records if r["valid"]]

    # Shape distribution across valid subjects.
    shape_counter: Counter = Counter(
        tuple(r["shape"]) for r in valid_records if "shape" in r
    )
    spacing_counter: Counter = Counter(
        tuple(round(s, 4) for s in r["spacing"])
        for r in valid_records if "spacing" in r
    )

    # Segmentation label distribution.
    all_labels: Counter = Counter()
    non_empty_seg = 0
    for r in valid_records:
        seg = r.get("segmentation_analysis", {})
        if seg.get("readable"):
            for lbl, cnt in seg.get("label_counts", {}).items():
                all_labels[int(lbl)] += int(cnt)
            if seg.get("has_tumor"):
                non_empty_seg += 1

    # Modality availability across ALL discovered subjects.
    modality_availability = {
        m: sum(1 for r in records if m in r.get("modalities", {}))
        for m in REQUIRED_MODALITIES
    }

    return {
        "dataset": "BraTS-style user-supplied dataset",
        "official_access": "https://www.med.upenn.edu/cbica/brats2021/",
        "access_note": (
            "Registration, authentication, and the applicable BraTS release terms "
            "are required. NeuroVR does not bypass any access control."
        ),
        "statistics_source": "filesystem inspection; no values fabricated",
        "discovered_subjects": len(records),
        "valid_subjects": report["valid_subject_count"],
        "invalid_subjects": len(records) - report["valid_subject_count"],
        "modality_availability": modality_availability,
        "shape_distribution": {str(list(k)): v for k, v in shape_counter.items()},
        "spacing_distribution": {str(list(k)): v for k, v in spacing_counter.items()},
        "segmentation_label_distribution": {str(k): v for k, v in sorted(all_labels.items())},
        "subjects_with_tumor": non_empty_seg,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse source, validate, copy, write manifest and splits."""
    parser = argparse.ArgumentParser(
        description="Import a legally obtained BraTS-style dataset into NeuroVR."
    )
    parser.add_argument(
        "--source", required=True, type=Path,
        help="Path to the root of the downloaded BraTS dataset.",
    )
    parser.add_argument(
        "--destination", type=Path, default=Path("data/segmentation/brats"),
        help="Where to copy the standardised subject folders.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for subject-level splits (default: from config).",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Validate without copying files or writing manifests.",
    )
    args = parser.parse_args()

    config = load_config()
    seed = args.seed if args.seed is not None else config["project"]["random_seed"]

    if args.validate_only:
        if not args.source.is_dir():
            print(f"Source directory not found: {args.source}")
            return 1
        subjects = discover_subjects(args.source)
        records = [validate_subject(sid, mods) for sid, mods in sorted(subjects.items())]
        valid = sum(r["valid"] for r in records)
        print(json.dumps({
            "discovered": len(records),
            "valid": valid,
            "invalid": len(records) - valid,
        }, indent=2))
        return 0

    report = import_dataset(args.source, args.destination)
    manifest = _build_manifest(report)

    # Write manifest alongside the subject folders.
    manifest_path = args.destination / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Also write a top-level manifest at data/segmentation/ for the web API.
    top_manifest = args.destination.parent / "brats_manifest.json"
    top_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Subject-level splits.
    valid_subjects = sorted(
        r["subject_id"] for r in report["subjects"] if r["valid"]
    )
    splits = _split_subjects(valid_subjects, seed)
    splits_path = args.destination / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

    summary = {
        "subject_count": report["subject_count"],
        "valid_subject_count": report["valid_subject_count"],
        "manifest": str(manifest_path),
        "splits": str(splits_path),
        "split_counts": {
            "train": len(splits["train_subjects"]),
            "validation": len(splits["validation_subjects"]),
            "test": len(splits["test_subjects"]),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
