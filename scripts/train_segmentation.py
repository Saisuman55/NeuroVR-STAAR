"""3D Brain Tumor Segmentation Training Entry Point for NeuroVR.

This script enforces academic honesty: it checks for verified, imported BraTS data
and exits gracefully when benchmark data has not yet been imported. Reference demonstration
volumes (e.g. anatomical.nii, mni152.nii.gz) are never treated as segmentation training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config, setup_logging


def check_brats_data_readiness(config: dict) -> tuple[bool, str, list[dict]]:
    """Inspect segmentation manifests in priority order."""
    seg_root = Path(config["paths"]["segmentation_dataset"])
    top_manifest = seg_root / "brats_manifest.json"
    legacy_manifest = seg_root / "brats" / "dataset_manifest.json"

    manifest_path = None
    if top_manifest.is_file():
        manifest_path = top_manifest
    elif legacy_manifest.is_file():
        manifest_path = legacy_manifest

    if manifest_path is None:
        return False, f"No BraTS manifest found at {top_manifest} or {legacy_manifest}", []

    try:
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as err:
        return False, f"Failed to parse manifest at {manifest_path}: {err}", []

    subjects = data.get("report", {}).get("subjects", [])
    valid_subjects = [s for s in subjects if s.get("valid", False)]
    if not valid_subjects:
        return False, "BraTS dataset manifest contains 0 valid subjects.", []

    return True, f"Found {len(valid_subjects)} valid BraTS subjects.", valid_subjects


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NeuroVR 3D Segmentation Training Entry Point"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to central configuration file",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check readiness and exit without starting training",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config)

    logger.info("Checking 3D segmentation training readiness...")
    is_ready, message, subjects = check_brats_data_readiness(config)

    if not is_ready:
        print("=" * 60)
        print("NeuroVR 3D Segmentation Training Status:")
        print(f"Status: NOT READY ({message})")
        print("BraTS dataset not imported. Segmentation training cannot start.")
        print("To import real BraTS data legitimately, run:")
        print("  .venv/bin/python scripts/import_brats_dataset.py --source /path/to/brats")
        print("=" * 60)
        return 0

    print(f"BraTS dataset ready: {message}")
    if args.check_only:
        return 0

    # In future steps with real data imported: launch 3D U-Net training loop
    return 0


if __name__ == "__main__":
    sys.exit(main())
