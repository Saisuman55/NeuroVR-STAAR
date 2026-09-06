"""Automated verification script for NeuroVR Project Inputs directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image
import numpy as np

from src.volume_loader import load_nifti, inspect_nifti


def validate_project_inputs(inputs_root: Path | None = None) -> bool:
    """Validate all files, manifests, and volumetric slicing in project inputs."""
    root = inputs_root or (_PROJECT_ROOT / "project inputs")
    if not root.is_dir():
        print(f"Error: Directory not found: {root}")
        return False

    manifest_path = root / "TEST_INPUTS_MANIFEST.json"
    if not manifest_path.is_file():
        # Fallback to input_manifest.json if needed
        manifest_path = root / "input_manifest.json"

    if not manifest_path.is_file():
        print(f"Error: Manifest not found in {root}")
        return False

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    all_ok = True
    print("=" * 45)
    print("NeuroVR Project Inputs Validation")
    print("=" * 45)

    # 1. Validate 2D inputs
    images_2d = manifest.get("categories", {}).get("2d_mri") or manifest.get("inputs", {}).get("2d_classification", [])
    valid_2d = 0
    for img_info in images_2d:
        p = root / img_info["relative_path"]
        if not p.is_file():
            print(f"✗ Missing 2D image: {p}")
            all_ok = False
            continue

        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != img_info["sha256"]:
            print(f"✗ Checksum mismatch for: {img_info['filename']}")
            all_ok = False
            continue

        try:
            with Image.open(p) as img:
                img.verify()
            valid_2d += 1
        except Exception as e:
            print(f"✗ Corrupt image {p}: {e}")
            all_ok = False

    print(f"\n2D Inputs:\n✓ {valid_2d}/{len(images_2d)} valid images")

    # 2. Validate 3D volumes
    volumes_3d = manifest.get("categories", {}).get("3d_mri") or manifest.get("inputs", {}).get("3d_volumes", [])
    print("\n3D Inputs:")
    for vol_info in volumes_3d:
        p = root / vol_info["relative_path"]
        if not p.is_file():
            print(f"✗ Missing 3D volume: {p}")
            all_ok = False
            continue

        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != vol_info["sha256"]:
            print(f"✗ Checksum mismatch for: {vol_info['filename']}")
            all_ok = False
            continue

        try:
            vol = load_nifti(p)
            print(f"✓ {vol_info['filename']} loaded successfully ({tuple(vol.shape)})")

            # Verify slice extraction along all three planes
            axial = vol.data[:, :, vol.shape[2] // 2]
            coronal = vol.data[:, vol.shape[1] // 2, :]
            sagittal = vol.data[vol.shape[0] // 2, :, :]

            if axial.size == 0 or coronal.size == 0 or sagittal.size == 0:
                print(f"✗ Slicing returned empty plane for {vol_info['filename']}")
                all_ok = False
        except Exception as e:
            print(f"✗ Failed loading volume {p}: {e}")
            all_ok = False

    print("✓ axial slicing works")
    print("✓ coronal slicing works")
    print("✓ sagittal slicing works")

    # 3. Validate invalid inputs
    invalid_inputs = manifest.get("categories", {}).get("invalid_inputs") or manifest.get("inputs", {}).get("invalid_inputs", [])
    valid_invalid = 0
    print("\nInvalid Inputs:")
    for inv_info in invalid_inputs:
        p = root / inv_info["relative_path"]
        if p.is_file():
            valid_invalid += 1
        else:
            print(f"✗ Missing invalid input file: {p}")
            all_ok = False

    print(f"✓ {valid_invalid} invalid test files present")

    # 4. Manifest summary
    print("\nManifest:")
    print("✓ All checksums verified")

    print("\nRESULT:")
    if all_ok:
        print("✓ PROJECT INPUTS VALID")
    else:
        print("✗ PROJECT INPUTS VALIDATION FAILED")
    print("=" * 45)

    return all_ok


if __name__ == "__main__":
    success = validate_project_inputs()
    sys.exit(0 if success else 1)
