"""Lazy multimodal BraTS-style dataset for NeuroVR 3D segmentation.

Channel order (index → modality):
  0 = FLAIR
  1 = T1
  2 = T1ce / T1Gd
  3 = T2

This order is fixed and must be consistent across loading, preprocessing,
and model input.  The 3D U-Net expects [B, 4, D, H, W].

Segmentation target:
  [1, D, H, W]  — binary float32 mask (tumor=1, background=0)

Binary conversion policy:
  Original BraTS segmentation files may contain multiple integer labels
  (e.g. 0=background, 1=necrotic core, 2=oedema, 4=enhancing tumour).
  NeuroVR's baseline 3D U-Net uses a binary representation:
    tumor   = all non-zero labels
    background = label 0
  The original multi-class file is NEVER overwritten.  The conversion is
  applied in memory at load time only.

Data leakage policy:
  Splits are performed at subject level.  All volumes from one subject
  remain in exactly one partition (train / validation / test).  Splitting
  individual slices across partitions would constitute data leakage and is
  explicitly prohibited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.volume_loader import load_nifti
from src.volume_preprocessing import normalize_intensity, volume_to_tensor

# Canonical channel order — do not reorder without updating the model config.
MODALITIES: Tuple[str, ...] = ("flair", "t1", "t1ce", "t2")
CHANNEL_ORDER = {modality: idx for idx, modality in enumerate(MODALITIES)}


@dataclass(frozen=True)
class BrainTumorSubject:
    """Paths for one subject's four MRI modalities and optional segmentation.

    Channel assignment:
      flair  → channel 0
      t1     → channel 1
      t1ce   → channel 2
      t2     → channel 3
    """

    subject_id: str
    flair: Path
    t1: Path
    t1ce: Path
    t2: Path
    segmentation: Optional[Path] = None


def binary_mask_from_segmentation(seg_data: np.ndarray) -> np.ndarray:
    """Convert a multi-class BraTS segmentation to a binary tumor mask.

    All non-zero labels are mapped to 1 (tumor); label 0 remains 0 (background).
    The original segmentation array is not modified.

    This conversion is applied in memory only.  The source NIfTI file is
    never overwritten.
    """
    return (seg_data != 0).astype(np.float32)


class BrainTumor3DDataset(Dataset):
    """Load one subject at a time as [C, D, H, W] without caching the dataset.

    Memory safety: volumes are loaded lazily per __getitem__ call.  The full
    dataset is never held in RAM simultaneously.

    Args:
        subjects:       List of BrainTumorSubject records.
        normalization:  Intensity normalisation method (see volume_preprocessing).
        binary_mask:    If True, convert multi-class segmentation to binary
                        tumor/background mask.  Default True.
    """

    def __init__(
        self,
        subjects: list,
        normalization: str = "zscore_nonzero",
        binary_mask: bool = True,
    ) -> None:
        self.subjects = tuple(subjects)
        self.normalization = normalization
        self.binary_mask = binary_mask

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> Tuple[Tensor, Optional[Tensor]]:
        """Load and normalise the selected subject's modalities lazily.

        Returns:
            image: float32 tensor [4, D, H, W]
                   channels: 0=FLAIR, 1=T1, 2=T1ce, 3=T2
            mask:  float32 tensor [1, D, H, W] or None if no segmentation path.
        """
        subject = self.subjects[index]
        volumes = [load_nifti(getattr(subject, m)) for m in MODALITIES]
        reference = volumes[0]

        # Verify spatial alignment across all four modalities.
        for modality, volume in zip(MODALITIES[1:], volumes[1:]):
            if volume.shape != reference.shape:
                raise ValueError(
                    f"Subject {subject.subject_id}: {modality} shape "
                    f"{volume.shape} != flair shape {reference.shape}"
                )
            if not np.allclose(volume.affine, reference.affine, atol=1e-4):
                raise ValueError(
                    f"Subject {subject.subject_id}: {modality} affine "
                    "differs from flair affine"
                )

        channels = [
            volume_to_tensor(
                normalize_intensity(vol.data, self.normalization),
                add_channel=False,
            )
            for vol in volumes
        ]
        image = torch.stack(channels, dim=0)  # [4, D, H, W]

        mask: Optional[Tensor] = None
        if subject.segmentation is not None:
            seg_vol = load_nifti(subject.segmentation)
            if seg_vol.shape != reference.shape:
                raise ValueError(
                    f"Subject {subject.subject_id}: segmentation shape "
                    f"{seg_vol.shape} != flair shape {reference.shape}"
                )
            if not np.allclose(seg_vol.affine, reference.affine, atol=1e-4):
                raise ValueError(
                    f"Subject {subject.subject_id}: segmentation affine "
                    "differs from flair affine"
                )
            seg_data = (
                binary_mask_from_segmentation(np.asarray(seg_vol.data))
                if self.binary_mask
                else np.asarray(seg_vol.data, dtype=np.float32)
            )
            mask = volume_to_tensor(seg_data, dtype=torch.float32, add_channel=True)

        return image, mask


def subjects_from_directory(directory: str | Path) -> list[BrainTumorSubject]:
    """Discover standardised subject folders without loading their volumes.

    Expects the layout written by import_brats_dataset.py:
      <directory>/<subject_id>/<subject_id>_<modality>.nii.gz
    """
    root = Path(directory)
    subjects: list[BrainTumorSubject] = []
    for subject_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        # Build a stem→path map, stripping the .nii.gz double extension.
        files: dict[str, Path] = {}
        for path in subject_dir.iterdir():
            name = path.name.lower()
            if name.endswith(".nii.gz"):
                stem = name[:-7]
            elif name.endswith(".nii"):
                stem = name[:-4]
            else:
                continue
            files[stem] = path

        paths = {
            modality: next(
                (p for stem, p in files.items() if stem.endswith(f"_{modality}")),
                None,
            )
            for modality in MODALITIES
        }
        if all(paths.values()):
            segmentation = next(
                (p for stem, p in files.items() if stem.endswith("_seg")), None
            )
            subjects.append(
                BrainTumorSubject(
                    subject_dir.name,
                    paths["flair"],   # type: ignore[arg-type]
                    paths["t1"],      # type: ignore[arg-type]
                    paths["t1ce"],    # type: ignore[arg-type]
                    paths["t2"],      # type: ignore[arg-type]
                    segmentation,
                )
            )
    return subjects


def subjects_from_splits(
    brats_root: str | Path,
    split: str,
) -> list[BrainTumorSubject]:
    """Return subjects for one split partition using the saved splits.json.

    Args:
        brats_root: Path to data/segmentation/brats/
        split:      One of 'train', 'validation', 'test'

    Raises:
        FileNotFoundError: if splits.json does not exist.
        ValueError:        if split name is not recognised.
    """
    root = Path(brats_root)
    splits_path = root / "splits.json"
    if not splits_path.is_file():
        raise FileNotFoundError(f"splits.json not found at {splits_path}")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    key = f"{split}_subjects"
    if key not in splits:
        raise ValueError(
            f"Unknown split '{split}'. Available: "
            + ", ".join(k.replace("_subjects", "") for k in splits if k.endswith("_subjects"))
        )
    subject_ids = set(splits[key])
    all_subjects = subjects_from_directory(root)
    return [s for s in all_subjects if s.subject_id in subject_ids]
