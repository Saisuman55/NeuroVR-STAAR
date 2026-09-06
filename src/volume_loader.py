"""Read-only NIfTI loading and validation utilities for NeuroVR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from nibabel.orientations import aff2axcodes


NIFTI_EXTENSIONS = frozenset({".nii", ".nii.gz"})


@dataclass(frozen=True)
class NiftiVolume:
    """A loaded volume and the metadata needed for spatial interpretation."""

    data: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, float, float]
    shape: tuple[int, ...]
    dtype: str
    orientation: tuple[str | None, ...]
    header: Any
    path: Path


@dataclass(frozen=True)
class NiftiInspection:
    """Validation and intensity facts about one NIfTI file."""

    path: Path
    valid: bool
    status: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    spacing: tuple[float, float, float] | None = None
    orientation: tuple[str | None, ...] | None = None
    has_nan: bool = False
    has_inf: bool = False
    is_constant: bool = False
    is_empty: bool = False
    intensity_statistics: dict[str, float] | None = None
    nan_count: int = 0
    inf_count: int = 0
    messages: tuple[str, ...] = ()


def _validate_path(path: str | Path) -> Path:
    """Validate a path and return it as a Path."""
    candidate = Path(path)
    if candidate.suffix.lower() == ".gz" and candidate.name.lower().endswith(".nii.gz"):
        return candidate
    if candidate.suffix.lower() != ".nii":
        raise ValueError(f"Unsupported NIfTI extension: {candidate.name}")
    return candidate


def _validate_affine(affine: np.ndarray) -> None:
    """Raise when an affine is not a finite, invertible 4x4 transform."""
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError("NIfTI affine must be a finite 4x4 matrix")
    if not np.isfinite(np.linalg.det(affine[:3, :3])) or np.isclose(np.linalg.det(affine[:3, :3]), 0):
        raise ValueError("NIfTI affine has a singular spatial transform")


def get_voxel_spacing(volume: NiftiVolume | nib.spatialimages.SpatialImage) -> tuple[float, float, float]:
    """Return positive voxel spacing in millimeters from an affine or image."""
    affine = volume.affine
    if affine is None:
        raise ValueError("NIfTI image has no affine")
    _validate_affine(np.asarray(affine))
    spacing = tuple(float(value) for value in np.linalg.norm(np.asarray(affine)[:3, :3], axis=0))
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError(f"Invalid voxel spacing: {spacing}")
    return spacing


def get_orientation(volume: NiftiVolume | nib.spatialimages.SpatialImage) -> tuple[str | None, ...]:
    """Return axis orientation codes without reorienting the volume."""
    affine = np.asarray(volume.affine)
    _validate_affine(affine)
    return tuple(code if code is not None else None for code in aff2axcodes(affine))


def load_nifti(path: str | Path) -> NiftiVolume:
    """Load a NIfTI volume while preserving data, affine, header, and orientation."""
    nifti_path = _validate_path(path)
    if not nifti_path.is_file():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    try:
        image = nib.load(str(nifti_path))
        data = np.asanyarray(image.dataobj)
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
        raise ValueError(f"Invalid NIfTI file: {nifti_path}") from error
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume, found {data.ndim}D shape {data.shape}")
    affine = np.asarray(image.affine, dtype=np.float64).copy()
    _validate_affine(affine)
    return NiftiVolume(
        data=np.asarray(data),
        affine=affine,
        spacing=get_voxel_spacing(image),
        shape=tuple(data.shape),
        dtype=str(data.dtype),
        orientation=get_orientation(image),
        header=image.header.copy(),
        path=nifti_path,
    )


def inspect_nifti(
    path: str | Path,
    percentiles: Sequence[float] = (1.0, 50.0, 99.0),
) -> NiftiInspection:
    """Inspect a NIfTI file and return clear validation and intensity messages."""
    try:
        volume = load_nifti(path)
    except (FileNotFoundError, ValueError) as error:
        return NiftiInspection(Path(path), False, "invalid", messages=(str(error),))
    data = np.asarray(volume.data)
    has_nan = bool(np.issubdtype(data.dtype, np.inexact) and np.isnan(data).any())
    has_inf = bool(np.issubdtype(data.dtype, np.inexact) and np.isinf(data).any())
    nan_count = int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.inexact) else 0
    inf_count = int(np.isinf(data).sum()) if np.issubdtype(data.dtype, np.inexact) else 0
    finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.inexact) else data
    if not percentiles or any(value < 0 or value > 100 for value in percentiles):
        raise ValueError("percentiles must contain values between 0 and 100")
    stats = {"min": float(np.min(finite)), "max": float(np.max(finite)),
             "mean": float(np.mean(finite)), "std": float(np.std(finite)),
             "median": float(np.median(finite))} if finite.size else None
    if finite.size:
        stats.update({f"percentile_{value:g}": float(np.percentile(finite, value)) for value in percentiles})
    is_constant = bool(finite.size and np.all(finite == finite.flat[0]))
    is_empty = bool(finite.size == 0 or not np.any(finite))
    messages = tuple(
        message for message, condition in (
            ("volume contains NaN values", has_nan),
            ("volume contains infinite values", has_inf),
            ("volume is constant", is_constant),
            ("volume is empty (all zero or non-finite)", is_empty),
        ) if condition
    )
    return NiftiInspection(
        volume.path, not (has_nan or has_inf), "valid", volume.shape,
        volume.dtype, volume.spacing, volume.orientation, has_nan, has_inf,
        is_constant, is_empty, stats, nan_count, inf_count, messages,
    )


def validate_nifti(path: str | Path) -> tuple[bool, tuple[str, ...]]:
    """Return whether a NIfTI is usable as a finite 3D volume and why."""
    inspection = inspect_nifti(path)
    return inspection.valid, inspection.messages


def voxel_to_physical(volume: NiftiVolume, coordinates: Sequence[float]) -> np.ndarray:
    """Convert a voxel coordinate to physical RAS+ coordinates using the affine."""
    if len(coordinates) != 3:
        raise ValueError("Voxel coordinates must contain exactly three values")
    return np.asarray(apply_affine(volume.affine, coordinates), dtype=np.float64)


def physical_to_voxel(volume: NiftiVolume, coordinates: Sequence[float]) -> np.ndarray:
    """Convert physical RAS+ coordinates to voxel coordinates using affine inverse."""
    if len(coordinates) != 3:
        raise ValueError("Physical coordinates must contain exactly three values")
    return np.asarray(apply_affine(np.linalg.inv(volume.affine), coordinates), dtype=np.float64)