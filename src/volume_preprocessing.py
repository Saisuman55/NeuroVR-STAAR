"""Configurable, metadata-aware preprocessing for 3D MRI volumes."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.ndimage import zoom

from src.volume_loader import NiftiVolume


def convert_intensity(data: np.ndarray, dtype: np.dtype = np.dtype("float32")) -> np.ndarray:
    """Convert image intensities to a floating-point array."""
    return np.asarray(data, dtype=dtype)


def handle_nan_inf(data: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Replace non-finite intensity values without changing array shape."""
    values = np.asarray(data, dtype=np.float32).copy()
    return np.nan_to_num(values, nan=fill_value, posinf=fill_value, neginf=fill_value)


def clip_intensity(data: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    """Clip intensities to configured percentiles, preserving the input shape."""
    lower, upper = percentiles
    if not 0 <= lower < upper <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    values = handle_nan_inf(data)
    bounds = np.percentile(values, [lower, upper])
    return np.clip(values, bounds[0], bounds[1])


def normalize_intensity(
    data: np.ndarray,
    method: str = "zscore_nonzero",
    percentiles: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize intensities using an explicit, configurable method.

    ``zscore_nonzero`` avoids treating a zero background as tissue. ``minmax``
    scales the complete finite array to [0, 1]. Percentile clipping is optional
    and should be selected from the target dataset's documented protocol.
    """
    values = handle_nan_inf(convert_intensity(data))
    if percentiles is not None:
        values = clip_intensity(values, percentiles)
    method = method.lower()
    if method == "none":
        return values
    if method == "zscore_nonzero":
        mask = values != 0
        reference = values[mask]
        if reference.size == 0:
            return np.zeros_like(values)
        mean, standard_deviation = float(reference.mean()), float(reference.std())
        return (values - mean) / standard_deviation if standard_deviation else np.zeros_like(values)
    if method == "zscore":
        standard_deviation = float(values.std())
        return (values - float(values.mean())) / standard_deviation if standard_deviation else np.zeros_like(values)
    if method == "minmax":
        minimum, maximum = float(values.min()), float(values.max())
        return (values - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(values)
    if method == "percentile":
        if percentiles is None:
            raise ValueError("percentiles are required for percentile normalization")
        lower, upper = np.percentile(values, percentiles)
        if upper <= lower:
            return np.zeros_like(values)
        return np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    raise ValueError(f"Unsupported intensity normalization method: {method}")


def resample_volume(
    volume: NiftiVolume,
    target_spacing: Sequence[float],
    is_mask: bool = False,
) -> NiftiVolume:
    """Resample a volume and preserve physical orientation in its updated affine."""
    target = np.asarray(tuple(target_spacing), dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)) or np.any(target <= 0):
        raise ValueError("target_spacing must contain three positive finite values")
    if volume.data.ndim != 3:
        raise ValueError("Only 3D arrays can be resampled")
    factors = np.asarray(volume.spacing) / target
    order = 0 if is_mask else 1
    data = zoom(volume.data, factors, order=order, mode="nearest", prefilter=not is_mask)
    affine = volume.affine.copy()
    affine[:3, :3] = volume.affine[:3, :3] @ np.diag(1 / factors)
    return replace(
        volume,
        data=data.astype(volume.data.dtype if is_mask else np.float32, copy=False),
        affine=affine,
        spacing=tuple(float(value) for value in target),
        shape=tuple(data.shape),
        dtype=str(data.dtype),
    )


def spatial_shape(volume: NiftiVolume) -> tuple[int, int, int]:
    """Return and validate the three spatial dimensions of a volume."""
    if len(volume.shape) != 3:
        raise ValueError(f"Expected 3D shape, found {volume.shape}")
    return tuple(int(value) for value in volume.shape)


def pad_or_crop_volume(volume: NiftiVolume, target_shape: Sequence[int]) -> NiftiVolume:
    """Center-pad or center-crop a volume while updating its affine translation."""
    target = np.asarray(tuple(target_shape), dtype=int)
    current = np.asarray(spatial_shape(volume), dtype=int)
    if target.shape != (3,) or np.any(target <= 0):
        raise ValueError("target_shape must contain three positive integers")
    starts = np.maximum((current - target) // 2, 0)
    ends = np.minimum(starts + target, current)
    slices = tuple(slice(int(start), int(end)) for start, end in zip(starts, ends))
    cropped = volume.data[slices]
    leading_padding = [max((wanted - size) // 2, 0) for wanted, size in zip(target, cropped.shape)]
    remainder = target - np.asarray(cropped.shape) - np.asarray(leading_padding)
    padding = [(before, int(extra)) for before, extra in zip(leading_padding, remainder)]
    data = np.pad(cropped, padding, mode="constant")
    source_offset = starts - np.asarray([item[0] for item in padding])
    affine = volume.affine.copy()
    affine[:3, 3] = volume.affine[:3, :3] @ source_offset + volume.affine[:3, 3]
    return replace(volume, data=data, affine=affine, shape=tuple(data.shape), dtype=str(data.dtype))


def validate_mask(mask: NiftiVolume) -> tuple[bool, tuple[str, ...]]:
    """Validate a mask's dimensionality, finiteness, and integer-valued labels."""
    messages: list[str] = []
    if mask.data.ndim != 3:
        messages.append("mask is not 3D")
    if not np.all(np.isfinite(mask.data)):
        messages.append("mask contains NaN or infinite values")
    if not np.allclose(mask.data, np.rint(mask.data)):
        messages.append("mask contains non-integer labels")
    return not messages, tuple(messages)


def validate_image_mask_compatibility(image: NiftiVolume, mask: NiftiVolume) -> tuple[bool, tuple[str, ...]]:
    """Check image-mask array shape and spatial metadata compatibility."""
    messages: list[str] = []
    if image.shape != mask.shape:
        messages.append(f"shape mismatch: image {image.shape}, mask {mask.shape}")
    if not np.allclose(image.affine, mask.affine):
        messages.append("affine mismatch between image and mask")
    return not messages, tuple(messages)


def volume_to_tensor(
    data: np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    add_channel: bool = True,
) -> torch.Tensor:
    """Convert a contiguous 3D array to a CPU or explicitly selected tensor."""
    values = np.ascontiguousarray(data)
    tensor = torch.as_tensor(values, dtype=dtype)
    if add_channel:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)