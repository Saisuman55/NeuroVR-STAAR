import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from src.volume_loader import load_nifti
from src.volume_preprocessing import (
    clip_intensity,
    convert_intensity,
    handle_nan_inf,
    normalize_intensity,
    pad_or_crop_volume,
    resample_volume,
    validate_image_mask_compatibility,
    validate_mask,
    volume_to_tensor,
)


class VolumePreprocessingTests(unittest.TestCase):
    def make_volume(self, root: Path, name: str, data: np.ndarray, spacing=(2.0, 3.0, 4.0)) -> object:
        affine = np.diag([*spacing, 1.0])
        path = root / name
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return load_nifti(path)

    def test_intensity_conversion_and_nonfinite_handling(self) -> None:
        values = handle_nan_inf(convert_intensity(np.array([np.nan, np.inf, -np.inf, 2])))
        np.testing.assert_array_equal(values, [0, 0, 0, 2])
        self.assertEqual(values.dtype, np.float32)

    def test_configurable_clipping_and_normalization(self) -> None:
        data = np.array([0, 1, 2, 3, 100], dtype=float)
        clipped = clip_intensity(data, (0, 80))
        self.assertAlmostEqual(float(clipped.max()), 22.4)
        normalized = normalize_intensity(data, "minmax", (0, 80))
        self.assertAlmostEqual(float(normalized.max()), 1.0)
        percentile = normalize_intensity(data, "percentile", (0, 80))
        self.assertAlmostEqual(float(percentile.max()), 1.0)

    def test_deterministic_normalization(self) -> None:
        data = np.arange(8, dtype=float).reshape(2, 2, 2)
        np.testing.assert_array_equal(normalize_intensity(data), normalize_intensity(data))

    def test_resampling_image_and_mask_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = self.make_volume(root, "image.nii", np.arange(8, dtype=np.float32).reshape(2, 2, 2))
            mask = self.make_volume(root, "mask.nii", np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], dtype=np.uint8))
            resized_image = resample_volume(image, (1, 1, 1))
            resized_mask = resample_volume(mask, (1, 1, 1), is_mask=True)
            self.assertEqual(resized_image.shape, (4, 6, 8))
            self.assertEqual(resized_mask.shape, (4, 6, 8))
            self.assertEqual(resized_mask.dtype, "uint8")
            self.assertTrue(set(np.unique(resized_mask.data)).issubset({0, 1}))
            np.testing.assert_allclose(resized_mask.spacing, (1, 1, 1))
            np.testing.assert_allclose(resized_mask.affine[:3, :3], np.diag([1, 1, 1]))

    def test_padding_and_cropping_updates_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            volume = self.make_volume(Path(directory), "scan.nii", np.ones((3, 4, 5)))
            padded = pad_or_crop_volume(volume, (5, 6, 7))
            cropped = pad_or_crop_volume(padded, (3, 4, 5))
            self.assertEqual(padded.shape, (5, 6, 7))
            self.assertEqual(cropped.shape, (3, 4, 5))
            self.assertTrue(np.all(np.isfinite(padded.affine)))

    def test_mask_and_image_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = self.make_volume(root, "image.nii", np.zeros((2, 2, 2)))
            mask = self.make_volume(root, "mask.nii", np.zeros((2, 2, 2), dtype=np.uint8))
            valid, messages = validate_mask(mask)
            self.assertTrue(valid)
            valid, messages = validate_image_mask_compatibility(image, mask)
            self.assertTrue(valid)
            mismatched = self.make_volume(root, "mismatch.nii", np.zeros((2, 2, 3), dtype=np.uint8))
            valid, messages = validate_image_mask_compatibility(image, mismatched)
            self.assertFalse(valid)
            self.assertIn("shape mismatch", messages[0])

    def test_tensor_conversion_on_cpu(self) -> None:
        tensor = volume_to_tensor(np.ones((2, 3, 4), dtype=np.float32))
        self.assertEqual(tensor.shape, (1, 2, 3, 4))
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
