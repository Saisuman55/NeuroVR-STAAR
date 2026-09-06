import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nibabel as nib
import numpy as np

from src.volume_loader import (
    get_orientation,
    get_voxel_spacing,
    inspect_nifti,
    load_nifti,
    physical_to_voxel,
    validate_nifti,
    voxel_to_physical,
)


class VolumeLoaderTests(unittest.TestCase):
    def save_volume(self, root: Path, name: str, data: np.ndarray, affine: np.ndarray) -> Path:
        path = root / name
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return path

    def test_load_nii_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.diag([2.0, 3.0, 4.0, 1.0])
            path = self.save_volume(root, "scan.nii", np.ones((4, 5, 6), dtype=np.float32), affine)
            volume = load_nifti(path)
            self.assertEqual(volume.shape, (4, 5, 6))
            self.assertEqual(volume.spacing, (2.0, 3.0, 4.0))
            np.testing.assert_array_equal(volume.affine, affine)
            self.assertEqual(volume.dtype, "float32")
            self.assertEqual(get_orientation(volume), ("R", "A", "S"))

    def test_load_nii_gz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_volume(Path(directory), "scan.nii.gz", np.zeros((2, 3, 4)), np.eye(4))
            self.assertEqual(load_nifti(path).shape, (2, 3, 4))

    def test_missing_and_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                load_nifti(root / "missing.nii")
            with self.assertRaises(ValueError):
                load_nifti(root / "scan.png")

    def test_invalid_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.nii"
            path.write_bytes(b"invalid")
            valid, messages = validate_nifti(path)
            self.assertFalse(valid)
            self.assertTrue(messages)

    def test_inspection_reports_nan_inf_and_constant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.eye(4)
            constant = self.save_volume(root, "constant.nii", np.ones((2, 2, 2)), affine)
            report = inspect_nifti(constant)
            self.assertTrue(report.is_constant)
            self.assertFalse(report.has_nan)
            self.assertEqual(report.intensity_statistics["mean"], 1.0)
            mixed = self.save_volume(root, "mixed.nii", np.array([[[np.nan, np.inf]]]), affine)
            report = inspect_nifti(mixed)
            self.assertTrue(report.has_nan)
            self.assertTrue(report.has_inf)
            self.assertEqual(report.nan_count, 1)
            self.assertEqual(report.inf_count, 1)
            self.assertFalse(report.valid)

    def test_4d_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.save_volume(Path(directory), "four_d.nii", np.zeros((2, 2, 2, 2)), np.eye(4))
            valid, messages = validate_nifti(path)
            self.assertFalse(valid)
            self.assertIn("4D", messages[0])

    def test_coordinate_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            affine = np.array([[2, 0, 0, 10], [0, 3, 0, -4], [0, 0, 4, 2], [0, 0, 0, 1]], dtype=float)
            path = self.save_volume(Path(directory), "scan.nii", np.zeros((3, 3, 3)), affine)
            volume = load_nifti(path)
            physical = voxel_to_physical(volume, (1, 2, 0.5))
            np.testing.assert_allclose(physical, [12, 2, 4])
            np.testing.assert_allclose(physical_to_voxel(volume, physical), [1, 2, 0.5])

    def test_invalid_affine_and_spacing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.nii"
            path.touch()
            fake_image = SimpleNamespace(
                dataobj=np.zeros((2, 2, 2)),
                affine=np.zeros((4, 4)),
                header=SimpleNamespace(copy=lambda: object()),
            )
            with patch("src.volume_loader.nib.load", return_value=fake_image):
                valid, messages = validate_nifti(path)
            self.assertFalse(valid)
            self.assertIn("affine", messages[0])


if __name__ == "__main__":
    unittest.main()
