"""Tests for Step 10: real dataset acquisition, validation, and MRI input integration."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from src.utils import load_config
from src.web_api import create_app, AnalysisState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti(path: Path, shape=(4, 4, 4), affine=None) -> Path:
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine), str(path))
    return path


def _make_png(path: Path, size=(8, 8)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 150, 200)).save(path)
    return path


def _png_bytes(size=(8, 8)) -> io.BytesIO:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(20, 40, 60)).save(buf, format="PNG")
    buf.seek(0)
    return buf


def _nifti_bytes(shape=(4, 4, 4)) -> io.BytesIO:
    buf = io.BytesIO()
    img = nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4))
    file_map = img.make_file_map({"image": buf, "header": buf})
    img.to_file_map(file_map)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Classification dataset manifest
# ---------------------------------------------------------------------------

class ClassificationManifestTests(unittest.TestCase):
    def test_manifest_written_by_downloader_helpers(self) -> None:
        """Manifest must record actual discovered counts, not hard-coded values."""
        from scripts.download_classification_dataset import inspect_files, create_manifest

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cls in ("glioma", "meningioma"):
                _make_png(root / "Training" / cls / "a.jpg")
            report = inspect_files(root)
            create_manifest(root, root, report)
            manifest_path = root / "dataset_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text())
            discovered = manifest["discovered_by_neurovr"]
            self.assertEqual(discovered["images_per_class"]["glioma"], 1)
            self.assertEqual(discovered["images_per_class"]["meningioma"], 1)
            self.assertEqual(discovered["images_per_class"]["notumor"], 0)

    def test_four_class_validation(self) -> None:
        """All four expected classes must be detected."""
        from scripts.download_classification_dataset import discover_split_class_files, EXPECTED_CLASSES

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cls in EXPECTED_CLASSES:
                _make_png(root / "Training" / cls / "img.jpg")
            splits = discover_split_class_files(root)
            for cls in EXPECTED_CLASSES:
                self.assertEqual(len(splits["Training"][cls]), 1, f"Missing class: {cls}")

    def test_corrupt_image_detected(self) -> None:
        """Corrupt images must be listed in corrupted_files, not silently skipped."""
        from scripts.download_classification_dataset import inspect_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "Training" / "glioma" / "bad.jpg"
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_bytes(b"not an image")
            report = inspect_files(root)
            self.assertEqual(len(report["corrupted_files"]), 1)

    def test_training_testing_split_detection(self) -> None:
        """Files under Testing/ must be assigned to the Testing split."""
        from scripts.download_classification_dataset import discover_split_class_files

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_png(root / "Training" / "glioma" / "train.jpg")
            _make_png(root / "Testing" / "glioma" / "test.jpg")
            splits = discover_split_class_files(root)
            self.assertEqual(len(splits["Training"]["glioma"]), 1)
            self.assertEqual(len(splits["Testing"]["glioma"]), 1)


# ---------------------------------------------------------------------------
# BraTS modality detection
# ---------------------------------------------------------------------------

class BraTSModalityTests(unittest.TestCase):
    def test_modality_detection_by_filename(self) -> None:
        from scripts.import_brats_dataset import detect_modality

        self.assertEqual(detect_modality("BraTS21_001_flair.nii.gz"), "flair")
        self.assertEqual(detect_modality("BraTS21_001_t1.nii.gz"), "t1")
        self.assertEqual(detect_modality("BraTS21_001_t1ce.nii.gz"), "t1ce")
        self.assertEqual(detect_modality("BraTS21_001_t2.nii.gz"), "t2")
        self.assertEqual(detect_modality("BraTS21_001_seg.nii.gz"), "seg")
        self.assertIsNone(detect_modality("readme.txt"))

    def test_subject_discovery(self) -> None:
        from scripts.import_brats_dataset import discover_subjects

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "BraTS21_001"
            subject.mkdir()
            for modality in ("flair", "t1", "t1ce", "t2", "seg"):
                _make_nifti(subject / f"BraTS21_001_{modality}.nii.gz")
            subjects = discover_subjects(root)
            self.assertIn("BraTS21_001", subjects)
            self.assertEqual(set(subjects["BraTS21_001"].keys()), {"flair", "t1", "t1ce", "t2", "seg"})

    def test_missing_modality_detected(self) -> None:
        from scripts.import_brats_dataset import discover_subjects, validate_subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "BraTS21_002"
            subject.mkdir()
            # Only flair and t1 — missing t1ce, t2, seg
            for modality in ("flair", "t1"):
                _make_nifti(subject / f"BraTS21_002_{modality}.nii.gz")
            subjects = discover_subjects(root)
            record = validate_subject("BraTS21_002", subjects["BraTS21_002"])
            self.assertFalse(record["valid"])
            self.assertIn("t1ce", record["missing_modalities"])
            self.assertIn("t2", record["missing_modalities"])
            self.assertIn("seg", record["missing_modalities"])

    def test_modality_shape_consistency(self) -> None:
        from scripts.import_brats_dataset import discover_subjects, validate_subject

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject = root / "BraTS21_003"
            subject.mkdir()
            for modality in ("flair", "t1", "t1ce", "t2", "seg"):
                shape = (4, 4, 4) if modality != "t2" else (4, 4, 5)  # mismatch
                _make_nifti(subject / f"BraTS21_003_{modality}.nii.gz", shape=shape)
            subjects = discover_subjects(root)
            record = validate_subject("BraTS21_003", subjects["BraTS21_003"])
            self.assertFalse(record["valid"])
            self.assertTrue(any("shape" in e for e in record.get("errors", [])))


# ---------------------------------------------------------------------------
# Subject-level splits and leakage
# ---------------------------------------------------------------------------

class SubjectSplitTests(unittest.TestCase):
    def test_subject_level_split_no_leakage(self) -> None:
        from scripts.import_brats_dataset import _split_subjects

        subjects = [f"sub_{i:03d}" for i in range(20)]
        splits = _split_subjects(subjects, seed=42)
        train = set(splits["train_subjects"])
        val = set(splits["validation_subjects"])
        test = set(splits["test_subjects"])
        self.assertFalse(train & val, "Train/validation leakage")
        self.assertFalse(train & test, "Train/test leakage")
        self.assertFalse(val & test, "Validation/test leakage")
        self.assertEqual(train | val | test, set(subjects))

    def test_split_deterministic(self) -> None:
        from scripts.import_brats_dataset import _split_subjects

        subjects = [f"sub_{i:03d}" for i in range(30)]
        self.assertEqual(_split_subjects(subjects, seed=7), _split_subjects(subjects, seed=7))

    def test_split_ratios(self) -> None:
        from scripts.import_brats_dataset import _split_subjects

        subjects = [f"sub_{i:03d}" for i in range(100)]
        splits = _split_subjects(subjects, seed=42)
        self.assertGreater(len(splits["train_subjects"]), 60)
        self.assertGreater(len(splits["validation_subjects"]), 5)
        self.assertGreater(len(splits["test_subjects"]), 5)

    def test_splits_json_written(self) -> None:
        from scripts.import_brats_dataset import _split_subjects

        subjects = [f"sub_{i:03d}" for i in range(10)]
        splits = _split_subjects(subjects, seed=42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps(splits), encoding="utf-8")
            loaded = json.loads(path.read_text())
            self.assertIn("train_subjects", loaded)
            self.assertIn("validation_subjects", loaded)
            self.assertIn("test_subjects", loaded)


# ---------------------------------------------------------------------------
# 3D dataset class
# ---------------------------------------------------------------------------

class BrainTumor3DDatasetTests(unittest.TestCase):
    def _make_subject_dir(self, root: Path, subject_id: str, shape=(4, 4, 4)) -> Path:
        subject = root / subject_id
        subject.mkdir(parents=True, exist_ok=True)
        for modality in ("flair", "t1", "t1ce", "t2"):
            _make_nifti(subject / f"{subject_id}_{modality}.nii.gz", shape=shape)
        _make_nifti(subject / f"{subject_id}_seg.nii.gz", shape=shape)
        return subject

    def test_subjects_from_directory(self) -> None:
        from src.datasets.brain_tumor_3d_dataset import subjects_from_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_subject_dir(root, "sub_001")
            subjects = subjects_from_directory(root)
            self.assertEqual(len(subjects), 1)
            self.assertEqual(subjects[0].subject_id, "sub_001")
            self.assertIsNotNone(subjects[0].segmentation)

    def test_dataset_len_and_getitem_shape(self) -> None:
        from src.datasets.brain_tumor_3d_dataset import subjects_from_directory, BrainTumor3DDataset
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_subject_dir(root, "sub_001", shape=(4, 4, 4))
            subjects = subjects_from_directory(root)
            dataset = BrainTumor3DDataset(subjects)
            self.assertEqual(len(dataset), 1)
            image, mask = dataset[0]
            self.assertEqual(image.shape, (4, 4, 4, 4))  # [C, D, H, W]
            self.assertIsNotNone(mask)
            self.assertEqual(mask.shape, (1, 4, 4, 4))   # [1, D, H, W]

    def test_dataset_no_segmentation(self) -> None:
        from src.datasets.brain_tumor_3d_dataset import BrainTumorSubject, BrainTumor3DDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subject_dir = root / "sub_002"
            subject_dir.mkdir()
            paths = {}
            for modality in ("flair", "t1", "t1ce", "t2"):
                p = subject_dir / f"sub_002_{modality}.nii.gz"
                _make_nifti(p)
                paths[modality] = p
            subject = BrainTumorSubject("sub_002", paths["flair"], paths["t1"], paths["t1ce"], paths["t2"], None)
            dataset = BrainTumor3DDataset([subject])
            image, mask = dataset[0]
            self.assertIsNone(mask)


# ---------------------------------------------------------------------------
# NIfTI metadata and slice extraction via API
# ---------------------------------------------------------------------------

class WebApiDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_config = load_config("config/config.yaml")
        cls.temp_directory = tempfile.TemporaryDirectory()
        root = Path(cls.temp_directory.name)
        config = copy.deepcopy(base_config)
        config["paths"] = {
            "uploads": str(root / "uploads"),
            "predictions": str(root / "predictions"),
            "reports": str(root / "reports"),
            "classifier_models": str(root / "classifier"),
            "segmenter_models": str(root / "segmenter"),
            "classification_dataset": str(root / "classification"),
            "segmentation_dataset": str(root / "segmentation"),
        }
        config["output"] = {
            "directory": str(root / "outputs"),
            "predictions_directory": str(root / "predictions"),
            "reports_directory": str(root / "reports"),
        }
        config["logging"] = {
            "directory": str(root / "logs"),
            "file": str(root / "logs" / "test.log"),
            "level": "INFO",
            "console": False,
        }
        cls.app = create_app(config)
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_directory.cleanup()

    def setUp(self) -> None:
        self.app.extensions["neurovr_service"].state = AnalysisState()

    # --- dataset status endpoints ---

    def test_datasets_endpoint_returns_both_pipelines(self) -> None:
        response = self.client.get("/api/datasets")
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn("classification", data)
        self.assertIn("segmentation", data)

    def test_classification_dataset_endpoint(self) -> None:
        response = self.client.get("/api/datasets/classification")
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn("status", data)
        self.assertIn("total_images", data)

    def test_segmentation_dataset_endpoint(self) -> None:
        response = self.client.get("/api/datasets/segmentation")
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertIn("status", data)
        self.assertIn("subjects", data)

    def test_input_2d_endpoint(self) -> None:
        response = self.client.get("/api/input/2d")
        self.assertEqual(response.status_code, 200)
        self.assertIn("files", response.json)

    def test_input_3d_endpoint(self) -> None:
        response = self.client.get("/api/input/3d")
        self.assertEqual(response.status_code, 200)
        self.assertIn("files", response.json)

    # --- 2D upload ---

    def test_2d_upload_jpg(self) -> None:
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), color=(80, 120, 160)).save(buf, format="JPEG")
        buf.seek(0)
        response = self.client.post(
            "/api/upload",
            data={"file": (buf, "scan.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["upload"]["file_type"], "2d_image")
        self.assertTrue(response.json["upload"]["validation"]["valid"])

    def test_2d_upload_png(self) -> None:
        response = self.client.post(
            "/api/upload",
            data={"file": (_png_bytes(), "scan.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["upload"]["file_type"], "2d_image")

    def test_corrupt_2d_upload_rejected(self) -> None:
        response = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"not an image"), "scan.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    # --- 3D upload and metadata ---

    def _upload_nifti(self, shape=(4, 4, 4)):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.nii.gz"
            _make_nifti(path, shape=shape)
            with path.open("rb") as f:
                response = self.client.post(
                    "/api/upload",
                    data={"file": (f, "study.nii.gz")},
                    content_type="multipart/form-data",
                )
        return response

    def test_3d_nifti_upload_returns_metadata(self) -> None:
        response = self._upload_nifti(shape=(8, 8, 8))
        self.assertEqual(response.status_code, 200)
        upload = response.json["upload"]
        self.assertEqual(upload["file_type"], "3d_volume")
        meta = upload["metadata"]
        self.assertEqual(meta["shape"], [8, 8, 8])
        self.assertIsNotNone(meta["voxel_spacing"])
        self.assertIsNotNone(meta["orientation"])
        self.assertIsNotNone(meta["dtype"])
        self.assertIsNotNone(meta["min_intensity"])
        self.assertIsNotNone(meta["max_intensity"])
        self.assertIsNotNone(meta["mean_intensity"])
        self.assertIsNotNone(meta["median_intensity"])
        self.assertEqual(meta["number_of_slices"], 8)

    def test_volume_metadata_endpoint(self) -> None:
        response = self._upload_nifti(shape=(6, 6, 6))
        volume_id = response.json["upload"]["volume_id"]
        meta_response = self.client.get(f"/api/volume/{volume_id}/metadata")
        self.assertEqual(meta_response.status_code, 200)
        self.assertEqual(meta_response.json["volume_id"], volume_id)
        self.assertEqual(meta_response.json["shape"], [6, 6, 6])

    def test_volume_metadata_unknown_id(self) -> None:
        response = self.client.get("/api/volume/nonexistent_id/metadata")
        self.assertEqual(response.status_code, 404)

    # --- slice extraction ---

    def test_axial_slice_returns_png(self) -> None:
        response = self._upload_nifti(shape=(8, 8, 8))
        volume_id = response.json["upload"]["volume_id"]
        slice_response = self.client.get(f"/api/volume/{volume_id}/slice/axial/3")
        self.assertEqual(slice_response.status_code, 200)
        self.assertEqual(slice_response.content_type, "image/png")

    def test_coronal_slice_returns_png(self) -> None:
        response = self._upload_nifti(shape=(8, 8, 8))
        volume_id = response.json["upload"]["volume_id"]
        slice_response = self.client.get(f"/api/volume/{volume_id}/slice/coronal/3")
        self.assertEqual(slice_response.status_code, 200)

    def test_sagittal_slice_returns_png(self) -> None:
        response = self._upload_nifti(shape=(8, 8, 8))
        volume_id = response.json["upload"]["volume_id"]
        slice_response = self.client.get(f"/api/volume/{volume_id}/slice/sagittal/3")
        self.assertEqual(slice_response.status_code, 200)

    def test_invalid_axis_rejected(self) -> None:
        response = self._upload_nifti(shape=(8, 8, 8))
        volume_id = response.json["upload"]["volume_id"]
        slice_response = self.client.get(f"/api/volume/{volume_id}/slice/diagonal/0")
        self.assertEqual(slice_response.status_code, 400)

    def test_out_of_range_slice_rejected(self) -> None:
        response = self._upload_nifti(shape=(4, 4, 4))
        volume_id = response.json["upload"]["volume_id"]
        slice_response = self.client.get(f"/api/volume/{volume_id}/slice/axial/99")
        self.assertEqual(slice_response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
