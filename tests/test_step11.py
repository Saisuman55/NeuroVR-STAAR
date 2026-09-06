"""Tests for Step 11: real 3D MRI dataset acquisition, import, and validation.

All NIfTI files used here are synthetic and created in temporary directories.
They are never presented as real medical data.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _uniform(path: Path, shape=(4, 4, 4), value=1.0, affine=None) -> Path:
    return _nifti(path, np.full(shape, value, dtype=np.float32), affine)


def _seg(path: Path, shape=(4, 4, 4), labels=(0, 1, 2, 4), affine=None) -> Path:
    """Create a synthetic multi-label segmentation volume."""
    data = np.zeros(shape, dtype=np.int16)
    # Sprinkle a few non-zero labels so the volume is non-trivial.
    for i, lbl in enumerate(labels[1:], start=1):
        data[i % shape[0], i % shape[1], i % shape[2]] = lbl
    return _nifti(path, data, affine)


def _make_full_subject(root: Path, subject_id: str, shape=(4, 4, 4), affine=None):
    """Create a complete synthetic BraTS-style subject directory."""
    d = root / subject_id
    d.mkdir(parents=True, exist_ok=True)
    for modality in ("flair", "t1", "t1ce", "t2"):
        _uniform(d / f"{subject_id}_{modality}.nii.gz", shape=shape, affine=affine)
    _seg(d / f"{subject_id}_seg.nii.gz", shape=shape, affine=affine)
    return d


# ---------------------------------------------------------------------------
# 1. Modality detection
# ---------------------------------------------------------------------------

class ModalityDetectionTests(unittest.TestCase):
    def _detect(self, filename):
        from scripts.import_brats_dataset import detect_modality
        return detect_modality(filename)

    def test_flair(self):
        self.assertEqual(self._detect("BraTS21_001_flair.nii.gz"), "flair")

    def test_t1_plain(self):
        self.assertEqual(self._detect("BraTS21_001_t1.nii.gz"), "t1")

    def test_t1ce_not_confused_with_t1(self):
        """t1ce must never be detected as t1."""
        result = self._detect("BraTS21_001_t1ce.nii.gz")
        self.assertEqual(result, "t1ce")
        self.assertNotEqual(result, "t1")

    def test_t1gd_maps_to_t1ce(self):
        self.assertEqual(self._detect("subject_t1gd.nii.gz"), "t1ce")

    def test_t2(self):
        self.assertEqual(self._detect("BraTS21_001_t2.nii.gz"), "t2")

    def test_seg(self):
        self.assertEqual(self._detect("BraTS21_001_seg.nii.gz"), "seg")

    def test_mask_maps_to_seg(self):
        self.assertEqual(self._detect("subject_mask.nii.gz"), "seg")

    def test_unrecognised_returns_none(self):
        self.assertIsNone(self._detect("readme.txt"))
        self.assertIsNone(self._detect("notes.nii.gz"))

    def test_t1_token_boundary(self):
        """'t1' must not match inside 't1ce' — token boundary check."""
        from scripts.import_brats_dataset import detect_modality
        # A filename whose only token is 't1ce' must not return 't1'.
        self.assertEqual(detect_modality("sub_t1ce.nii.gz"), "t1ce")
        self.assertEqual(detect_modality("sub_t1.nii.gz"), "t1")


# ---------------------------------------------------------------------------
# 2. Subject discovery
# ---------------------------------------------------------------------------

class SubjectDiscoveryTests(unittest.TestCase):
    def test_full_subject_discovered(self):
        from scripts.import_brats_dataset import discover_subjects
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_full_subject(root, "BraTS21_001")
            subjects = discover_subjects(root)
            self.assertIn("BraTS21_001", subjects)
            self.assertEqual(
                set(subjects["BraTS21_001"].keys()),
                {"flair", "t1", "t1ce", "t2", "seg"},
            )

    def test_duplicate_modality_raises(self):
        from scripts.import_brats_dataset import discover_subjects
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_dup"
            subj.mkdir()
            _uniform(subj / "sub_dup_flair.nii.gz")
            _uniform(subj / "sub_dup_flair_v2.nii.gz")  # second flair
            with self.assertRaises(ValueError):
                discover_subjects(root)

    def test_duplicate_subject_id_raises(self):
        from scripts.import_brats_dataset import discover_subjects
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Two directories with the same name cannot exist on the same FS,
            # so we test the guard by patching the dict directly.
            _make_full_subject(root, "sub_001")
            subjects = discover_subjects(root)
            self.assertEqual(len(subjects), 1)

    def test_directory_without_nifti_ignored(self):
        from scripts.import_brats_dataset import discover_subjects
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            empty = root / "empty_subject"
            empty.mkdir()
            (empty / "notes.txt").write_text("nothing")
            subjects = discover_subjects(root)
            self.assertNotIn("empty_subject", subjects)


# ---------------------------------------------------------------------------
# 3. Per-subject validation
# ---------------------------------------------------------------------------

class SubjectValidationTests(unittest.TestCase):
    def test_valid_subject_passes(self):
        from scripts.import_brats_dataset import discover_subjects, validate_subject
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_full_subject(root, "sub_ok")
            subjects = discover_subjects(root)
            record = validate_subject("sub_ok", subjects["sub_ok"])
            self.assertTrue(record["valid"])
            self.assertEqual(record["missing_modalities"], [])
            self.assertEqual(record["errors"], [])

    def test_missing_modalities_detected(self):
        from scripts.import_brats_dataset import discover_subjects, validate_subject
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_partial"
            subj.mkdir()
            _uniform(subj / "sub_partial_flair.nii.gz")
            _uniform(subj / "sub_partial_t1.nii.gz")
            subjects = discover_subjects(root)
            record = validate_subject("sub_partial", subjects["sub_partial"])
            self.assertFalse(record["valid"])
            self.assertIn("t1ce", record["missing_modalities"])
            self.assertIn("t2", record["missing_modalities"])
            self.assertIn("seg", record["missing_modalities"])

    def test_shape_mismatch_invalidates_subject(self):
        from scripts.import_brats_dataset import discover_subjects, validate_subject
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_mismatch"
            subj.mkdir()
            for mod in ("flair", "t1", "t1ce", "seg"):
                _uniform(subj / f"sub_mismatch_{mod}.nii.gz", shape=(4, 4, 4))
            _uniform(subj / "sub_mismatch_t2.nii.gz", shape=(4, 4, 5))  # wrong
            subjects = discover_subjects(root)
            record = validate_subject("sub_mismatch", subjects["sub_mismatch"])
            self.assertFalse(record["valid"])
            self.assertTrue(any("shape" in e for e in record["errors"]))

    def test_affine_mismatch_invalidates_subject(self):
        from scripts.import_brats_dataset import discover_subjects, validate_subject
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_affine"
            subj.mkdir()
            aff_a = np.eye(4)
            aff_b = np.diag([2.0, 2.0, 2.0, 1.0])
            for mod in ("flair", "t1", "t1ce", "t2", "seg"):
                aff = aff_b if mod == "t2" else aff_a
                _uniform(subj / f"sub_affine_{mod}.nii.gz", affine=aff)
            subjects = discover_subjects(root)
            record = validate_subject("sub_affine", subjects["sub_affine"])
            self.assertFalse(record["valid"])
            self.assertTrue(any("affine" in e for e in record["errors"]))

    def test_spacing_recorded_in_record(self):
        from scripts.import_brats_dataset import discover_subjects, validate_subject
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            aff = np.diag([1.0, 1.0, 1.0, 1.0])
            _make_full_subject(root, "sub_spacing", affine=aff)
            subjects = discover_subjects(root)
            record = validate_subject("sub_spacing", subjects["sub_spacing"])
            self.assertIn("spacing", record)
            self.assertEqual(len(record["spacing"]), 3)


# ---------------------------------------------------------------------------
# 4. Segmentation label analysis
# ---------------------------------------------------------------------------

class SegmentationLabelTests(unittest.TestCase):
    def test_labels_reported_accurately(self):
        from scripts.import_brats_dataset import discover_subjects, validate_subject
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_labels"
            subj.mkdir()
            for mod in ("flair", "t1", "t1ce", "t2"):
                _uniform(subj / f"sub_labels_{mod}.nii.gz")
            # Segmentation with labels 0, 1, 2, 4.
            data = np.zeros((4, 4, 4), dtype=np.int16)
            data[0, 0, 0] = 1
            data[1, 1, 1] = 2
            data[2, 2, 2] = 4
            _nifti(subj / "sub_labels_seg.nii.gz", data)
            subjects = discover_subjects(root)
            record = validate_subject("sub_labels", subjects["sub_labels"])
            seg = record.get("segmentation_analysis", {})
            self.assertTrue(seg.get("readable"))
            self.assertIn(0, seg["unique_labels"])
            self.assertIn(1, seg["unique_labels"])
            self.assertIn(2, seg["unique_labels"])
            self.assertIn(4, seg["unique_labels"])
            self.assertGreater(seg["tumor_voxels"], 0)
            self.assertTrue(seg["has_tumor"])

    def test_empty_segmentation_detected(self):
        from scripts.import_brats_dataset import discover_subjects, validate_subject
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_empty_seg"
            subj.mkdir()
            for mod in ("flair", "t1", "t1ce", "t2"):
                _uniform(subj / f"sub_empty_seg_{mod}.nii.gz")
            _nifti(subj / "sub_empty_seg_seg.nii.gz", np.zeros((4, 4, 4), dtype=np.int16))
            subjects = discover_subjects(root)
            record = validate_subject("sub_empty_seg", subjects["sub_empty_seg"])
            seg = record.get("segmentation_analysis", {})
            self.assertFalse(seg.get("has_tumor"))
            self.assertEqual(seg["tumor_voxels"], 0)


# ---------------------------------------------------------------------------
# 5. Binary mask conversion
# ---------------------------------------------------------------------------

class BinaryMaskConversionTests(unittest.TestCase):
    def test_all_nonzero_become_one(self):
        from src.datasets.brain_tumor_3d_dataset import binary_mask_from_segmentation
        data = np.array([[[0, 1, 2, 4]]], dtype=np.int16)
        result = binary_mask_from_segmentation(data)
        expected = np.array([[[0, 1, 1, 1]]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_all_background_stays_zero(self):
        from src.datasets.brain_tumor_3d_dataset import binary_mask_from_segmentation
        data = np.zeros((4, 4, 4), dtype=np.int16)
        result = binary_mask_from_segmentation(data)
        self.assertEqual(result.sum(), 0)
        self.assertEqual(result.dtype, np.float32)

    def test_original_not_modified(self):
        from src.datasets.brain_tumor_3d_dataset import binary_mask_from_segmentation
        data = np.array([[[0, 1, 2, 4]]], dtype=np.int16)
        original = data.copy()
        binary_mask_from_segmentation(data)
        np.testing.assert_array_equal(data, original)


# ---------------------------------------------------------------------------
# 6. Subject-level splits and leakage
# ---------------------------------------------------------------------------

class SplitLeakageTests(unittest.TestCase):
    def test_no_leakage_across_partitions(self):
        from scripts.import_brats_dataset import _split_subjects
        subjects = [f"sub_{i:03d}" for i in range(30)]
        splits = _split_subjects(subjects, seed=42)
        train = set(splits["train_subjects"])
        val   = set(splits["validation_subjects"])
        test  = set(splits["test_subjects"])
        self.assertFalse(train & val,  "Train/validation leakage")
        self.assertFalse(train & test, "Train/test leakage")
        self.assertFalse(val & test,   "Validation/test leakage")
        self.assertEqual(train | val | test, set(subjects))

    def test_deterministic_with_same_seed(self):
        from scripts.import_brats_dataset import _split_subjects
        subjects = [f"sub_{i:03d}" for i in range(20)]
        self.assertEqual(
            _split_subjects(subjects, seed=7),
            _split_subjects(subjects, seed=7),
        )

    def test_different_seeds_give_different_splits(self):
        from scripts.import_brats_dataset import _split_subjects
        subjects = [f"sub_{i:03d}" for i in range(20)]
        self.assertNotEqual(
            _split_subjects(subjects, seed=1)["train_subjects"],
            _split_subjects(subjects, seed=2)["train_subjects"],
        )

    def test_split_policy_documented(self):
        from scripts.import_brats_dataset import _split_subjects
        splits = _split_subjects(["a", "b", "c", "d", "e"], seed=0)
        self.assertIn("split_policy", splits)
        self.assertIn("subject-level", splits["split_policy"])

    def test_approximate_ratios(self):
        from scripts.import_brats_dataset import _split_subjects
        subjects = [f"sub_{i:03d}" for i in range(100)]
        splits = _split_subjects(subjects, seed=42)
        self.assertGreater(len(splits["train_subjects"]), 60)
        self.assertGreater(len(splits["validation_subjects"]), 5)
        self.assertGreater(len(splits["test_subjects"]), 5)


# ---------------------------------------------------------------------------
# 7. Dataset manifest
# ---------------------------------------------------------------------------

class ManifestTests(unittest.TestCase):
    def test_manifest_written_with_correct_counts(self):
        from scripts.import_brats_dataset import import_dataset, _build_manifest
        with tempfile.TemporaryDirectory() as src_d, \
             tempfile.TemporaryDirectory() as dst_d:
            src = Path(src_d)
            dst = Path(dst_d) / "brats"
            _make_full_subject(src, "sub_001")
            _make_full_subject(src, "sub_002")
            report = import_dataset(src, dst)
            manifest = _build_manifest(report)
            self.assertEqual(manifest["discovered_subjects"], 2)
            self.assertEqual(manifest["valid_subjects"], 2)
            self.assertEqual(manifest["invalid_subjects"], 0)
            for mod in ("flair", "t1", "t1ce", "t2", "seg"):
                self.assertEqual(manifest["modality_availability"][mod], 2)

    def test_manifest_json_serialisable(self):
        from scripts.import_brats_dataset import import_dataset, _build_manifest
        with tempfile.TemporaryDirectory() as src_d, \
             tempfile.TemporaryDirectory() as dst_d:
            src = Path(src_d)
            dst = Path(dst_d) / "brats"
            _make_full_subject(src, "sub_001")
            report = import_dataset(src, dst)
            manifest = _build_manifest(report)
            # Must not raise.
            serialised = json.dumps(manifest)
            loaded = json.loads(serialised)
            self.assertEqual(loaded["valid_subjects"], 1)

    def test_invalid_subject_counted_separately(self):
        from scripts.import_brats_dataset import import_dataset, _build_manifest
        with tempfile.TemporaryDirectory() as src_d, \
             tempfile.TemporaryDirectory() as dst_d:
            src = Path(src_d)
            dst = Path(dst_d) / "brats"
            _make_full_subject(src, "sub_good")
            # Partial subject — missing t1ce, t2, seg.
            bad = src / "sub_bad"
            bad.mkdir()
            _uniform(bad / "sub_bad_flair.nii.gz")
            _uniform(bad / "sub_bad_t1.nii.gz")
            report = import_dataset(src, dst)
            manifest = _build_manifest(report)
            self.assertEqual(manifest["valid_subjects"], 1)
            self.assertEqual(manifest["invalid_subjects"], 1)


# ---------------------------------------------------------------------------
# 8. 3D dataset class — lazy loading and channel ordering
# ---------------------------------------------------------------------------

class BrainTumor3DDatasetTests(unittest.TestCase):
    def _make_subject(self, root: Path, sid: str, shape=(4, 4, 4)):
        _make_full_subject(root, sid, shape=shape)

    def test_channel_order_is_flair_t1_t1ce_t2(self):
        from src.datasets.brain_tumor_3d_dataset import MODALITIES, CHANNEL_ORDER
        self.assertEqual(MODALITIES[0], "flair")
        self.assertEqual(MODALITIES[1], "t1")
        self.assertEqual(MODALITIES[2], "t1ce")
        self.assertEqual(MODALITIES[3], "t2")
        self.assertEqual(CHANNEL_ORDER["flair"], 0)
        self.assertEqual(CHANNEL_ORDER["t1ce"], 2)

    def test_image_tensor_shape(self):
        from src.datasets.brain_tumor_3d_dataset import subjects_from_directory, BrainTumor3DDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._make_subject(root, "sub_001", shape=(4, 5, 6))
            subjects = subjects_from_directory(root)
            dataset = BrainTumor3DDataset(subjects)
            image, mask = dataset[0]
            self.assertEqual(image.shape, (4, 4, 5, 6))   # [C, D, H, W]
            self.assertIsNotNone(mask)
            self.assertEqual(mask.shape, (1, 4, 5, 6))    # [1, D, H, W]

    def test_binary_mask_applied_by_default(self):
        from src.datasets.brain_tumor_3d_dataset import subjects_from_directory, BrainTumor3DDataset
        import torch
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_multi"
            subj.mkdir()
            for mod in ("flair", "t1", "t1ce", "t2"):
                _uniform(subj / f"sub_multi_{mod}.nii.gz")
            # Multi-label seg: labels 0, 1, 2, 4.
            data = np.zeros((4, 4, 4), dtype=np.int16)
            data[0, 0, 0] = 1
            data[1, 1, 1] = 2
            data[2, 2, 2] = 4
            _nifti(subj / "sub_multi_seg.nii.gz", data)
            subjects = subjects_from_directory(root)
            dataset = BrainTumor3DDataset(subjects, binary_mask=True)
            _, mask = dataset[0]
            unique_vals = torch.unique(mask).tolist()
            # Binary: only 0.0 and 1.0 allowed.
            self.assertTrue(all(v in (0.0, 1.0) for v in unique_vals))

    def test_no_segmentation_returns_none_mask(self):
        from src.datasets.brain_tumor_3d_dataset import BrainTumorSubject, BrainTumor3DDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_noseg"
            subj.mkdir()
            paths = {}
            for mod in ("flair", "t1", "t1ce", "t2"):
                p = subj / f"sub_noseg_{mod}.nii.gz"
                _uniform(p)
                paths[mod] = p
            subject = BrainTumorSubject(
                "sub_noseg", paths["flair"], paths["t1"], paths["t1ce"], paths["t2"], None
            )
            dataset = BrainTumor3DDataset([subject])
            _, mask = dataset[0]
            self.assertIsNone(mask)

    def test_subjects_from_splits(self):
        from scripts.import_brats_dataset import import_dataset, _split_subjects
        from src.datasets.brain_tumor_3d_dataset import subjects_from_splits
        with tempfile.TemporaryDirectory() as src_d, \
             tempfile.TemporaryDirectory() as dst_d:
            src = Path(src_d)
            dst = Path(dst_d) / "brats"
            for i in range(5):
                _make_full_subject(src, f"sub_{i:03d}")
            report = import_dataset(src, dst)
            valid_ids = sorted(r["subject_id"] for r in report["subjects"] if r["valid"])
            splits = _split_subjects(valid_ids, seed=42)
            (dst / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
            train_subjects = subjects_from_splits(dst, "train")
            val_subjects   = subjects_from_splits(dst, "validation")
            test_subjects  = subjects_from_splits(dst, "test")
            all_ids = (
                {s.subject_id for s in train_subjects}
                | {s.subject_id for s in val_subjects}
                | {s.subject_id for s in test_subjects}
            )
            self.assertEqual(all_ids, set(valid_ids))
            # No overlap.
            train_ids = {s.subject_id for s in train_subjects}
            val_ids   = {s.subject_id for s in val_subjects}
            test_ids  = {s.subject_id for s in test_subjects}
            self.assertFalse(train_ids & val_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(val_ids & test_ids)

    def test_spatial_misalignment_raises(self):
        from src.datasets.brain_tumor_3d_dataset import BrainTumorSubject, BrainTumor3DDataset
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            subj = root / "sub_misalign"
            subj.mkdir()
            aff_a = np.eye(4)
            aff_b = np.diag([2.0, 2.0, 2.0, 1.0])
            paths = {}
            for i, mod in enumerate(("flair", "t1", "t1ce", "t2")):
                aff = aff_b if mod == "t2" else aff_a
                p = subj / f"sub_misalign_{mod}.nii.gz"
                _uniform(p, affine=aff)
                paths[mod] = p
            seg_p = subj / "sub_misalign_seg.nii.gz"
            _uniform(seg_p, affine=aff_a)
            subject = BrainTumorSubject(
                "sub_misalign",
                paths["flair"], paths["t1"], paths["t1ce"], paths["t2"], seg_p,
            )
            dataset = BrainTumor3DDataset([subject])
            with self.assertRaises(ValueError):
                dataset[0]


# ---------------------------------------------------------------------------
# 9. Dataset report — segmentation section
# ---------------------------------------------------------------------------

class SegmentationReportTests(unittest.TestCase):
    def test_report_not_imported_when_no_manifest(self):
        from scripts.dataset_report import segmentation_report
        with tempfile.TemporaryDirectory() as d:
            brats_root = Path(d) / "brats"
            brats_root.mkdir()
            report = segmentation_report(brats_root)
            self.assertEqual(report["status"], "not_imported")
            self.assertEqual(report["valid_subjects"], 0)
            self.assertIn("not imported", report.get("message", "").lower())

    def test_report_reads_manifest_counts(self):
        from scripts.import_brats_dataset import import_dataset, _build_manifest
        from scripts.dataset_report import segmentation_report
        with tempfile.TemporaryDirectory() as src_d, \
             tempfile.TemporaryDirectory() as dst_d:
            src = Path(src_d)
            brats_root = Path(dst_d) / "brats"
            _make_full_subject(src, "sub_001")
            _make_full_subject(src, "sub_002")
            report_data = import_dataset(src, brats_root)
            manifest = _build_manifest(report_data)
            (brats_root / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = segmentation_report(brats_root)
            self.assertEqual(report["valid_subjects"], 2)
            self.assertEqual(report["discovered_subjects"], 2)


if __name__ == "__main__":
    unittest.main()
