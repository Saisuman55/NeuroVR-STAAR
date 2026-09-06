"""Step 12B integration tests: Real demo inputs, NIfTI volumes, manifests, and Web API."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from src.utils import load_config
from src.volume_loader import inspect_nifti, load_nifti
from src.web_api import create_app


class TestStep12bManifestsAndInputs(unittest.TestCase):
    """Verify integrity of real 2D and 3D demonstration inputs and manifests."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).parent.parent
        cls.demo_2d_dir = cls.project_root / "data" / "demo_inputs" / "2d"
        cls.demo_3d_dir = cls.project_root / "data" / "demo_inputs" / "3d"

    def test_2d_manifest_structure_and_integrity(self) -> None:
        manifest_path = self.demo_2d_dir / "demo_manifest.json"
        self.assertTrue(manifest_path.is_file(), "2D demo manifest missing")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest.get("total_images"), 12)
        self.assertEqual(manifest.get("images_per_class"), 3)
        self.assertEqual(len(manifest.get("images", [])), 12)

        expected_classes = {"glioma", "meningioma", "notumor", "pituitary"}
        self.assertEqual(set(manifest.get("classes", [])), expected_classes)

        class_counts: dict[str, int] = {}
        for entry in manifest["images"]:
            class_name = entry["class_label"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            # Path must be relative and exist
            img_path = self.demo_2d_dir / class_name / entry["filename"]
            self.assertTrue(img_path.is_file(), f"Image missing: {img_path}")

            # Verify checksum
            digest = hashlib.sha256(img_path.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"], f"Checksum mismatch for {entry['filename']}")

            # Verify image readable by PIL
            with Image.open(img_path) as img:
                img.verify()
                self.assertGreater(img.size[0], 0)
                self.assertGreater(img.size[1], 0)

        for c in expected_classes:
            self.assertEqual(class_counts[c], 3, f"Class {c} must have exactly 3 demo images")

    def test_3d_manifest_structure_and_integrity(self) -> None:
        manifest_path = self.demo_3d_dir / "demo_manifest.json"
        self.assertTrue(manifest_path.is_file(), "3D demo manifest missing")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest.get("total_subjects"), 2)
        self.assertFalse(manifest.get("segmentation_available"))
        subjects = manifest.get("subjects", [])
        self.assertEqual(len(subjects), 2)

        for subj in subjects:
            self.assertFalse(subj["segmentation_available"])
            self.assertIsNone(subj["segmentation_labels"])
            self.assertIn("source_dataset", subj)
            self.assertIn("source_url", subj)

            rel_path = self.demo_3d_dir / subj["relative_path"]
            self.assertTrue(rel_path.is_file(), f"3D volume missing: {rel_path}")

            digest = hashlib.sha256(rel_path.read_bytes()).hexdigest()
            self.assertEqual(digest, subj["sha256"], f"Checksum mismatch for {subj['filename']}")
            self.assertEqual(rel_path.stat().st_size, subj["file_size_bytes"])

    def test_3d_volume_loading_nibabel_anatomical(self) -> None:
        vol_path = self.demo_3d_dir / "nibabel_anatomical" / "anatomical.nii"
        self.assertTrue(vol_path.is_file())

        volume = load_nifti(vol_path)
        self.assertEqual(volume.shape, (33, 41, 25))
        self.assertEqual(len(volume.spacing), 3)
        self.assertIsNotNone(volume.affine)
        self.assertEqual(volume.affine.shape, (4, 4))

        inspection = inspect_nifti(vol_path)
        self.assertTrue(inspection.valid)
        self.assertIsNotNone(inspection.intensity_statistics)

        # Slice extraction along all 3 orthogonal axes
        axial_slice = volume.data[:, :, 12]
        coronal_slice = volume.data[:, 20, :]
        sagittal_slice = volume.data[16, :, :]

        self.assertEqual(axial_slice.shape, (33, 41))
        self.assertEqual(coronal_slice.shape, (33, 25))
        self.assertEqual(sagittal_slice.shape, (41, 25))

    def test_3d_volume_loading_mni152(self) -> None:
        vol_path = self.demo_3d_dir / "mni152_brain" / "mni152.nii.gz"
        self.assertTrue(vol_path.is_file())

        volume = load_nifti(vol_path)
        self.assertEqual(volume.shape, (207, 256, 215))
        self.assertEqual(len(volume.spacing), 3)
        self.assertIsNotNone(volume.affine)

        inspection = inspect_nifti(vol_path)
        self.assertTrue(inspection.valid)
        self.assertGreater(inspection.intensity_statistics["max"], inspection.intensity_statistics["min"])


class TestStep12bWebApiRealInputs(unittest.TestCase):
    """Verify Web API with real demonstration inputs and honest status reporting."""

    def setUp(self) -> None:
        self.config = load_config()
        self.app = create_app(self.config)
        self.client = self.app.test_client()
        self.project_root = Path(__file__).parent.parent
        self.demo_2d_dir = self.project_root / "data" / "demo_inputs" / "2d"
        self.demo_3d_dir = self.project_root / "data" / "demo_inputs" / "3d"

    def test_health_endpoint(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json.get("status"), "ok")

    def test_input_listing_endpoints(self) -> None:
        resp_2d = self.client.get("/api/input/2d")
        self.assertEqual(resp_2d.status_code, 200)
        self.assertEqual(resp_2d.json.get("status"), "ready")
        self.assertEqual(resp_2d.json.get("count"), 12)

        resp_3d = self.client.get("/api/input/3d")
        self.assertEqual(resp_3d.status_code, 200)
        self.assertEqual(resp_3d.json.get("status"), "ready")
        self.assertEqual(resp_3d.json.get("count"), 2)

    def test_dataset_status_endpoint_honesty(self) -> None:
        resp = self.client.get("/api/datasets")
        self.assertEqual(resp.status_code, 200)
        data = resp.json

        # Classification has real 7200 images
        self.assertEqual(data["classification"]["status"], "ready")
        self.assertEqual(data["classification"]["total_images"], 7200)

        # Segmentation has not been imported yet (no fake subjects)
        self.assertEqual(data["segmentation"]["status"], "not_imported")
        self.assertEqual(data["segmentation"]["subjects"], 0)

    def test_real_2d_upload_and_analyze(self) -> None:
        sample_file = self.demo_2d_dir / "glioma" / "Tr-gl_157.jpg"
        with open(sample_file, "rb") as f:
            resp = self.client.post(
                "/api/upload",
                data={"file": (f, "Tr-gl_157.jpg")},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp.status_code, 200)
        upload_data = resp.json.get("upload", {})
        self.assertEqual(upload_data.get("file_type"), "2d_image")
        self.assertTrue(upload_data.get("validation", {}).get("valid"))

        # Analyze must state requires_checkpoint without faking predictions
        analyze_resp = self.client.post("/api/analyze")
        self.assertEqual(analyze_resp.status_code, 503)
        self.assertEqual(analyze_resp.json.get("status"), "requires_checkpoint")
        self.assertIn("not loaded", analyze_resp.json.get("classification", {}).get("message", "").lower())

    def test_real_3d_upload_and_slices(self) -> None:
        vol_file = self.demo_3d_dir / "nibabel_anatomical" / "anatomical.nii"
        with open(vol_file, "rb") as f:
            resp = self.client.post(
                "/api/upload",
                data={"file": (f, "anatomical.nii")},
                content_type="multipart/form-data",
            )
        self.assertEqual(resp.status_code, 200)
        upload_data = resp.json.get("upload", {})
        self.assertEqual(upload_data.get("file_type"), "3d_volume")
        vol_id = upload_data.get("volume_id")
        self.assertIsNotNone(vol_id)

        meta = upload_data.get("metadata", {})
        self.assertEqual(meta.get("shape"), [33, 41, 25])
        self.assertEqual(meta.get("voxel_spacing"), [2.0, 2.0, 2.0])

        # Metadata endpoint
        meta_resp = self.client.get(f"/api/volume/{vol_id}/metadata")
        self.assertEqual(meta_resp.status_code, 200)
        self.assertEqual(meta_resp.json.get("volume_id"), vol_id)

        # Slice endpoints: axial, coronal, sagittal
        for axis, idx in [("axial", 12), ("coronal", 20), ("sagittal", 16)]:
            slice_resp = self.client.get(f"/api/volume/{vol_id}/slice/{axis}/{idx}")
            self.assertEqual(slice_resp.status_code, 200)
            self.assertEqual(slice_resp.content_type, "image/png")
            self.assertGreater(len(slice_resp.data), 0)

        # Invalid axis returns 400
        invalid_axis = self.client.get(f"/api/volume/{vol_id}/slice/oblique/0")
        self.assertEqual(invalid_axis.status_code, 400)

        # Out-of-bounds index returns 400
        oob_slice = self.client.get(f"/api/volume/{vol_id}/slice/axial/999")
        self.assertEqual(oob_slice.status_code, 400)

    def test_unsupported_upload_rejected(self) -> None:
        fake_txt = io.BytesIO(b"Hello world")
        resp = self.client.post(
            "/api/upload",
            data={"file": (fake_txt, "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported file type", resp.json.get("message", ""))


if __name__ == "__main__":
    unittest.main()
