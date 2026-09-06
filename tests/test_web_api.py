import copy
import io
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from src.utils import load_config
from src.web_api import AnalysisState, create_app


class WebApiTests(unittest.TestCase):
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

    def png_bytes(self) -> io.BytesIO:
        stream = io.BytesIO()
        Image.new("RGB", (4, 4), color=(20, 40, 60)).save(stream, format="PNG")
        stream.seek(0)
        return stream

    def test_app_starts_and_main_page_loads(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NEUROVR", response.data)
        self.assertIn(b"3D tumor mesh unavailable", response.data)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")

    def test_missing_file_handling(self) -> None:
        response = self.client.post("/api/upload", data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("No file", response.json["message"])

    def test_unsupported_file_rejection(self) -> None:
        response = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(b"data"), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported", response.json["message"])

    def test_upload_validation_and_local_storage(self) -> None:
        response = self.client.post(
            "/api/upload",
            data={"file": (self.png_bytes(), "study.png")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["upload"]["file_type"], "2d_image")
        self.assertTrue(response.json["upload"]["validation"]["valid"])

    def test_nifti_upload_validation_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), str(path))
            with path.open("rb") as nifti_file:
                response = self.client.post(
                    "/api/upload",
                    data={"file": (nifti_file, "study.nii.gz")},
                    content_type="multipart/form-data",
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["upload"]["file_type"], "3d_volume")
        self.assertTrue(response.json["upload"]["validation"]["valid"])

    def test_status_results_and_no_model_state(self) -> None:
        response = self.client.get("/api/status")
        self.assertEqual(response.json["status"], "idle")
        self.client.post(
            "/api/upload",
            data={"file": (self.png_bytes(), "study.png")},
            content_type="multipart/form-data",
        )
        response = self.client.post("/api/analyze")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["status"], "requires_checkpoint")
        self.assertEqual(response.json["classification"]["status"], "requires_checkpoint")
        response = self.client.get("/api/results")
        self.assertEqual(response.json["status"], "requires_analysis")

    def test_analyze_requires_upload(self) -> None:
        response = self.client.post("/api/analyze")
        self.assertEqual(response.status_code, 400)

    def test_report_is_blocked_without_analysis(self) -> None:
        response = self.client.post("/api/report")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json["status"], "analysis_required")


if __name__ == "__main__":
    unittest.main()
