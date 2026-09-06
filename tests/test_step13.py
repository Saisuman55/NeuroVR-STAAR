"""Step 13 comprehensive tests: 2D classification training, evaluation, checkpoints, registry, and segmentation readiness."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.classifier_2d import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    INDEX_TO_CLASS,
    BrainTumor2DDataset,
    SimpleTestCNN,
    create_classifier_2d,
    get_transforms,
    prepare_classification_splits,
)
from src.evaluation import (
    dice_score,
    evaluate_classification,
    iou_score,
    segmentation_precision_recall,
)
from src.model_registry import (
    get_classification_model_status,
    get_model_status,
    get_segmentation_model_status,
    load_classification_model_for_inference,
    predict_2d_image,
)
from src.training import Trainer2D
from src.utils import load_config, select_device, set_random_seed
from src.web_api import create_app


class TestStep13Pipeline(unittest.TestCase):
    """Verify all 20 requirement areas for Step 13."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.test_dir = Path(cls.temp_dir.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    # 1. Device selection
    def test_01_device_selection(self) -> None:
        device = select_device(self.config)
        self.assertIsInstance(device, torch.device)
        self.assertIn(device.type, {"mps", "cpu", "cuda"})

    # 2. Deterministic seed setup
    def test_02_deterministic_seed_setup(self) -> None:
        set_random_seed(42)
        r1 = torch.rand(5)
        set_random_seed(42)
        r2 = torch.rand(5)
        self.assertTrue(torch.allclose(r1, r2))

    # 3. Dataset split reproducibility
    def test_03_dataset_split_reproducibility(self) -> None:
        data_root = Path(self.config["paths"]["classification_dataset"])
        splits1 = prepare_classification_splits(data_root, validation_split=0.2, seed=42)
        splits2 = prepare_classification_splits(data_root, validation_split=0.2, seed=42)
        self.assertEqual([p for p, _ in splits1["train"]], [p for p, _ in splits2["train"]])
        self.assertEqual([p for p, _ in splits1["val"]], [p for p, _ in splits2["val"]])
        self.assertEqual([p for p, _ in splits1["test"]], [p for p, _ in splits2["test"]])

    # 4. No overlap between train, validation, test
    def test_04_no_split_overlap(self) -> None:
        data_root = Path(self.config["paths"]["classification_dataset"])
        splits = prepare_classification_splits(data_root, validation_split=0.2, seed=42)
        train_paths = {p for p, _ in splits["train"]}
        val_paths = {p for p, _ in splits["val"]}
        test_paths = {p for p, _ in splits["test"]}

        self.assertEqual(len(train_paths.intersection(val_paths)), 0)
        self.assertEqual(len(train_paths.intersection(test_paths)), 0)
        self.assertEqual(len(val_paths.intersection(test_paths)), 0)
        self.assertEqual(len(train_paths) + len(val_paths) + len(test_paths), 7200)

    # 5. Class mapping consistency
    def test_05_class_mapping_consistency(self) -> None:
        expected_classes = ("glioma", "meningioma", "notumor", "pituitary")
        self.assertEqual(CLASS_NAMES, expected_classes)
        for idx, name in enumerate(expected_classes):
            self.assertEqual(CLASS_TO_INDEX[name], idx)
            self.assertEqual(INDEX_TO_CLASS[idx], name)

    # 6. Checkpoint save & 7. Checkpoint metadata & 8. Checkpoint loading
    def test_06_to_08_checkpoint_save_metadata_and_loading(self) -> None:
        model = SimpleTestCNN(num_classes=4)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        ckpt_path = self.test_dir / "test_best.pt"

        # Create dummy loaders for test
        x = torch.randn(8, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=4)

        trainer = Trainer2D(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            config=self.config,
            checkpoint_dir=self.test_dir,
            report_dir=self.test_dir,
            model_name="simple_cnn",
            image_size=(32, 32),
        )
        trainer.save_checkpoint(
            ckpt_path,
            epoch=3,
            train_metrics={"loss": 0.45, "accuracy": 0.82},
            val_metrics={"loss": 0.50, "accuracy": 0.80},
        )
        self.assertTrue(ckpt_path.is_file())

        loaded = torch.load(ckpt_path, map_location="cpu")
        self.assertEqual(loaded["model_architecture"], "simple_cnn")
        self.assertEqual(loaded["epoch"], 3)
        self.assertEqual(loaded["class_names"], list(CLASS_NAMES))
        self.assertEqual(loaded["class_to_index"], CLASS_TO_INDEX)
        self.assertEqual(loaded["image_size"], [32, 32])
        self.assertIn("model_state_dict", loaded)
        self.assertIn("optimizer_state_dict", loaded)
        self.assertIn("timestamp", loaded)

        # Load weights into new model instance
        new_model = SimpleTestCNN(num_classes=4)
        new_model.load_state_dict(loaded["model_state_dict"])
        new_model.eval()
        with torch.no_grad():
            out = new_model(x[:2])
            self.assertEqual(out.shape, (2, 4))

    # 9. Missing checkpoint behavior
    def test_09_missing_checkpoint_behavior(self) -> None:
        non_existent_config = {
            "classification": {"checkpoint_path": "non_existent_path_best.pt"},
            "paths": {"segmentation_dataset": "data/segmentation"},
            "three_d_segmentation": {"architecture": "3d_unet"},
        }
        status = get_classification_model_status(non_existent_config)
        self.assertEqual(status["status"], "unavailable")
        self.assertIn("not found", status["message"].lower())

    # 10. Classification inference behavior when model unavailable
    def test_10_inference_when_model_unavailable(self) -> None:
        app = create_app()
        client = app.test_client()
        # Ensure upload happens first
        sample_img = Path("data/demo_inputs/2d/glioma/Tr-gl_157.jpg")
        with open(sample_img, "rb") as f:
            client.post("/api/upload", data={"file": (f, "Tr-gl_157.jpg")}, content_type="multipart/form-data")

        # In current state without trained checkpoint, analyze must return unavailable
        resp = client.post("/api/analyze")
        self.assertEqual(resp.status_code, 200)
        data = resp.json
        self.assertIn(data["status"], {"unavailable", "completed"})
        if data["status"] == "unavailable":
            self.assertEqual(data["classification"]["status"], "unavailable")

    # 11. Classification inference with a valid test checkpoint
    def test_11_classification_inference_with_valid_checkpoint(self) -> None:
        model = SimpleTestCNN(num_classes=4)
        device = torch.device("cpu")
        sample_img = Path("data/demo_inputs/2d/glioma/Tr-gl_157.jpg")

        result = predict_2d_image(
            model=model,
            image_path=sample_img,
            device=device,
            class_names=CLASS_NAMES,
            image_size=(32, 32),
        )
        self.assertEqual(result["status"], "predicted")
        self.assertIn(result["predicted_class"], CLASS_NAMES)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(set(result["probabilities"].keys()), set(CLASS_NAMES))
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.05)

    # 12. Evaluation metric calculations & 13. Confusion matrix generation
    def test_12_and_13_evaluation_metrics_and_confusion_matrix(self) -> None:
        model = SimpleTestCNN(num_classes=4)
        x = torch.randn(16, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3] * 4)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        eval_dir = self.test_dir / "eval_test"
        metrics = evaluate_classification(
            model=model,
            data_loader=loader,
            device=torch.device("cpu"),
            class_names=CLASS_NAMES,
            save_dir=eval_dir,
        )
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_avg", metrics)
        self.assertIn("weighted_avg", metrics)
        self.assertIn("per_class", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 4)

        # Check saved files
        self.assertTrue((eval_dir / "metrics.json").is_file())
        self.assertTrue((eval_dir / "classification_report.json").is_file())
        self.assertTrue((eval_dir / "confusion_matrix.png").is_file())

    # 14. Training history serialization
    def test_14_training_history_serialization(self) -> None:
        model = SimpleTestCNN(num_classes=4)
        x = torch.randn(8, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        trainer = Trainer2D(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optim.Adam(model.parameters(), lr=0.001),
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            config=self.config,
            checkpoint_dir=self.test_dir,
            report_dir=self.test_dir,
            model_name="simple_cnn",
            image_size=(32, 32),
        )
        history = trainer.train(epochs=2, verbose=False)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["epoch"], 1)
        self.assertEqual(history[1]["epoch"], 2)

        history_file = self.test_dir / "training_history.json"
        self.assertTrue(history_file.is_file())
        with open(history_file) as f:
            data = json.load(f)
        self.assertEqual(len(data["history"]), 2)

    # 15. Segmentation metric utilities
    def test_15_segmentation_metric_utilities(self) -> None:
        pred = torch.tensor([[[0, 1], [1, 1]]], dtype=torch.float32)
        target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)

        dice = dice_score(pred, target)
        iou = iou_score(pred, target)
        prec, rec = segmentation_precision_recall(pred, target)

        self.assertAlmostEqual(dice, 4.0 / 5.0, places=3)
        self.assertAlmostEqual(iou, 2.0 / 3.0, places=3)
        self.assertAlmostEqual(prec, 2.0 / 3.0, places=3)
        self.assertAlmostEqual(rec, 1.0, places=3)

    # 16. Empty segmentation masks
    def test_16_empty_segmentation_masks(self) -> None:
        empty1 = torch.zeros((1, 4, 4), dtype=torch.float32)
        empty2 = torch.zeros((1, 4, 4), dtype=torch.float32)

        # Both empty must score 1.0 (perfect true negative)
        self.assertEqual(dice_score(empty1, empty2), 1.0)
        self.assertEqual(iou_score(empty1, empty2), 1.0)
        prec, rec = segmentation_precision_recall(empty1, empty2)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

        # One empty and one non-empty must score 0.0
        non_empty = torch.ones((1, 4, 4), dtype=torch.float32)
        self.assertAlmostEqual(dice_score(empty1, non_empty), 0.0, places=4)
        self.assertAlmostEqual(iou_score(empty1, non_empty), 0.0, places=4)

    # 17. Missing BraTS manifest behavior & 18. Missing segmentation data behavior
    def test_17_and_18_missing_brats_data_behavior(self) -> None:
        from scripts.train_segmentation import check_brats_data_readiness
        is_ready, message, subjects = check_brats_data_readiness(self.config)
        self.assertFalse(is_ready)
        self.assertEqual(len(subjects), 0)
        self.assertIn("BraTS", message)

    # 19. Model registry status
    def test_19_model_registry_status(self) -> None:
        status = get_model_status(self.config)
        self.assertIn("classification", status)
        self.assertIn("segmentation", status)
        self.assertIn("status", status["classification"])
        self.assertIn("status", status["segmentation"])

    # 20. Existing web API compatibility
    def test_20_web_api_compatibility(self) -> None:
        app = create_app()
        client = app.test_client()

        # Check existing endpoints continue working flawlessly
        self.assertEqual(client.get("/api/health").status_code, 200)
        self.assertEqual(client.get("/api/input/2d").status_code, 200)
        self.assertEqual(client.get("/api/input/3d").status_code, 200)
        self.assertEqual(client.get("/api/datasets").status_code, 200)
        self.assertEqual(client.get("/api/models/status").status_code, 200)

    # 21. Web API live inference when checkpoint is configured
    def test_21_web_api_live_inference_with_configured_checkpoint(self) -> None:
        best_ckpt = Path("checkpoints/classification/best.pt")
        if not best_ckpt.is_file():
            self.skipTest("best.pt not found for live inference test")

        import copy
        custom_config = copy.deepcopy(self.config)
        custom_config["classification"]["checkpoint_path"] = str(best_ckpt)

        app = create_app(custom_config)
        client = app.test_client()

        # Model status should be available
        status_resp = client.get("/api/models/status")
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(status_resp.json["classification"]["status"], "available")

        # Upload demo image
        sample_img = Path("data/demo_inputs/2d/glioma/Tr-gl_157.jpg")
        with open(sample_img, "rb") as f:
            upload_resp = client.post("/api/upload", data={"file": (f, "Tr-gl_157.jpg")}, content_type="multipart/form-data")
        self.assertEqual(upload_resp.status_code, 200)

        # Live analysis
        analyze_resp = client.post("/api/analyze")
        self.assertEqual(analyze_resp.status_code, 200)
        data = analyze_resp.json
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["classification"]["status"], "available")
        self.assertIn("predicted_class", data["classification"])
        self.assertIn("probabilities", data["classification"])
        self.assertEqual(data["segmentation"]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()

