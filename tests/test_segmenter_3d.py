import unittest

import torch

from src.segmenter_3d import (
    bce_loss,
    combined_bce_dice_loss,
    create_segmenter_3d,
    dice_coefficient,
    dice_loss,
    logits_to_probabilities,
    parameter_count,
    predict_mask_from_logits,
    segmentation_metrics,
    validate_segmenter_config,
)
from src.utils import load_config, select_device


class Segmenter3DTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("config/config.yaml")
        cls.model = create_segmenter_3d(cls.config)
        cls.inputs = torch.randn(1, 4, 16, 16, 16)

    def test_model_construction_and_channels(self) -> None:
        self.assertEqual(self.model.input_channels, 4)
        self.assertEqual(self.model.output_channels, 1)
        self.assertGreater(parameter_count(self.model), 0)
        self.assertEqual(parameter_count(self.model), parameter_count(self.model, trainable_only=True))

    def test_cpu_forward_shape(self) -> None:
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.inputs)
        self.assertEqual(output.shape, (1, 1, 16, 16, 16))

    def test_invalid_input_shape_and_channels(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 4, 16, 16))
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 1, 16, 16, 16))

    def test_mps_forward_if_available(self) -> None:
        """3D U-Net forward pass on MPS with CPU fallback for unsupported ops.

        MaxPool3d (aten::max_pool3d_with_indices) is not yet implemented on the
        MPS backend. PYTORCH_ENABLE_MPS_FALLBACK=1 routes those ops to CPU
        transparently, which is the project's documented Apple Silicon strategy.
        The architecture is not changed to work around this backend limitation.
        """
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        model = create_segmenter_3d(self.config).to("mps")
        with torch.no_grad():
            output = model(self.inputs.to("mps"))
        self.assertEqual(output.shape, (1, 1, 16, 16, 16))

    def test_probability_and_thresholding(self) -> None:
        logits = torch.tensor([[[[[-2.0, 2.0]]]]])
        probabilities = logits_to_probabilities(logits)
        mask = predict_mask_from_logits(logits, threshold=0.5)
        self.assertLess(float(probabilities[0, 0, 0, 0, 0]), 0.5)
        self.assertEqual(mask.tolist(), [[[[[0.0, 1.0]]]]])

    def test_losses(self) -> None:
        logits = torch.zeros(1, 1, 2, 2, 2)
        target = torch.zeros_like(logits)
        self.assertGreaterEqual(float(bce_loss(logits, target)), 0.0)
        self.assertGreaterEqual(float(dice_loss(logits, target)), 0.0)
        self.assertGreaterEqual(float(combined_bce_dice_loss(logits, target)), 0.0)

    def test_metrics_and_empty_edge_cases(self) -> None:
        target = torch.tensor([[[[[1.0, 0.0]]]]])
        self.assertEqual(dice_coefficient(target, target).item(), 1.0)
        metrics = segmentation_metrics(target, target)
        self.assertEqual(metrics, {name: 1.0 for name in ("dice", "iou", "precision", "recall", "voxel_accuracy")})
        empty = torch.zeros_like(target)
        self.assertEqual(segmentation_metrics(empty, empty)["dice"], 1.0)
        self.assertEqual(segmentation_metrics(empty, target)["dice"], 0.0)
        self.assertEqual(segmentation_metrics(target, empty)["precision"], 0.0)

    def test_configuration_loading_and_invalid_configuration(self) -> None:
        self.assertEqual(self.config["three_d_segmentation"]["architecture"], "3d_unet")
        with self.assertRaises(ValueError):
            validate_segmenter_config({})
        invalid = {"three_d_segmentation": dict(self.config["three_d_segmentation"], base_channels=0)}
        with self.assertRaises(ValueError):
            create_segmenter_3d(invalid)

    def test_device_selection_uses_existing_utility(self) -> None:
        device = select_device(self.config)
        self.assertIn(str(device), ("mps", "cpu"))


if __name__ == "__main__":
    unittest.main()
