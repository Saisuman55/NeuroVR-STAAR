import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data_loader import (
    SUPPORTED_IMAGE_EXTENSIONS,
    inspect_classification_dataset,
    inspect_segmentation_dataset,
    split_dataset,
)


EXPECTED_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")


class DataLoaderTests(unittest.TestCase):
    def make_image(self, path: Path, color: int = 0, size: tuple[int, int] = (8, 8)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full(size[::-1], color, dtype=np.uint8)).save(path)

    def test_classification_class_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "glioma" / "one.jpg")
            report = inspect_classification_dataset(root, EXPECTED_CLASSES)
            self.assertEqual(report.class_counts["glioma"], 1)
            self.assertIn("meningioma", report.missing_classes)

    def test_supported_image_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extension in SUPPORTED_IMAGE_EXTENSIONS:
                self.make_image(root / "glioma" / f"sample{extension}")
            report = inspect_classification_dataset(root, EXPECTED_CLASSES)
            self.assertEqual(report.valid_images, 3)

    def test_invalid_corrupted_file_handling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupted = root / "glioma" / "broken.png"
            corrupted.parent.mkdir()
            corrupted.write_bytes(b"not an image")
            report = inspect_classification_dataset(root, EXPECTED_CLASSES)
            self.assertEqual(report.invalid_images, 1)
            self.assertFalse(report.samples[0].valid)

    def test_unexpected_class_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "healthy" / "sample.png")
            report = inspect_classification_dataset(root, EXPECTED_CLASSES)
            self.assertEqual(report.unexpected_classes, ["healthy"])
            self.assertEqual(report.class_counts["healthy"], 1)

    def test_segmentation_image_mask_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "scan.jpg")
            self.make_image(root / "scan_mask.png", 255)
            report = inspect_segmentation_dataset(root)
            self.assertEqual(len(report.pairs), 1)
            self.assertTrue(report.pairs[0].valid)
            self.assertEqual(report.pairs[0].key, "scan")

    def test_missing_mask_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "without.jpg")
            report = inspect_segmentation_dataset(root)
            self.assertEqual(report.images_without_masks, ["without"])
            self.assertEqual(report.valid_pairs, 0)

    def test_missing_image_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "without_image_mask.png", 255)
            report = inspect_segmentation_dataset(root)
            self.assertEqual(report.masks_without_images, ["without_image"])

    def test_binary_mask_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_image(root / "scan.jpg")
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:4, 2:4] = 255
            Image.fromarray(mask).save(root / "scan_mask.png")
            report = inspect_segmentation_dataset(root)
            self.assertTrue(report.pairs[0].mask_is_binary)
            self.assertEqual(report.pairs[0].mask_values, (0, 255))

    def test_empty_and_non_empty_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("empty", 0), ("tumor", 255)):
                self.make_image(root / f"{name}.jpg")
                self.make_image(root / f"{name}_mask.png", value)
            report = inspect_segmentation_dataset(root)
            self.assertEqual(report.empty_mask_count, 1)
            self.assertEqual(report.non_empty_mask_count, 1)
            self.assertEqual(report.foreground_pixels, 64)

    def test_deterministic_dataset_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("z.png", "a.png", "m.png"):
                self.make_image(root / "glioma" / name)
            report = inspect_classification_dataset(root, EXPECTED_CLASSES)
            self.assertEqual([sample.path.name for sample in report.samples], ["a.png", "m.png", "z.png"])

    def test_deterministic_splitting(self) -> None:
        samples = tuple(range(20))
        first = split_dataset(samples, seed=42)
        second = split_dataset(samples, seed=42)
        self.assertEqual(first, second)

    def test_split_keeps_pairs_together(self) -> None:
        pairs = tuple((f"image_{index}", f"mask_{index}") for index in range(20))
        split = split_dataset(pairs, seed=42)
        self.assertEqual(set(split.train) | set(split.validation) | set(split.test), set(pairs))
        for partition in (split.train, split.validation, split.test):
            self.assertTrue(all(image.replace("image", "mask") == mask for image, mask in partition))

    def test_no_duplicate_samples_across_splits(self) -> None:
        split = split_dataset(tuple(range(30)), seed=7)
        partitions = [set(split.train), set(split.validation), set(split.test)]
        self.assertEqual(len(set.union(*partitions)), 30)
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])


if __name__ == "__main__":
    unittest.main()
