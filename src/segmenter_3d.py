"""Lightweight 3D U-Net baseline and binary segmentation utilities."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


Config = dict[str, Any]


def _normalization(name: str, channels: int) -> nn.Module:
    """Create the configured 3D feature normalization layer."""
    normalized = name.lower()
    if normalized in {"batchnorm3d", "batch_norm_3d", "batchnorm"}:
        return nn.BatchNorm3d(channels)
    if normalized in {"instancenorm3d", "instance_norm_3d", "instancenorm"}:
        return nn.InstanceNorm3d(channels, affine=True)
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 3D normalization: {name}")


def _activation(name: str) -> nn.Module:
    """Create the configured activation module."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    if normalized in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 3D activation: {name}")


class ConvBlock3D(nn.Module):
    """Two convolution-normalization-activation layers for a U-Net stage."""

    def __init__(self, in_channels: int, out_channels: int, normalization: str, activation: str, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for input_width, output_width in ((in_channels, out_channels), (out_channels, out_channels)):
            layers.extend((
                nn.Conv3d(input_width, output_width, kernel_size=3, padding=1, bias=False),
                _normalization(normalization, output_width),
                _activation(activation),
            ))
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the convolutional feature block."""
        return self.layers(inputs)


class EncoderBlock3D(nn.Module):
    """U-Net encoder stage that returns features for a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, normalization: str, activation: str, dropout: float) -> None:
        super().__init__()
        self.block = ConvBlock3D(in_channels, out_channels, normalization, activation, dropout)
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return stage features and their downsampled representation."""
        features = self.block(inputs)
        return features, self.downsample(features)


class DecoderBlock3D(nn.Module):
    """U-Net decoder stage with learned upsampling and a skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, normalization: str, activation: str, dropout: float) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock3D(out_channels + skip_channels, out_channels, normalization, activation, dropout)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        """Upsample, concatenate the skip features, and decode."""
        inputs = self.upsample(inputs)
        if inputs.shape[2:] != skip.shape[2:]:
            raise ValueError(f"Skip connection shape mismatch: {inputs.shape[2:]} versus {skip.shape[2:]}")
        return self.block(torch.cat((inputs, skip), dim=1))


class UNet3D(nn.Module):
    """A compact volumetric U-Net for binary or multi-channel segmentation."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        settings = validate_segmenter_config(config)
        input_channels = settings["input_channels"]
        output_channels = settings["output_channels"]
        base_channels = settings["base_channels"]
        normalization = settings["normalization"]
        activation = settings["activation"]
        dropout = settings["dropout"]
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.encoder1 = EncoderBlock3D(input_channels, base_channels, normalization, activation, dropout)
        self.encoder2 = EncoderBlock3D(base_channels, base_channels * 2, normalization, activation, dropout)
        self.encoder3 = EncoderBlock3D(base_channels * 2, base_channels * 4, normalization, activation, dropout)
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8, normalization, activation, dropout)
        self.decoder3 = DecoderBlock3D(base_channels * 8, base_channels * 4, base_channels * 4, normalization, activation, dropout)
        self.decoder2 = DecoderBlock3D(base_channels * 4, base_channels * 2, base_channels * 2, normalization, activation, dropout)
        self.decoder1 = DecoderBlock3D(base_channels * 2, base_channels, base_channels, normalization, activation, dropout)
        self.output = nn.Conv3d(base_channels, output_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return segmentation logits with shape [B, output_channels, D, H, W]."""
        if inputs.ndim != 5:
            raise ValueError(f"Expected [B, C, D, H, W], found {tuple(inputs.shape)}")
        if inputs.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, found {inputs.shape[1]}")
        skip1, down1 = self.encoder1(inputs)
        skip2, down2 = self.encoder2(down1)
        skip3, down3 = self.encoder3(down2)
        decoded = self.bottleneck(down3)
        decoded = self.decoder3(decoded, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder1(decoded, skip1)
        return self.output(decoded)


def validate_segmenter_config(config: Config) -> Config:
    """Validate and return the dedicated 3D segmentation settings."""
    settings = config.get("three_d_segmentation")
    if not isinstance(settings, dict):
        raise ValueError("Missing three_d_segmentation configuration")
    required = ("architecture", "input_channels", "output_channels", "base_channels", "normalization", "activation", "dropout", "threshold", "loss", "bce_weight", "dice_weight")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"Missing 3D segmentation settings: {', '.join(missing)}")
    if str(settings["architecture"]).lower() not in {"3d_unet", "unet3d", "3d-u-net"}:
        raise ValueError(f"Unsupported 3D segmentation architecture: {settings['architecture']}")
    for key in ("input_channels", "output_channels", "base_channels"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= float(settings["dropout"]) < 1:
        raise ValueError("dropout must be in [0, 1)")
    if not 0 <= float(settings["threshold"]) <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if float(settings["bce_weight"]) < 0 or float(settings["dice_weight"]) < 0:
        raise ValueError("loss weights cannot be negative")
    if float(settings["bce_weight"]) + float(settings["dice_weight"]) <= 0:
        raise ValueError("at least one loss weight must be positive")
    return settings


def create_segmenter_3d(config: Config) -> UNet3D:
    """Construct the configured 3D U-Net without loading weights or data."""
    return UNet3D(config)


def logits_to_probabilities(logits: Tensor) -> Tensor:
    """Convert binary segmentation logits to probabilities with sigmoid."""
    return torch.sigmoid(logits)


def predict_mask_from_logits(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """Convert binary logits to a float mask using an explicit threshold."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return (logits_to_probabilities(logits) >= threshold).to(dtype=logits.dtype)


def dice_coefficient(prediction: Tensor, target: Tensor, smooth: float = 1e-6) -> Tensor:
    """Compute soft/binary Dice, defining two empty masks as one."""
    prediction, target = _validate_binary_tensors(prediction, target)
    prediction = prediction.float().reshape(prediction.shape[0], -1)
    target = target.float().reshape(target.shape[0], -1)
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    score = (2 * intersection + smooth) / (denominator + smooth)
    return torch.where(denominator == 0, torch.ones_like(score), score).mean()


def dice_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Return one minus the Dice coefficient for binary logits."""
    return 1 - dice_coefficient(logits_to_probabilities(logits), target)


def bce_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Compute binary cross entropy directly from logits."""
    _validate_binary_tensors(logits, target)
    return nn.functional.binary_cross_entropy_with_logits(logits, target.float())


def combined_bce_dice_loss(logits: Tensor, target: Tensor, bce_weight: float = 0.5, dice_weight: float = 0.5) -> Tensor:
    """Combine BCE-with-logits and Dice losses using explicit weights."""
    if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
        raise ValueError("loss weights must be non-negative and not both zero")
    return bce_weight * bce_loss(logits, target) + dice_weight * dice_loss(logits, target)


def _validate_binary_tensors(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """Validate matching prediction and target tensor shapes."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction and target shapes differ: {prediction.shape} versus {target.shape}")
    if prediction.ndim < 1:
        raise ValueError("Segmentation tensors must have at least one dimension")
    return prediction, target


def segmentation_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Return Dice, IoU, precision, recall, and voxel accuracy.

    Both-empty masks score 1.0 for all metrics; one empty and one non-empty mask
    scores 0.0. This convention makes an exactly correct empty prediction explicit.
    """
    prediction, target = _validate_binary_tensors(prediction, target)
    predicted = prediction.bool()
    actual = target.bool()
    true_positive = (predicted & actual).sum().float()
    false_positive = (predicted & ~actual).sum().float()
    false_negative = (~predicted & actual).sum().float()
    true_negative = (~predicted & ~actual).sum().float()
    predicted_count = predicted.sum()
    actual_count = actual.sum()
    union = (predicted | actual).sum()
    both_empty = predicted_count == 0 and actual_count == 0
    if both_empty:
        return {name: 1.0 for name in ("dice", "iou", "precision", "recall", "voxel_accuracy")}
    precision = true_positive / (true_positive + false_positive) if predicted_count else torch.tensor(0.0, device=prediction.device)
    recall = true_positive / (true_positive + false_negative) if actual_count else torch.tensor(0.0, device=prediction.device)
    return {
        "dice": float((2 * true_positive / (predicted_count + actual_count)).item()),
        "iou": float((true_positive / union).item()) if union else 0.0,
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "voxel_accuracy": float(((true_positive + true_negative) / prediction.numel()).item()),
    }


def parameter_count(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters, optionally restricting to trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if not trainable_only or parameter.requires_grad)