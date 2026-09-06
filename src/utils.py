"""Reusable utilities for the NeuroVR pipeline."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


Config = dict[str, Any]


def load_config(config_path: str | Path = "config/config.yaml") -> Config:
    """Load a YAML configuration file and return its mapping."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in configuration file: {path}") from error

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return config


def select_device(config: Config) -> torch.device:
    """Select MPS when available and configured; otherwise return CPU.

    When MPS is selected, callers should set PYTORCH_ENABLE_MPS_FALLBACK=1
    in the environment so that operations not yet implemented on MPS (such as
    aten::max_pool3d_with_indices) fall back to CPU automatically. This is the
    project's documented Apple Silicon compatibility strategy; the 3D U-Net
    architecture is not modified to work around backend limitations.
    """
    compute = config.get("compute", {})
    primary_device = str(compute.get("primary_device", "mps")).lower()
    mps_available = torch.backends.mps.is_available()
    if primary_device == "mps" and mps_available:
        # Ensure MPS fallback is active for ops not yet supported on MPS.
        import os
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device(str(compute.get("fallback_device", "cpu")))


def set_random_seed(seed: int) -> None:
    """Set reproducible seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_output_directories(config: Config) -> list[Path]:
    """Create and return configured output and model directories."""
    paths = config.get("paths", {})
    output = config.get("output", {})
    logging_config = config.get("logging", {})
    configured_paths = [
        output.get("directory"),
        output.get("predictions_directory"),
        output.get("reports_directory"),
        paths.get("classifier_models"),
        paths.get("segmenter_models"),
        paths.get("uploads"),
        paths.get("predictions"),
        paths.get("reports"),
        logging_config.get("directory"),
    ]
    directories = []
    for configured_path in configured_paths:
        if configured_path:
            directory = Path(configured_path)
            directory.mkdir(parents=True, exist_ok=True)
            directories.append(directory)
    return list(dict.fromkeys(directories))


def setup_logging(config: Config) -> logging.Logger:
    """Configure console and file logging from the project settings."""
    logging_config = config.get("logging", {})
    level_name = str(logging_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger = logging.getLogger("neurovr")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if logging_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    log_file = logging_config.get("file")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_environment_info(device: torch.device) -> dict[str, str | bool]:
    """Return runtime versions and MPS status for diagnostics."""
    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "selected_device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }