"""Shared configuration, paths and small utilities.

This module centralises project-wide constants (directories, training
hyper-parameters, model input shape) and helpers used across the package
so that every other module stays consistent and easy to configure.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATASET_DIR: Path = PROJECT_ROOT / "dataset"
MODELS_DIR: Path = PROJECT_ROOT / "models"
IMAGES_DIR: Path = PROJECT_ROOT / "images"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Dataset / model constants
# ---------------------------------------------------------------------------
IMG_HEIGHT: int = 28
IMG_WIDTH: int = 28
IMG_CHANNELS: int = 1
IMG_SHAPE: tuple[int, int, int] = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

NUM_CLASSES: int = 10
CLASS_NAMES: tuple[str, ...] = tuple(str(i) for i in range(NUM_CLASSES))

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
SEED: int = 42
EPOCHS: int = 15
BATCH_SIZE: int = 32
VALIDATION_SPLIT: float = 0.2
LEARNING_RATE: float = 1e-3

# ---------------------------------------------------------------------------
# Model file names
# ---------------------------------------------------------------------------
MODEL_KERAS: Path = MODELS_DIR / "model.keras"
MODEL_H5: Path = MODELS_DIR / "model.h5"


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for NumPy, Python and (optionally) TensorFlow.

    Args:
        seed: The integer seed used to make training reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # TensorFlow is optional (e.g. for unit tests of pure helpers).
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dirs() -> None:
    """Create all output directories if they do not exist yet."""
    for directory in (DATASET_DIR, MODELS_DIR, IMAGES_DIR, LOGS_DIR, NOTEBOOKS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
