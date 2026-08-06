"""Data loading and preprocessing for the MNIST handwritten digit dataset.

Responsibilities:
    * Download MNIST automatically (Keras caches it under ``dataset/``).
    * Print useful dataset information.
    * Normalise pixel values to the ``[0, 1]`` range.
    * Reshape images to the CNN input shape ``(28, 28, 1)``.
    * Optionally one-hot encode the labels.
    * Show a grid of sample images with their true labels.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from src.utils import (
    DATASET_DIR,
    IMG_SHAPE,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_CLASSES,
    ensure_dirs,
)

def load_data() -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """Download (if needed) and load the MNIST dataset.

    The raw ``mnist.npz`` archive is cached under ``dataset/``. If it is
    already present the network is not contacted again.

    Returns:
        ``((x_train, y_train), (x_test, y_test))`` raw numpy arrays.
        Training images have shape ``(60000, 28, 28)``, test images
        ``(10000, 28, 28)``; labels are integer digits in ``[0, 9]``.
    """
    ensure_dirs()
    mnist_file = DATASET_DIR / "mnist.npz"

    if not mnist_file.exists():
        print("MNIST not found locally, downloading...")
        fpath = keras.utils.get_file(
            fname="mnist.npz",
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            file_hash="731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1",
            cache_subdir="",
            cache_dir=str(DATASET_DIR),
        )
        print(f"MNIST downloaded and cached at: {fpath}")
    else:
        print(f"MNIST found locally at: {mnist_file}")

    with np.load(mnist_file, allow_pickle=True) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    return (x_train, y_train), (x_test, y_test)


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Normalise and reshape raw MNIST images.

    Args:
        images: Array of shape ``(N, 28, 28)`` with dtype uint8 (0-255).

    Returns:
        Array of shape ``(N, 28, 28, 1)`` with float32 values in ``[0, 1]``.
    """
    images = images.astype("float32") / 255.0
    images = images.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
    return images


def one_hot_encode(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Convert integer labels to one-hot vectors.

    Args:
        labels: Array of integer labels in ``[0, num_classes)``.
        num_classes: Number of output classes (10 for digits).

    Returns:
        Array of shape ``(N, num_classes)``.
    """
    return keras.utils.to_categorical(labels, num_classes)


def preprocess_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    one_hot: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Full pipeline: normalise, reshape and (optionally) one-hot encode.

    Args:
        x_train: Raw training images ``(60000, 28, 28)``.
        y_train: Raw training labels ``(60000,)``.
        x_test: Raw test images ``(10000, 28, 28)``.
        y_test: Raw test labels ``(10000,)``.
        one_hot: Whether to one-hot encode labels.

    Returns:
        Processed ``(x_train, y_train, x_test, y_test)`` tuples ready for the CNN.
    """
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    if one_hot:
        y_train = one_hot_encode(y_train)
        y_test = one_hot_encode(y_test)

    return x_train, y_train, x_test, y_test


def display_dataset_info(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print a short summary of the dataset to the console."""
    print("=" * 60)
    print("MNIST DATASET INFORMATION")
    print("=" * 60)
    print(f"Image size            : {IMG_HEIGHT} x {IMG_WIDTH} pixels (grayscale)")
    print(f"Number of classes     : {NUM_CLASSES} (0-9)")
    print(f"Training samples      : {x_train.shape[0]}")
    print(f"Test samples          : {x_test.shape[0]}")
    print(f"Training image shape  : {x_train.shape}")
    print(f"Training label shape  : {y_train.shape}")
    print(f"Test image shape      : {x_test.shape}")
    print(f"Test label shape      : {y_test.shape}")
    print(f"Pixel value range     : {x_train.min():.2f} - {x_train.max():.2f} (after scaling)")
    unique, counts = np.unique(np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train, return_counts=True)
    print("Class distribution    : " + ", ".join(f"{int(u)}:{int(c)}" for u, c in zip(unique, counts)))
    print("=" * 60)


def show_sample_images(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_samples: int = 25,
    save_path: str | None = None,
) -> plt.Figure:
    """Display a grid of sample images with their labels.

    Args:
        x_train: Preprocessed training images.
        y_train: Preprocessed (one-hot) or integer training labels.
        num_samples: Number of images to plot (perfect square preferred).
        save_path: Optional path to save the figure.

    Returns:
        The matplotlib ``Figure`` object.
    """
    labels = np.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train
    grid = int(np.sqrt(num_samples))
    if grid * grid < num_samples:
        grid += 1

    fig, axes = plt.subplots(grid, grid, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(x_train[i].squeeze(), cmap="gray")
        axes[i].set_title(f"Label: {labels[i]}", fontsize=10)
        axes[i].axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Sample MNIST Images", fontsize=14)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sample images saved to: {save_path}")

    return fig
