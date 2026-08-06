"""Prediction utilities for both MNIST test images and custom images.

Custom images are converted to the same format as MNIST (28x28 grayscale,
pixel values in [0, 1]) before being passed to the model. Both matplotlib
(slow but dependency-free) and OpenCV (fast) decoders are supported.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from tensorflow import keras

from src.utils import IMG_HEIGHT, IMG_WIDTH, MODEL_KERAS, NUM_CLASSES


def load_model(path=MODEL_KERAS) -> keras.Model:
    """Load a trained Keras model from disk."""
    model = keras.models.load_model(path)
    print(f"Model loaded from: {path}")
    return model


def _read_image_bytes(path: Path) -> np.ndarray:
    """Read an image file into a numpy BGR/BW array.

    Args:
        path: Path to the image file.

    Returns:
        A numpy array of the raw image.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if CV2_AVAILABLE:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"OpenCV could not decode the image: {path}")
        return image

    # Fallback: decode with matplotlib.
    from matplotlib.image import imread

    return imread(path)


def preprocess_custom_image(image_bytes: bytes | np.ndarray) -> np.ndarray:
    """Convert a raw uploaded image to MNIST-compatible input.

    Pipeline: decode -> grayscale -> crop to the digit's bounding box ->
    resize to 28x28 -> invert if needed (white digit on black background
    is expected) -> scale to [0, 1].

    Args:
        image_bytes: Raw bytes of the image, or an already decoded array.

    Returns:
        Array of shape ``(1, 28, 28, 1)`` ready for ``model.predict``.
    """
    if isinstance(image_bytes, (bytes, bytearray)):
        if CV2_AVAILABLE:
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode the uploaded image bytes.")
        else:
            image = plt.imread(io.BytesIO(image_bytes))
    else:
        image = image_bytes

    if CV2_AVAILABLE:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Binarise so we can find the digit and normalise it to fill the frame.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        array = binary.astype("float32") / 255.0
    else:
        gray = np.mean(image, axis=-1) if image.ndim == 3 else image
        gray = np.array(gray * 255.0, dtype="uint8")
        binary = gray
        array = binary.astype("float32") / 255.0

    # MNIST uses white digits on a black background.
    if np.mean(array) > 0.5:
        array = 1.0 - array

    array = _crop_to_content(array)
    array = _resize_to_28(array)

    return array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)


def _crop_to_content(image: np.ndarray, margin: int = 2) -> np.ndarray:
    """Crop the image tightly around the non-background (digit) region.

    Args:
        image: A 2D float array with values in ``[0, 1]``.
        margin: Extra rows/cols to keep around the digit.

    Returns:
        A cropped 2D array, or the original image if no content is found.
    """
    mask = image > 0.05
    if not mask.any():
        return image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(rmin - margin, 0)
    rmax = min(rmax + margin, image.shape[0] - 1)
    cmin = max(cmin - margin, 0)
    cmax = min(cmax + margin, image.shape[1] - 1)
    return image[rmin : rmax + 1, cmin : cmax + 1]


def _resize_to_28(image: np.ndarray) -> np.ndarray:
    """Resize a 2D image to 28x28 preserving aspect ratio and centring.

    Args:
        image: 2D float array in ``[0, 1]``.

    Returns:
        A 28x28 float array.
    """
    if CV2_AVAILABLE:
        resized = cv2.resize(
            image, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA
        )
    else:
        resized = _resize_naive(image, (IMG_WIDTH, IMG_HEIGHT))

    # Guard against uint8-style inputs (values 0-255) coming back from cv2.
    if resized.max() > 1.5:
        resized = resized / 255.0
    return resized.astype("float32")


def _resize_naive(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize used only when OpenCV is unavailable."""
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = size
    rows = (np.arange(dst_h) * src_h // dst_h).astype(int)
    cols = (np.arange(dst_w) * src_w // dst_w).astype(int)
    return image[rows][:, cols]


def predict_custom_image(model: keras.Model, image_bytes: bytes | np.ndarray) -> dict:
    """Predict the digit on a custom handwritten image.

    Args:
        model: Trained Keras model.
        image_bytes: Raw image bytes or decoded numpy array.

    Returns:
        Dictionary with ``digit`` and ``confidence`` keys.
    """
    processed = preprocess_custom_image(image_bytes)
    probabilities = model.predict(processed, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    return {"digit": digit, "confidence": confidence, "probabilities": probabilities}


def predict_mnist_sample(model: keras.Model, image: np.ndarray) -> dict:
    """Predict a single already-preprocessed MNIST image.

    Args:
        model: Trained Keras model.
        image: Single image shaped ``(28, 28, 1)`` or ``(28, 28)``.

    Returns:
        Dictionary with ``digit`` and ``confidence`` keys.
    """
    if image.ndim == 2:
        image = image[..., np.newaxis]
    if image.max() > 1.0:
        image = image / 255.0

    batch = image[np.newaxis, ...]
    probabilities = model.predict(batch, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit])
    return {"digit": digit, "confidence": confidence, "probabilities": probabilities}


def main(argv: list[str] | None = None) -> None:
    """Predict a custom image from the command line.

    Usage:
        python -m src.predict path/to/digit.png
    """
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) < 1:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)

    image_path = Path(args[0])
    model = load_model()
    image_bytes = image_path.read_bytes()

    result = predict_custom_image(model, image_bytes)
    print(f"Image            : {image_path}")
    print(f"Predicted digit  : {result['digit']}")
    print(f"Confidence       : {result['confidence'] * 100:.2f}%")
    print(f"Class scores     : {[f'{v:.3f}' for v in result['probabilities']]}")


if __name__ == "__main__":
    main()
