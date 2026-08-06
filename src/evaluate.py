"""Model evaluation: metrics, confusion matrix and visualisations.

Loads the best saved model, evaluates it on the held-out test set and
produces the deliverable artifacts: test metrics, classification report,
confusion matrix and sample correct/incorrect prediction grids.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from src.utils import CLASS_NAMES, IMAGES_DIR, MODEL_KERAS, ensure_dirs


def load_model(path=MODEL_KERAS) -> keras.Model:
    """Load a trained Keras model from disk.

    Args:
        path: Path to the ``.keras`` model file.

    Returns:
        The loaded Keras model.
    """
    ensure_dirs()
    model = keras.models.load_model(path)
    print(f"Model loaded from: {path}")
    return model


def evaluate_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Evaluate the model on the test set and return key metrics.

    Args:
        model: Trained Keras model.
        x_test: Normalised test images.
        y_test: One-hot test labels.

    Returns:
        Dictionary with ``loss`` and ``accuracy``.
    """
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    metrics = {"loss": float(loss), "accuracy": float(accuracy)}
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Test Loss     : {metrics['loss']:.4f}")
    print(f"Test Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print("=" * 60)
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path=IMAGES_DIR / "confusion_matrix.png",
) -> plt.Figure:
    """Plot and save a confusion matrix.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        save_path: Where to save the figure.

    Returns:
        The matplotlib ``Figure`` object.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    save_path = ensure_parent(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved to: {save_path}")
    return fig


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print the scikit-learn classification report to the console."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, digits=4
    )
    print(report)


def show_sample_predictions(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_samples: int = 10,
    save_path=IMAGES_DIR / "sample_predictions.png",
) -> plt.Figure:
    """Plot sample images with their predicted vs true labels.

    Args:
        model: Trained Keras model.
        x_test: Normalised test images.
        y_test: One-hot test labels.
        num_samples: Number of samples to show.
        save_path: Where to save the figure.

    Returns:
        The matplotlib ``Figure`` object.
    """
    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    y_pred = np.argmax(model.predict(x_test[:num_samples], verbose=0), axis=1)

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 4))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(x_test[i].squeeze(), cmap="gray")
        color = "green" if y_pred[i] == y_true[i] else "red"
        axes[i].set_title(f"True: {y_true[i]}\nPred: {y_pred[i]}", fontsize=10, color=color)
        axes[i].axis("off")

    fig.suptitle("Sample Predictions (green = correct, red = wrong)", fontsize=12)
    fig.tight_layout()

    save_path = ensure_parent(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Sample predictions saved to: {save_path}")
    return fig


def show_correct_incorrect(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_correct: int = 5,
    n_incorrect: int = 5,
    save_path=IMAGES_DIR / "correct_incorrect.png",
) -> plt.Figure:
    """Show a grid of correct and incorrect predictions side by side.

    Args:
        model: Trained Keras model.
        x_test: Normalised test images.
        y_test: One-hot test labels.
        n_correct: Number of correct samples to display.
        n_incorrect: Number of incorrect samples to display.
        save_path: Where to save the figure.

    Returns:
        The matplotlib ``Figure`` object.
    """
    y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)

    correct_idx = np.where(y_pred == y_true)[0][:n_correct]
    incorrect_idx = np.where(y_pred != y_true)[0][:n_incorrect]

    fig, axes = plt.subplots(2, max(n_correct, n_incorrect), figsize=(14, 5))
    if axes.ndim == 1:
        axes = axes.reshape(2, -1)

    row_titles = ["Correct Predictions", "Incorrect Predictions"]
    for row, idx_list in enumerate([correct_idx, incorrect_idx]):
        for col, idx in enumerate(idx_list):
            axes[row][col].imshow(x_test[idx].squeeze(), cmap="gray")
            axes[row][col].set_title(
                f"True: {y_true[idx]}\nPred: {y_pred[idx]} ({confidence[idx]:.2f})",
                fontsize=9,
            )
            axes[row][col].axis("off")
        for col in range(len(idx_list), max(n_correct, n_incorrect)):
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(row_titles[row], fontsize=11, rotation=90, labelpad=30)

    fig.tight_layout()
    save_path = ensure_parent(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Correct/incorrect grid saved to: {save_path}")
    return fig


def ensure_parent(path) -> Path:
    """Create the parent directory of *path* and return the path."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    """Evaluate the saved model end-to-end from the command line.

    Usage:
        python -m src.evaluate
    """
    import numpy as np

    from src.preprocess import load_data, preprocess_data
    from src.utils import set_seed

    set_seed()
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )

    model = load_model()
    metrics = evaluate_model(model, x_test, y_test)

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=1), axis=1)

    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    show_sample_predictions(model, x_test, y_test, num_samples=10)
    show_correct_incorrect(model, x_test, y_test)

    print(f"\nFinal test accuracy: {metrics['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
