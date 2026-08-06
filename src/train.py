"""CNN model definition, compilation and training.

Builds the convolutional neural network described in the project
specification and trains it with EarlyStopping, ModelCheckpoint and
TensorBoard callbacks. The best weights are persisted both as
``model.keras`` and ``model.h5``.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import (
    BATCH_SIZE,
    EPOCHS,
    IMG_SHAPE,
    LEARNING_RATE,
    LOGS_DIR,
    MODEL_H5,
    MODEL_KERAS,
    MODELS_DIR,
    NUM_CLASSES,
    VALIDATION_SPLIT,
    ensure_dirs,
)


def build_model(input_shape: tuple[int, int, int] = IMG_SHAPE) -> keras.Model:
    """Construct the CNN architecture.

    Architecture (as per specification):
        Conv2D(32) -> MaxPool2D -> Conv2D(64) -> MaxPool2D
        -> Flatten -> Dense(128) -> Dropout(0.5) -> Dense(10, softmax)

    Args:
        input_shape: Shape of a single input image, ``(28, 28, 1)``.

    Returns:
        A compiled Keras model.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _build_callbacks(
    model_keras_path=MODEL_KERAS,
    model_h5_path=MODEL_H5,
    logs_dir=LOGS_DIR,
):
    """Create EarlyStopping, ModelCheckpoint and TensorBoard callbacks."""
    checkpoint_keras = keras.callbacks.ModelCheckpoint(
        filepath=str(model_keras_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    checkpoint_h5 = keras.callbacks.ModelCheckpoint(
        filepath=str(model_h5_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=0,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=str(logs_dir / f"fit_{int(time.time())}"),
        histogram_freq=1,
    )
    return [checkpoint_keras, checkpoint_h5, early_stopping, tensorboard]


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> keras.callbacks.History:
    """Train the CNN on preprocessed training data.

    Args:
        x_train: Normalised training images ``(N, 28, 28, 1)``.
        y_train: One-hot training labels ``(N, 10)``.
        x_val: Normalised validation images.
        y_val: One-hot validation labels.
        epochs: Maximum number of epochs to train.
        batch_size: Mini-batch size.

    Returns:
        The Keras ``History`` object with per-epoch metrics.
    """
    ensure_dirs()
    model = build_model()
    model.summary()

    callbacks = _build_callbacks()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # The callbacks already saved the best weights; re-load to be safe and
    # export an equivalent .h5 copy.
    best_model = keras.models.load_model(MODEL_KERAS)
    best_model.save(MODEL_H5)
    print(f"\nBest model saved to:\n  {MODEL_KERAS}\n  {MODEL_H5}")
    return history


def main() -> None:
    """Train the CNN end-to-end from the command line.

    Usage:
        python -m src.train
    """
    from sklearn.model_selection import train_test_split

    from src.preprocess import display_dataset_info, load_data, preprocess_data
    from src.utils import IMAGES_DIR, set_seed

    set_seed()
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train, y_train, x_test, y_test = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    display_dataset_info(x_train, y_train, x_test, y_test)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"Train/Val split -> train: {len(x_train)}, val: {len(x_val)}")

    history = train_model(x_train, y_train, x_val, y_val)
    plot_training_history(history, save_path=IMAGES_DIR)


def plot_training_history(history: keras.callbacks.History, save_path=None) -> plt.Figure:
    """Plot and save accuracy and loss curves over epochs.

    Args:
        history: Keras ``History`` object from ``train_model``.
        save_path: Optional directory path where the figures are written.

    Returns:
        The matplotlib ``Figure`` object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    ax1.plot(epochs_range, acc, "b-", label="Training Accuracy")
    ax1.plot(epochs_range, val_acc, "r-", label="Validation Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, loss, "b-", label="Training Loss")
    ax2.plot(epochs_range, val_loss, "r-", label="Validation Loss")
    ax2.set_title("Training and Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()

    if save_path:
        from pathlib import Path

        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / "accuracy_loss_graph.png", dpi=150, bbox_inches="tight")
        print(f"Training history graph saved to: {path / 'accuracy_loss_graph.png'}")

    return fig


if __name__ == "__main__":
    main()
