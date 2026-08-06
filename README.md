# Handwritten Digit Recognition using Supervised Learning

A production-quality **Convolutional Neural Network (CNN)** that recognises
handwritten digits (0-9) with **~99% test accuracy**, trained on the classic
**MNIST** dataset using **TensorFlow / Keras**.

Built as a clean, modular, well-documented Python project — ideal for a
final-year college project and GitHub portfolio.

---

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Accuracy](#results--accuracy)
- [Custom Image Prediction](#custom-image-prediction)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Features

- **Automatic dataset download** — MNIST is fetched and cached in `dataset/`
  on first run, no manual setup required.
- **Data preprocessing** — pixel normalisation (0-255 to 0-1), reshaping to
  `(28, 28, 1)` and one-hot label encoding.
- **CNN architecture** — Conv2D, MaxPooling, Dropout and Softmax layers tuned
  for the digit recognition task.
- **Smart training** — EarlyStopping, ModelCheckpoint (best weights saved) and
  TensorBoard logging, with a fixed random seed for reproducibility.
- **Comprehensive evaluation** — test accuracy, loss, confusion matrix,
  classification report, accuracy/loss curves and correct vs incorrect
  prediction grids.
- **Two saved formats** — `model.keras` and `model.h5`.
- **Custom image prediction** — upload any handwritten digit image (PNG/JPG)
  via CLI or the Streamlit web app and get the predicted digit + confidence.
- **Interactive web app** — a clean Streamlit interface.

---

## Demo

| Prediction flow |
| --- |
| Upload a handwritten digit image, the CNN returns the digit and a confidence score. |

Example:

```
$ python -m src.predict images/custom_digit.png

Image            : images/custom_digit.png
Predicted digit  : 7
Confidence       : 99.87%
```

---

## Technologies

| Area | Tool |
| --- | --- |
| Language | Python 3.11+ |
| Deep learning | TensorFlow / Keras |
| Numerical computing | NumPy, Pandas |
| Data visualisation | Matplotlib, Seaborn |
| ML metrics | Scikit-learn |
| Image processing | OpenCV (with pure-Matplotlib fallback) |
| Web app | Streamlit |
| Notebooks | Jupyter |

---

## Project Structure

```
handwritten-digit-recognition/
│
├── dataset/            # MNIST archive (auto-downloaded)
├── models/             # Trained model.keras + model.h5
├── images/             # Generated plots + sample custom digits
├── notebooks/          # Jupyter training notebook
├── src/
│   ├── __init__.py
│   ├── utils.py        # Config, paths, seeds
│   ├── preprocess.py   # Data loading & preprocessing
│   ├── train.py        # Model definition & training
│   ├── evaluate.py     # Metrics, confusion matrix, plots
│   └── predict.py      # Prediction on test/custom images
│
├── app.py              # Streamlit web application
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Installation

Requires **Python 3.11+**.

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2. Create and activate a virtual environment (recommended)

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> TensorFlow 2.13+ is required. On Windows ensure the Microsoft Visual C++
> Redistributable is installed (it usually is by default).

---

## Usage

### Train the model

```bash
python -m src.train
```

- Downloads MNIST automatically into `dataset/`.
- Trains up to 15 epochs (early stopping on validation accuracy).
- Saves the best model as `models/model.keras` and `models/model.h5`.
- Saves `images/accuracy_loss_graph.png`.

### Evaluate the model

```bash
python -m src.evaluate
```

Prints test loss/accuracy and the classification report, and saves:

- `images/confusion_matrix.png`
- `images/sample_predictions.png`
- `images/correct_incorrect.png`

### Predict a custom image

```bash
python -m src.predict path/to/your/digit.png
```

### Launch the web app

```bash
streamlit run app.py
```

Upload a handwritten digit image (PNG/JPG) and view the predicted digit with
its confidence score.

### View TensorBoard logs

```bash
tensorboard --logdir logs
```

### Open the Jupyter notebook

```bash
jupyter notebook notebooks/handwritten_digit_recognition.ipynb
```

---

## Results & Accuracy

Trained on 48,000 images (80% of 60,000; 20% held out for validation) and
evaluated on the official 10,000-image test set:

| Metric | Value |
| --- | --- |
| Test Accuracy | **~99.0 %** |
| Test Loss | ~0.03 |
| Best Validation Accuracy | ~99.2 % |

The confusion matrix confirms errors are rare and mostly between visually
similar digits (e.g. 4 vs 9, 7 vs 2). All generated plots live in `images/`.

> Note: exact numbers vary slightly between runs due to floating-point
> non-determinism on different hardware, even with a fixed seed.

---

## Custom Image Prediction

Custom images are automatically converted to MNIST format before prediction:

1. Decoded and converted to grayscale.
2. Resized to 28 x 28 pixels.
3. Inverted if necessary so the digit is white on a black background.
4. Pixel values scaled to [0, 1].

The model then outputs a probability distribution over the ten digits; the
argmax is the predicted digit and its value is the confidence score.

---

## Screenshots

> Placeholder — add your own screenshots here:

- `images/sample_images.png` — sample MNIST digits
- `images/accuracy_loss_graph.png` — training curves
- `images/confusion_matrix.png` — confusion matrix
- `images/sample_predictions.png` — sample predictions
- `images/correct_incorrect.png` — correct vs incorrect predictions
- Web app: `streamlit run app.py`

---

## Future Improvements

- **Data augmentation** (shift / rotate / zoom) for extra robustness.
- **Deeper architectures** (ResNet-style blocks, Batch Normalisation) and
  hyper-parameter tuning.
- **Extended MNIST (EMNIST)** with letters and digits.
- **Model serving** via a REST API (FastAPI) or ONNX export for mobile.
- **Explainability** with Grad-CAM heatmaps.
- **Containerised deployment** with Docker.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
