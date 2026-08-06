"""Interactive web app for Handwritten Digit Recognition.

Run with:
    streamlit run app.py

Upload a handwritten digit image (or inspect the model), and the app shows
the predicted digit and its confidence score.
"""

from __future__ import annotations

import streamlit as st

import src.predict as predict
from src.evaluate import load_model as load_eval_model
from src.preprocess import load_data, preprocess_images, show_sample_images

st.set_page_config(page_title="Digit Recognizer", page_icon="0-9", layout="centered")


@st.cache_resource
def get_model():
    """Load the trained model once and cache it across reruns."""
    return load_eval_model()


def main() -> None:
    st.title("Handwritten Digit Recognition")
    st.caption("CNN trained on the MNIST dataset with TensorFlow/Keras")

    tab_predict, tab_info, tab_dataset = st.tabs(
        ["Predict", "Model Info", "Dataset Samples"]
    )

    with tab_predict:
        st.subheader("Upload a handwritten digit image")

        uploaded_file = st.file_uploader(
            "Choose a PNG/JPG image of a single handwritten digit",
            type=["png", "jpg", "jpeg"],
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            model = get_model()

            with st.spinner("Predicting..."):
                result = predict.predict_custom_image(model, image_bytes)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image_bytes, caption="Uploaded image", use_container_width=True)
            with col2:
                st.subheader("Result")
                st.metric("Predicted Digit", result["digit"])
                st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")

            st.progress(float(result["confidence"]))
            st.caption(
                "The image was converted to a 28x28 grayscale MNIST-style "
                "input before prediction."
            )

    with tab_info:
        st.subheader("Model Architecture")
        model = get_model()
        model.summary(print_fn=st.text)
        st.info(
            "Trained for up to 15 epochs with early stopping, batch size 32 "
            "and a 20% validation split. Target test accuracy >= 98%."
        )

    with tab_dataset:
        st.subheader("Sample MNIST Images")
        if st.button("Show 25 sample images"):
            (x_train, y_train), _ = load_data()
            x_train = preprocess_images(x_train)
            fig = show_sample_images(x_train, y_train, num_samples=25)
            st.pyplot(fig)
        st.caption("MNIST: 60,000 training / 10,000 test 28x28 grayscale digits.")


if __name__ == "__main__":
    main()
