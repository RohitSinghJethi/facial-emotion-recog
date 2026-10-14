
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# ---------------- Page configuration ----------------
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="hiiii",
    layout="wide"
)

st.title("hiiii Facial Expression Recognition")
st.write(
    "Upload a facial image to predict the person's "
    "facial expression using a CNN model."
)

# ---------------- Load model and class names ----------------
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("best_fer_model.keras")

    with open("class_names.json", "r") as f:
        classes = json.load(f)

    return model, classes

try:
    model, class_names = load_model_and_classes()
except Exception as e:
    st.error(f"Unable to load model or class names: {e}")
    st.info(
        "Place best_fer_model.keras and class_names.json "
        "in the same folder as app.py."
    )
    st.stop()

# ---------------- Upload image ----------------
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    if st.button("Recognize Expression", type="primary"):

        with st.spinner("Predicting expression..."):

            # Match the model's training preprocessing
            image = image.convert("L")
            image = image.resize((48, 48))

            img_array = np.array(image, dtype=np.float32)
            img_array = img_array / 255.0

            # Shape: (1, 48, 48, 1)
            img_array = np.expand_dims(img_array, axis=(0, -1))

            # Model prediction
            predictions = model.predict(img_array, verbose=0)[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index]) * 100

        with col2:
            st.subheader("Prediction")
            st.success(f"Expression: {predicted_class.title()}")
            st.metric("Model confidence", f"{confidence:.2f}%")

        st.divider()
        st.subheader("Probabilities for All Expressions")

        # Display in descending order
        sorted_indices = np.argsort(predictions)[::-1]

        for i in sorted_indices:
            label = class_names[i].title()
            probability = float(predictions[i])

            st.write(f"**{label}** — {probability * 100:.2f}%")
            st.progress(probability)

        st.caption(
            "Predictions are estimates from the model and "
            "may be inaccurate, especially for unclear images."
        )

