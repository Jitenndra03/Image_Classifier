import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("models/imageclassifier.h5")

model = load_trained_model()

# Set page config
st.set_page_config(page_title="Mood Classifier", page_icon="😊", layout="centered")

# Apply custom CSS
with open("assets/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# UI
st.title("😊 Mood Classifier")
st.subheader("Is the face happy or sad? Upload an image to find out.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        # Preprocess
        img_resized = img.resize((48, 48))  # Modify to your input shape
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        label = "Happy 😊" if prediction[0][0] > 0.5 else "Sad 😢"
        st.success(f"Prediction: **{label}**")
