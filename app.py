import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import time
import os

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("models/imageclassifier.h5")

model = load_trained_model()

# Set page config
st.set_page_config(page_title="Mood Classifier", page_icon="😊", layout="centered")

# Apply custom CSS (For 3D effects, animations, etc.)
custom_css = """
    <style>
    /* Background Gradient */
    body {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        color: white;
    }

    /* Header Style */
    h1 {
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 12px #000000;
        color: #FFFFFF;
        text-align: center;
    }

    /* Subheader Styling */
    .stMarkdown>h2 {
        color: #f1f1f1;
        font-size: 20px;
    }

    /* File Upload Box */
    .stFileUploader {
        color: #f0f0f0;
    }

    /* Predict Button Dynamic Effect */
    button:focus {
        outline: none;
    }
    .stButton>button {
        background-color: #123456;
        border: none;
        border-radius: 20px;
        color: #f1f1f1;
        padding: 0.8em 1.5em;
        cursor: pointer;
        font-size: 18px;
        box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.4s ease;
    }
    .stButton>button:hover {
        background-color: #f1f1f1;
        color: #123456;
        transform: scale(1.05);
        box-shadow: 5px 5px 12px rgba(0, 0, 0, 0.4);
    }

    /* Uploaded Image Box */
    img {
        border: 5px solid #6e8efb;
        border-radius: 15px;
        box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.8);
        transition: all 0.4s ease-out;
    }
    img:hover {
        transform: scale(1.02);
        box-shadow: 10px 10px 25px rgba(0, 0, 0, 1);
    }

    /* Footer (Custom Animation) */
    footer {
        text-align: center;
        margin-top: 50px;
        font-size: 12px;
        color: rgba(240,240,240,0.8);
    }

    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Title with gradient emoji
st.title("😊 **Mood Classifier** 😊")
st.subheader("Upload a face image to see if it's Happy 😊 or Sad 😢!")

# File Uploader
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Predict Button with Loading Animation
    if st.button("✨ Predict My Mood ✨"):
        with st.spinner("🌀 Analyzing the image... Please wait!"):
            time.sleep(2)  # Simulate a delay for loading effect

            # Preprocess Image
            img_resized = img.resize((256, 256))  # Match model input size
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            # Prediction
            prediction = model.predict(img_array)
            label = "Happy 😊" if prediction[0][0] > 0.5 else "Sad 😢"

            # Display Result
            st.success(f"🎉 **Mood Prediction: {label}** 🎉")
            st.balloons()

# Footer
st.markdown(
    """
    <footer>
        Mood Classifier App | Designed with ❤️
    </footer>
    """,
    unsafe_allow_html=True
)