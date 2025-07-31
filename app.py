import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_trained_model():
    return load_model("models/imageclassifier.h5")

model = load_trained_model()

# Set page config
st.set_page_config(page_title="Mood Classifier", page_icon="😊", layout="centered")

# Apply custom CSS (optional)
if os.path.exists("assets/styles.css"):
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
        try:
            # Preprocess image
            img_resized = img.resize((256, 256))  # Match model input
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            # Predict
            prediction = model.predict(img_array)
            label = "Happy 😊" if prediction[0][0] > 0.5 else "Sad 😢"

            # Show result
            st.success(f"Prediction: **{label}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
