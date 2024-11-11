import streamlit as st
import json
import numpy as np
import tensorflow as tf
import requests
import io
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model from GitHub
model_url = 'https://raw.githubusercontent.com/Jethro-Malabar/Trial/main/bantayBakhaw_modelV3.h5'
model_response = requests.get(model_url)
loaded_model = tf.keras.models.load_model(io.BytesIO(model_response.content))

# Load class indices from JSON file on GitHub
json_url = 'https://raw.githubusercontent.com/Jethro-Malabar/Trial/main/class_indices.json'
json_response = requests.get(json_url)
class_indices = json.loads(json_response.content)

class_names = list(class_indices.keys())  # Get the class names in order

# Suggested treatments for each disease
treatment_suggestions = {
    "Anthracnose": "Apply a fungicide spray and ensure good air circulation around the plant.",
    "Brown_Spot": "Use a copper-based fungicide and avoid overhead watering.",
    "Leaf_Blight": "Prune affected leaves and apply a broad-spectrum fungicide.",
    "Black_Spot": "Remove infected leaves and use a sulfur-based fungicide.",
    "White_Spot": "Treat with a systemic fungicide and maintain good soil drainage.",
    "Healthy": "No treatment necessary. Keep the plant healthy with proper care."
}

def predict_image(image):
    img_array = preprocess_image(image)
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class]
    confidence = np.max(predictions) * 100
    return predicted_class_name, confidence

def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Streamlit app layout
st.set_page_config(page_title="Bakhaw Disease Classifier", layout="centered")
st.title("🌿 Bakhaw Disease Classifier")
st.write("Upload an image of a mangrove leaf to classify any potential diseases. Our model will analyze and provide suggested treatments if necessary.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    predicted_class_name, confidence = predict_image(image)
    
    st.subheader("Prediction")
    st.write(f"**Disease Detected:** {predicted_class_name}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")
    
    if st.button("Show Suggested Treatment"):
        st.write(f"**Suggested Treatment for {predicted_class_name}:**")
        st.write(treatment_suggestions[predicted_class_name])
