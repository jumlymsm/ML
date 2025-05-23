import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your saved model
model = tf.keras.models.load_model("apple_leaf_model.h5")

# Class labels
class_names = ['Apple_Black_rot', 'Apple_Cedar_Rust', 'Apple_Healthy', 'Apple_Scab']

st.title("🍏 Apple Leaf Disease Detector")
st.write("Upload an apple leaf image to predict its disease class.")

uploaded_file = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")

    from PIL import Image
import numpy as np

# Assuming 'uploaded_file' is your uploaded image file
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((128, 128))              # Change to match your model input shape
    img = img.convert("RGB")                  # Ensure it's in RGB
    img_array = np.array(img) / 255.0         # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 128, 128, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    st.write("Predicted class:", class_names[predicted_class[0]])




