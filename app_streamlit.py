    # src/app_streamlit.py
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

st.title("MNIST Digit Recognizer (Demo)")

model = tf.keras.models.load_model("best_mnist_keras.h5")
 # path from Keras training

uploaded_file = st.file_uploader("Upload a handwritten digit (PNG/JPG)", type=['png','jpg','jpeg'])
if uploaded_file:
    img = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(img, caption='Uploaded', width=150)
    # Preprocess: resize to 28x28 and invert if needed
    img = ImageOps.invert(img)  # invert white-on-black issues; test with a few inputs
    img = img.resize((28,28))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1,28,28,1)
    pred = model.predict(arr)
    label = np.argmax(pred, axis=1)[0]
    st.write(f"Predicted digit: **{label}** (confidence {pred.max():.3f})")
