# src/app_draw_digit.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import cv2

# Load model
model = tf.keras.models.load_model("best_mnist_keras.h5")  # path from Keras training

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognizer")

st.write("Draw a digit (0–9) in the box below and click **Predict**.")

# Create the canvas
canvas_result = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Convert canvas image to grayscale
        img = canvas_result.image_data[:, :, :3].astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Invert to make background black and digit white (MNIST style)
        img = cv2.bitwise_not(img)

        # Threshold (binarize)
        _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

        # Find bounding box of the digit and crop
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]

        # Resize to 20x20 and pad to 28x28 (center the digit)
        img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        padded = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)

        # Normalize and reshape
        arr = padded.astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        # Predict
        preds = model.predict(arr)
        digit = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))

        st.markdown("---")
        st.markdown(f"### 🧠 Predicted Digit: **{digit}**")
        st.write(f"Confidence: **{confidence:.3f}**")

        # Optional: Show preprocessed image
        st.image(padded, width=100, caption="Preprocessed (28x28)")

        # Optional: Plot probabilities
        st.bar_chart(preds[0])
    else:
        st.warning("Please draw a digit first!")
