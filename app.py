import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle
import cv2
from scipy.ndimage import center_of_mass

# Load the trained model from the pickle file
with open('digit_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Function to preprocess uploaded image
def preprocess_image(image):
    img = Image.open(image).convert('L')
    img_np = np.array(img)

    # Invert if background is dark (optional)
    if np.mean(img_np) < 127:
        img_np = 255 - img_np

    # Adaptive thresholding for binarization
    img_bin = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 15, 8)

    # Dilate to make the digit thicker
    kernel = np.ones((5, 5), np.uint8)
    img_bin = cv2.dilate(img_bin, kernel, iterations=2)

    # Find contours and crop to the largest bounding box
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img_bin = img_bin[y:y+h, x:x+w]

    # Pad to square
    h, w = img_bin.shape
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    img_bin = np.pad(img_bin, ((pad_h, size - h - pad_h), (pad_w, size - w - pad_w)), 'constant', constant_values=0)

    # Resize to 28x28
    img_pil = Image.fromarray(img_bin)
    img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    img_np = np.array(img_pil)

    # Center the digit by center of mass
    cy, cx = center_of_mass(img_np > 0)
    shiftx = np.round(14 - cx).astype(int)
    shifty = np.round(14 - cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_np = cv2.warpAffine(img_np, M, (28, 28), borderValue=0)

    # Optional: Show processed image in Streamlit for debugging
    st.image(img_np, caption='Processed Image', width=100)

    img_array = img_np.reshape(1, -1) / 255.0
    return img_array

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (28x28 pixels) to predict the digit.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array = preprocess_image(uploaded_file)
    prediction = clf.predict(img_array)
    st.write(f"Predicted digit: {prediction[0]}") 