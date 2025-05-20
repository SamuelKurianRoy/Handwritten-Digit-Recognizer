import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import pickle
from scipy.ndimage import center_of_mass

# Load the trained model from the pickle file
with open('digit_model.pkl', 'rb') as f:
    clf = pickle.load(f)

def preprocess_image(image):
    # Load and convert to grayscale
    img = Image.open(image).convert('L')
    
    # Invert if background is dark
    img_np = np.array(img)
    if np.mean(img_np) < 127:
        img_np = 255 - img_np
    img = Image.fromarray(img_np)
    
    # Binarize the image
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    
    # Find the bounding box of non-zero pixels
    img_np = np.array(img)
    rows = np.any(img_np, axis=1)
    cols = np.any(img_np, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Crop the image
    img = img.crop((x_min, y_min, x_max + 1, y_max + 1))
    
    # Add padding to make it square
    size = max(img.size)
    new_img = Image.new('L', (size, size), 0)
    new_img.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    
    # Resize to 28x28
    new_img = new_img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(new_img).reshape(1, -1) / 255.0
    
    # Show the processed image
    st.image(new_img, caption='Processed Image', width=100)
    
    return img_array

# Streamlit UI
st.title("Handwritten Digit Recognition (0 vs 1)")
st.write("Upload an image of a handwritten digit (0 or 1) to predict.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array = preprocess_image(uploaded_file)
    prediction = clf.predict(img_array)
    st.write(f"Predicted digit: {prediction[0]}") 