import streamlit as st
import numpy as np
from PIL import Image

st.title("🐣 Egg Color Classifier (Simple Version)")
st.write("Upload an egg image. The app will guess whether it is Black, Brown, Violet, or White based on its average color.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Reference RGB values for each shell color (you can tweak these)
reference_colors = {
    "Black":  np.array([40, 40, 40], dtype=np.float32),
    "Brown":  np.array([140, 100, 70], dtype=np.float32),
    "Violet": np.array([120, 80, 150], dtype=np.float32),
    "White":  np.array([230, 230, 230], dtype=np.float32),
}

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize to reduce noise and speed up computation
    img_small = img.resize((150, 150))
    img_arr = np.array(img_small, dtype=np.float32)

    # Compute average color
    avg_color = img_arr.mean(axis=(0, 1))  # shape (3,)
    st.write(f"Average RGB: {avg_color.astype(int)}")

    # Compute distance between avg_color and each reference color
    best_label = None
    best_distance = float("inf")

    for label, ref in reference_colors.items():
        dist = np.linalg.norm(avg_color - ref)
        if dist < best_distance:
            best_distance = dist
            best_label = label

    st.success(f"🎨 Detected Color: **{best_label}**")
    st.write(f"(Lower distance = better match, distance = {best_distance:.2f})")
