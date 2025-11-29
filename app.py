import streamlit as st
import numpy as np
from PIL import Image

st.title("🐣 Egg Color Classifier (Simple Version)")
st.write("Upload an image and the app will classify whether the egg is Black, Brown, Violet, or White.")

# Reference average colors for each egg color
reference_colors = {
    "Black": np.array([40, 40, 40], dtype=np.float32),
    "Brown": np.array([150, 100, 70], dtype=np.float32),
    "Violet": np.array([130, 80, 150], dtype=np.float32),
    "White": np.array([230, 230, 230], dtype=np.float32),
}

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_small = img.resize((150, 150))
    img_arr = np.array(img_small, dtype=np.float32)

    avg_color = img_arr.mean(axis=(0, 1))
    st.write(f"Average RGB Color: {avg_color.astype(int)}")

    best_label = None
    best_dist = float("inf")

    for label, ref_color in reference_colors.items():
        dist = np.linalg.norm(avg_color - ref_color)
        if dist < best_dist:
            best_dist = dist
            best_label = label

    st.success(f"🎨 Predicted Egg Color: **{best_label}**")
    st.write(f"(Confidence score = lower distance → {best_dist:.2f})")
