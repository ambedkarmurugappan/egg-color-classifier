import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# LOAD TFLITE MODEL
# ------------------------------

interpreter = tf.lite.Interpreter(model_path="egg_color_model.tflite")
interpreter.allocate_tensors()

# Get input & output details of the TFLite model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names (same order as your training)
class_names = ["Black", "Brown", "Violet", "White"]

# ------------------------------
# STREAMLIT UI
# ------------------------------

st.title("🐣 Egg Color Classification (TFLite)")
st.write("Upload an egg image and the model will classify its color.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize for model
    img = img.resize((150, 150))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run TFLite model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get best class
    index = int(np.argmax(prediction))
    color = class_names[index]
    confidence = float(prediction[index] * 100)

    # Show result
    st.success(f"🎨 Detected Color: **{color}**")
    st.write(f"Confidence: {confidence:.2f}%")
