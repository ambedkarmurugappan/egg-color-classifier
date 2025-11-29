import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="egg_color_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Black", "Brown", "Violet", "White"]

st.title("🐣 Egg Color Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Resize & preprocess
    img = img.resize((150, 150))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    # Best prediction
    idx = int(np.argmax(prediction))
    color = class_names[idx]
    conf = float(prediction[idx]) * 100

    st.success(f"🎨 Predicted Color: **{color}**")
    st.write(f"Confidence: {conf:.2f}%")
