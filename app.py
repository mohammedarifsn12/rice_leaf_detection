import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

# Function to make predictions with the quantized model
def predict_image(image):
    # Preprocess the image (resize and normalize)
    image = image.resize((224, 224))  # Adjust size based on your model's input size
    image = np.array(image) / 255.0  # Normalize if necessary
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Prepare input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit UI
st.title('Quantized Model Deployment with Streamlit')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Process the image and make predictions
    image = Image.open(uploaded_file)
    prediction = predict_image(image)

    # Display the image and the prediction
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write(f"Prediction: {prediction}")

