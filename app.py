import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the quantized model
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

# Define class names for rice leaf diseases
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

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
st.title('Rice Leaf Disease Detection with Quantized Model')

st.markdown("### Upload an image to make predictions")

# File uploader with an image preview
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Show a button to make predictions
    if st.button('Predict'):
        # Make prediction using the quantized model
        prediction = predict_image(image)
        
        # Find the predicted class index using argmax
        predicted_class_index = np.argmax(prediction)

        # Display the predicted class name
        predicted_class = class_names[predicted_class_index]
        
        # Display the results
        st.markdown(f"### Prediction: {predicted_class}")
        st.markdown(f"Prediction Probability: {prediction[0][predicted_class_index]:.4f}")

        # Display the prediction array (optional)
        st.write("Raw Prediction Output:")
        st.write(prediction)

        # Provide a message if no clear prediction
        if prediction[0][predicted_class_index] < 0.5:  # Adjust threshold if needed
            st.warning("The model is not confident in this prediction.")

