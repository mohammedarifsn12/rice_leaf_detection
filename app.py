import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set Page Configurations
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌱",
    layout="centered",  # or "wide"
    initial_sidebar_state="expanded",
)

# TensorFlow Lite Model Prediction
def model_prediction(test_image):
    # Load the quantized model (TensorFlow Lite)
    interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
    interpreter.allocate_tensors()

    # Preprocess the image
    image = Image.open(test_image)
    image = image.resize((224, 224))  # Resize image to match the model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Prepare input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image_array.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)  # Return index of the max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Home Page with Description
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    """)

# Prediction Page - Disease Recognition
elif app_mode == "Disease Recognition":
    st.header("🌸 **Disease Recognition** 🌸")

    # Display a message that only supports image files
    st.markdown(
        """
        **Please upload a valid image file** (JPEG, PNG, or JPG) of a plant leaf to check for potential diseases.  
        The system will analyze the image and predict the plant disease. 🌿🔬
        """
    )
    
    # File uploader to upload images
    test_image = st.file_uploader(
        "Choose an Image:", 
        type=["jpeg", "png", "jpg"],  # Only accept image files
        label_visibility="visible"
    )
    
    # Display image after upload
    if test_image is not None:
        st.image(test_image, width=400, use_container_width=True)
    
    # Predict button
    if st.button("🔍 **Predict Disease**"):
        if test_image:
            st.snow()  # Adds a snow effect while prediction is running
            st.write("**Our Prediction**:")

            # Get the predicted class index from the model
            result_index = model_prediction(test_image)
            
            # List of class names for plant diseases (update this list based on your actual classes)
            class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']  # Update with your actual class names

            # Display the prediction
            st.success(f"🌟 **Prediction:** This plant shows signs of **{class_names[result_index]}**!")
        else:
            st.error("⚠️ Please upload an image file before clicking 'Predict'.")



