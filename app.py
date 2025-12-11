import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- 1. Define Constants and Model Path (Adjusted for .keras format) ---
# IMPORTANT: These must match the settings used during training.
IMAGE_SIZE = 256
CHANNELS = 3
# The class names list MUST be in the correct order (matching the model's output indices)
class_names = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
MODEL_PATH = "potatoes.keras"  # USE THE NEW .keras FORMAT to fix the loading error!

# --- 2. Load the Model (Uses caching for fast loading) ---
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model from the file system."""
    try:
        # tf.keras.models.load_model handles the native .keras format correctly.
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        # Display an error if loading fails
        st.error(f"Error loading model from {MODEL_PATH}.")
        st.caption("Please ensure the file is named 'potatoes.keras' and is in the same folder.")
        st.code(e)
        return None

model = load_model()

# --- 3. Prediction and Preprocessing Logic ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resizes, converts to array, and normalizes the image for the model."""
    
    # 1. Resize the image
    img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # 2. Convert to NumPy array
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    
    # 3. Normalize pixel values (0-1) - MUST match rescale=1./255 in training
    img_array_normalized = img_array / 255.0
    
    # 4. Add a batch dimension: (1, 256, 256, 3)
    img_array_batch = np.expand_dims(img_array_normalized, axis=0)
    
    return img_array_batch

def predict_class(model, processed_image: np.ndarray):
    """Runs prediction on the processed image."""
    
    predictions = model.predict(processed_image, verbose=0)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions[0])
    
    predicted_class = class_names[predicted_index]
    confidence = round(100 * (np.max(predictions[0])), 2)
    
    return predicted_class, confidence

# --- 4. Streamlit App Interface ---

st.set_page_config(
    page_title="Potato Disease Classifier",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🥔 Potato Leaf Disease Classifier")
st.markdown("Upload a potato leaf image (JPG/PNG) to get an instant diagnosis.")

# Check for model loading success and stop if it failed
if model is None:
    st.error("Application is stopped due to model loading failure.")
    st.stop()

# --- Image Uploader ---
uploaded_file = st.file_uploader(
    "Choose a potato leaf image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read the image file
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
   # Update the line to this:
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("") 

    # --- Prediction Button ---
    if st.button('Classify Leaf'):
        with st.spinner('Analyzing the image...'):
            # Preprocess and Predict
            processed_img = preprocess_image(image)
            predicted_class, confidence = predict_class(model, processed_img)
            
            # --- Result Formatting ---
            if 'healthy' in predicted_class.lower():
                color = 'green'
                emoji = '✅'
            elif 'blight' in predicted_class.lower():
                color = 'red'
                emoji = '🚨'
            else:
                color = 'blue'
                emoji = '❓'

            # Display the result
            st.markdown("---")
            st.subheader(f"Diagnosis Result: {emoji}")
            st.markdown(
                f"**Predicted Class:** <span style='color:{color}; font-size:24px;'>{predicted_class.replace('_', ' ')}</span>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Confidence:** <span style='color:{color}; font-size:24px;'>{confidence}%</span>",
                unsafe_allow_html=True
            )

st.markdown("---")
st.caption("Model trained using a Convolutional Neural Network (CNN).")