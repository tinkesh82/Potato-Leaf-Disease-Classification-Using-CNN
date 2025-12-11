# Potato-Leaf-Disease-Classification-Using-CNN


## Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify potato leaf images into one of three categories: **Healthy**, **Early Blight**, or **Late Blight**. The solution is packaged as a user-friendly web application using Streamlit, allowing for instant, high-accuracy diagnosis.

The system is designed to provide rapid diagnostic support, improving efficiency and resource allocation in agricultural settings.

## ✨ Features

*   **Three-Class Classification:** Distinguishes between Healthy, Early Blight, and Late Blight.
*   **Deep Learning Model:** Uses a multi-layer CNN architecture trained for optimal feature extraction.
*   **Web Interface (Streamlit):** Simple, interactive UI for image upload and instant prediction.
*   **Robust Preprocessing:** Handles image resizing and normalization to match model requirements.
*   **Confidence Scoring:** Provides the prediction along with the model's confidence level.

## 💻 Technical Stack

*   **Language:** Python 3.x
*   **Frameworks:**
    *   **TensorFlow / Keras:** For model building, training, and prediction.
    *   **Streamlit:** For the web application interface.
*   **Libraries:** `Pillow`, `NumPy`.

## 📂 Project Structure

Your project directory should contain the following files:
potato_classifier_app/
├── app.py # The main Streamlit application code
├── potatoes.keras # The trained CNN model file (CRITICAL)
└── README.md # This file

Code
---

## 🚀 Installation and Setup

Follow these steps to set up and run the application locally on your machine.

### 1. Create a Virtual Environment (Recommended)

Using a virtual environment prevents conflicts with your system's other Python packages.

# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
2. Install Dependencies
With the virtual environment activated, install the required libraries:
code
Bash
pip install streamlit tensorflow pillow numpy
3. Obtain the Model File
Ensure you have your trained model file, potatoes.keras, placed directly inside the project directory alongside app.py.
Note: If you used the older HDF5 format, the file must be named potatoes.h5, and you may need to update the MODEL_PATH variable in app.py. The .keras format is highly recommended.

⚙️ Usage
Once all dependencies are installed and your virtual environment is active, run the application using the Streamlit CLI:
code
Bash
streamlit run app.py
Streamlit will automatically open the application in your default web browser at http://localhost:8501.

🧠 Model Details
Specification	Value
Architecture	Convolutional Neural Network (CNN)
Input Shape	(256, 256, 3) (RGB)
Output Classes	3 (Potato__Early_blight, Potato__Late_blight, Potato__healthy)
Key Layers	Conv2D, MaxPooling2D, Flatten, Dense(Softmax)
Optimization	Adam Optimizer
Training Goal	Minimized Sparse Categorical Crossentropy Loss
