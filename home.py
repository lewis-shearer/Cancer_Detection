import streamlit as st
import random
from PIL import Image
import os
import tempfile
import time
import os
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
#from preprocess import predict
def predict(img_path, cancer_type):
    """
    Predicts the type of cancer in an image.

    Args:
        img_path (str): The path to the image file.
        cancer_type (str): The type of cancer to predict (e.g., 'Brain', 'Breast').

    Returns:
        tuple: A tuple containing the top 5 predicted labels and probabilities.
               Returns ('NA', 'NA', ...) if an error occurs.
    """

    # Set random seeds for consistent results
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define cancer types and their corresponding models and labels
    cancer_info = {
        'Brain': {'labels': ['Glioma', 'Menin', 'Tumor'], 'model_path': 'models/Brain_model_final_saved_model'},
        'Breast': {'labels': ['Benign', 'Malignant'], 'model_path': 'models/Breast_model_final_saved_model'},
        'Cervical': {'labels': ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate'], 'model_path': 'models/Cervix_model_final_saved_model'},
        'Kidney': {'labels': ['Normal', 'Tumor'], 'model_path': 'models/Kidney_model_final_saved_model'},
        'Lung/Colon': {'labels': ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma'], 'model_path': 'models/Lung_model_final_saved_model'},
        'Lymphoma': {'labels': ['Chronic Lymphocytic Leukemia', 'Follicular Lymphoma', 'Mantle Cell Lymphoma'], 'model_path': 'models/Lymphoma_model_final_saved_model'},
        'Oral': {'labels': ['Normal', 'Oral Squamous Cell Carcinoma'], 'model_path': 'models/Oral_model_final_saved_model'}
    }

    # Check if the cancer type is supported
    if cancer_type not in cancer_info:
        print(f"Error: Unsupported cancer type: {cancer_type}")
        return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'

    # Load the model and labels
    labels = cancer_info[cancer_type]['labels']
    model_path = cancer_info[cancer_type]['model_path']
    model = tf.keras.models.load_model(model_path, compile=False)
    optimizer = Adamax(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss='BinaryCrossentropy', metrics=['accuracy'])

    def preprocess_image(image_path):
        """Preprocesses the image for prediction."""
        try:
            img_string = tf.io.read_file(image_path)
            img = tf.image.decode_image(img_string, channels=3)
            img_resized = tf.image.resize(img, (224, 224))
            img_array = tf.expand_dims(img_resized, axis=0)
            img_array = tf.cast(img_array, dtype=tf.float32)
            return img_array
        except tf.errors.NotFoundError:
            print(f"Error: Could not read image at {image_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    # Preprocess the image
    img_array = preprocess_image(img_path)
    if img_array is None:
        return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'

    # Make the prediction
    prediction = model.predict(img_array)[0] #Get the prediction for the first image in the batch.
    probabilities = prediction.tolist()

    # Create a list of (probability, label) pairs and sort them
    label_probs = list(zip(probabilities, labels))
    label_probs.sort(key=lambda x: x[0], reverse=True)

    # Extract sorted probabilities and labels
    sorted_probs, sorted_labels = zip(*label_probs)
    sorted_probs = [round(prob * 100, 2) for prob in sorted_probs]

    # Pad the lists with 'NA' if they're shorter than 6
    padded_labels = list(sorted_labels[:6]) + ['NA'] * (6 - len(sorted_labels))
    padded_probs = list(sorted_probs[:6]) + ['NA'] * (6 - len(sorted_probs))

    # Return the top 5 labels and probabilities
    return padded_labels[0], padded_probs[0], padded_labels[1], padded_probs[1], padded_labels[2], padded_probs[2], padded_labels[3], padded_probs[3], padded_labels[4], padded_probs[4]




def welcome_page():
    # ... (welcome page code remains the same) ...
    st.title("🩺 Welcome to the CNN Cancer Detection App 🔬")
    st.write("""
    👋 Hello! This app uses a Convolutional Neural Network (CNN) to help detect cancer from medical images. 🖼️
    """)
    st.write("""
    Please use the sidebar on the left to navigate through the app. 🧭
    """)
    st.write("---")
    st.subheader("🚀 How to Use:")
    st.write("""
    1. 📂 Navigate to the "Detection" page in the sidebar to upload an image and get a prediction. 📊
    2. 📚 Explore the "Backend Information" section for details about the model and its implementation. 🧠
    3. 🖼️ Try our image preview demo on the demo page!
    """)
    st.write("---")
    st.write("⚠️ This application is for informational purposes only and should not be used as a substitute for professional medical advice. 👩‍⚕️👨‍⚕️")

def detection_page():
    # ... (detection page code remains the same) ...
    st.title("🔬 Cancer Detection 📊")
    st.write("Upload an image for cancer detection. 🖼️")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            # Write the uploaded file's content to the temporary file
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        st.write(f"Temporary file path: {temp_file_path}")


    cancer_type = st.selectbox("Select Cancer Area",
                               ["Brain", "Breast", "Cervical", "Kidney", "Lung/Colon", "Lymphoma", "Oral"])
    if "selected_image_key" in st.session_state:
        st.write(f"Currently Selected Image: {st.session_state.selected_image_key}")
    st.write(f"Currently Selected Area: {cancer_type}")
    if st.button("Run Detection"):
        with st.spinner("Running Detection..."):  # Loading box

            selected_image_path = temp_file_path
            st.session_state.demo_image_selected = temp_file_path
            label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5 = predict(
                img_path=selected_image_path, cancer_type=cancer_type)

            if cancer_type == 'Brain':
                st.markdown(f"""
                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                        <strong>Prediction:</strong> {label_1}<br>
                        <strong>Confidence:</strong> {prob_1}%<br><br>
                        <strong>Other Classes</strong>
                        <p>Confidence of {label_2}:  {prob_2}%<br>
                        <p>Confidence of {label_3}:  {prob_3}%<br>

                    """, unsafe_allow_html=True)

            elif cancer_type == 'Breast':
                st.markdown(f"""
                                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                        <strong>Prediction:</strong> {label_1}<br>
                                        <strong>Confidence:</strong> {prob_1}%<br><br>
                                        <strong>Other Classes</strong>
                                        <p>Confidence of {label_2}:  {prob_2}%<br>


                                    """, unsafe_allow_html=True)
            elif cancer_type == 'Cervical':
                st.markdown(f"""
                                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                        <strong>Prediction:</strong> {label_1}<br>
                                        <strong>Confidence:</strong> {prob_1}%<br><br>
                                        <strong>Other Classes</strong>
                                        <p>Confidence of {label_2}:  {prob_2}%<br>
                                        <p>Confidence of {label_3}:  {prob_3}%<br>
                                        <p>Confidence of {label_4}:  {prob_4}%<br>
                                        <p>Confidence of {label_5}:  {prob_5}%<br>

                                    """, unsafe_allow_html=True)
            elif cancer_type == 'Kidney':
                st.markdown(f"""
                                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                        <strong>Prediction:</strong> {label_1}<br>
                                        <strong>Confidence:</strong> {prob_1}%<br><br>
                                        <strong>Other Classes</strong>
                                        <p>Confidence of {label_2}:  {prob_2}%<br>


                                    """, unsafe_allow_html=True)
            elif cancer_type == 'Lung/Colon':
                st.markdown(f"""
                                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                        <strong>Prediction:</strong> {label_1}<br>
                                        <strong>Confidence:</strong> {prob_1}%<br><br>
                                        <strong>Other Classes</strong>
                                        <p>Confidence of {label_2}:  {prob_2}%<br>
                                        <p>Confidence of {label_3}:  {prob_3}%<br>
                                        <p>Confidence of {label_4}:  {prob_4}%<br>
                                        <p>Confidence of {label_5}:  {prob_5}%<br>


                                    """, unsafe_allow_html=True)
            elif cancer_type == 'Lymphoma':
                st.markdown(f"""
                                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                        <strong>Prediction:</strong> {label_1}<br>
                                        <strong>Confidence:</strong> {prob_1}%<br><br>
                                        <strong>Other Classes</strong>
                                        <p>Confidence of {label_2}:  {prob_2}%<br>
                                        <p>Confidence of {label_3}:  {prob_3}%<br>



                                    """, unsafe_allow_html=True)
            elif cancer_type == 'Oral':
                st.markdown(f"""
                                    <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                        <strong>Prediction:</strong> {label_1}<br>
                                        <strong>Confidence:</strong> {prob_1}%<br><br>
                                        <strong>Other Classes</strong>
                                        <p>Confidence of {label_2}:  {prob_2}%<br>



                                    """, unsafe_allow_html=True)

def backend_info_page():
    # ... (backend info page code remains the same) ...
    st.title("🧠 Backend Information 📚")
    st.write("Details about the CNN model and its implementation. ⚙️")
    st.write("""
    Here you can find information about the model architecture, training process, and datasets used. 📝
    """)
    st.write("---")
    st.subheader("🏗️ Model Architecture")
    st.write("Details about the CNN layers and parameters. 🧱")
    st.write("---")
    st.subheader("🚂 Training Process")
    st.write("Information about the training data, epochs, and optimization. 📈")
    st.write("---")
    st.subheader("💾 Datasets")
    st.write("Details about the datasets used for training and testing. 📊")


def demo_page():
    st.title("🖼️ Image Preview Demo")
    st.write("Click on an image preview, then press 'Run Detection'.")

    demo_images = {
        "Brain Glioma": "images/brain_glioma_0038.jpg",
        "Brain Menin": "images/brain_menin_0039.jpg",
        "Brain Tumor": "images/brain_tumor_0021.jpg",
        "Breast Benign": "images/breast_benign_0003.jpg",
        "Breast Malignant": "images/breast_malignant_0002.jpg",
    }

    selected_image_key = st.session_state.get("selected_image_key", list(demo_images.keys())[0])

    cols = st.columns(len(demo_images))
    image_keys = list(demo_images.keys())

    for i, (key, path) in enumerate(demo_images.items()):
        with cols[i]:
            try:
                img = Image.open(path)
                img.thumbnail((100, 100))
                st.image(img, caption=key, use_column_width=False, output_format="PNG")
                if st.button("Select", key=f"btn_{key}"):
                    selected_image_key = key
                    st.session_state.selected_image_key = selected_image_key
            except FileNotFoundError:
                st.error(f"Image not found: {path}.")
    cancer_type = st.selectbox("Select Cancer Area", ["Brain", "Breast", "Cervical", "Kidney", "Lung/Colon", "Lymphoma", "Oral"])
    if "selected_image_key" in st.session_state:
        st.write(f"Currently Selected Image: {st.session_state.selected_image_key}")
    st.write(f"Currently Selected Area: {cancer_type}")
    if st.button("Run Detection"):
        with st.spinner("Running Detection..."): #Loading box

            selected_image_path = demo_images[selected_image_key]
            st.session_state.demo_image_selected = selected_image_path
            label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5 = predict(img_path=demo_images[selected_image_key],  cancer_type=cancer_type)

            if cancer_type == 'Brain':
                st.markdown(f"""
                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                    <strong>Prediction:</strong> {label_1}<br>
                    <strong>Confidence:</strong> {prob_1}%<br><br>
                    <strong>Other Classes</strong>
                    <p>Confidence of {label_2}:  {prob_2}%<br>
                    <p>Confidence of {label_3}:  {prob_3}%<br>
                    
                """, unsafe_allow_html=True)

            elif cancer_type == 'Breast':
                st.markdown(f"""
                                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                    <strong>Prediction:</strong> {label_1}<br>
                                    <strong>Confidence:</strong> {prob_1}%<br><br>
                                    <strong>Other Classes</strong>
                                    <p>Confidence of {label_2}:  {prob_2}%<br>


                                """, unsafe_allow_html=True)
            elif cancer_type == 'Cervical':
                st.markdown(f"""
                                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                    <strong>Prediction:</strong> {label_1}<br>
                                    <strong>Confidence:</strong> {prob_1}%<br><br>
                                    <strong>Other Classes</strong>
                                    <p>Confidence of {label_2}:  {prob_2}%<br>
                                    <p>Confidence of {label_3}:  {prob_3}%<br>
                                    <p>Confidence of {label_4}:  {prob_4}%<br>
                                    <p>Confidence of {label_5}:  {prob_5}%<br>

                                """, unsafe_allow_html=True)
            elif cancer_type == 'Kidney':
                st.markdown(f"""
                                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                    <strong>Prediction:</strong> {label_1}<br>
                                    <strong>Confidence:</strong> {prob_1}%<br><br>
                                    <strong>Other Classes</strong>
                                    <p>Confidence of {label_2}:  {prob_2}%<br>


                                """, unsafe_allow_html=True)
            elif cancer_type == 'Lung/Colon':
                st.markdown(f"""
                                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                    <strong>Prediction:</strong> {label_1}<br>
                                    <strong>Confidence:</strong> {prob_1}%<br><br>
                                    <strong>Other Classes</strong>
                                    <p>Confidence of {label_2}:  {prob_2}%<br>
                                    <p>Confidence of {label_3}:  {prob_3}%<br>
                                    <p>Confidence of {label_4}:  {prob_4}%<br>
                                    <p>Confidence of {label_5}:  {prob_5}%<br>


                                """, unsafe_allow_html=True)
            elif cancer_type == 'Lymphoma':
                st.markdown(f"""
                                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                    <strong>Prediction:</strong> {label_1}<br>
                                    <strong>Confidence:</strong> {prob_1}%<br><br>
                                    <strong>Other Classes</strong>
                                    <p>Confidence of {label_2}:  {prob_2}%<br>
                                    <p>Confidence of {label_3}:  {prob_3}%<br>



                                """, unsafe_allow_html=True)
            elif cancer_type == 'Oral':
                st.markdown(f"""
                                <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                                    <strong>Prediction:</strong> {label_1}<br>
                                    <strong>Confidence:</strong> {prob_1}%<br><br>
                                    <strong>Other Classes</strong>
                                    <p>Confidence of {label_2}:  {prob_2}%<br>



                                """, unsafe_allow_html=True)


def main():
    st.sidebar.title("🧭 Navigation")
    pages = {
        "🩺 Welcome": welcome_page,
        "🔬 Detection": detection_page,
        "🧠 Backend Information": {
            "📚 Model Details": backend_info_page,
        },
        "🖼️ Demo": demo_page
    }

    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    if selected_page == "🧠 Backend Information":
        backend_pages = pages["🧠 Backend Information"]
        backend_selected_page = st.sidebar.radio("Backend", list(backend_pages.keys()))
        backend_pages[backend_selected_page]()

    else:
        pages[selected_page]()

if __name__ == "__main__":
    main()
