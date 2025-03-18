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
def predict(img, type):
      # Import Adamax

    # Set a seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    model_path = 'na'

    if type == 'Brain':
        labels = ['Glioma', 'Menin', 'Tumor']
        model_path = 'models/Brain_model_final.h5'

    elif type == 'Breast':
        labels = ['Benign', 'Malignant']
        model_path = 'models/Breast_model_final.h5'

    elif type == 'Cervical':
        labels = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        model_path = 'models/Cervix_model_final.h5'

    elif type == 'Kidney':
        labels = ['Normal', 'Tumor']
        model_path = 'models/Kidney_model_final.h5'

    elif type == 'Lung/Colon':
        labels = ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue',
                  'Lung Squamous Cell Carcinoma']
        model_path = 'models/Lung_model_final.h5'

    elif type == 'Lymphoma':
        labels = ['Chronic Lymphocytic Leukemia', 'Follicular Lymphoma', 'Mantle Cell Lymphoma']
        model_path = 'models/Lymphoma_model_final.h5'

    elif type == 'Oral':
        labels = ['Normal', 'Oral Squamous Cell Carcinoma']
        model_path = 'models/Oral_model_final.h5'

    if model_path == 'na':
        print(f"Error: Unsupported cancer type: {type}")
        return None, None, None, None, None, None, None, None, None, None  # added to stop error.

    model = tf.keras.models.load_model(model_path, compile=False)
    optimizer = Adamax(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss='BinaryCrossentropy', metrics=['accuracy'])

    
def preprocess(img, labels):
    nonlocal model
    img = tf.io.read_file(img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    if img is None:
        print("Error: Could not read image.")
        return None, None  # added none, none

    img_resized = tf.image.resize(img, [224, 224])
    img_array = tf.expand_dims(img_resized, axis=0)

    prediction = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = tf.argmax(prediction[0])

    # Get the predicted label
    predicted_label = labels[predicted_class_index]

    # Get the probabilities
    probabilities = prediction[0].numpy().tolist()
    print(predicted_label, probabilities)

    return probabilities, labels  

    def pad_to_length_6(array):
        probs_array = np.array(array, dtype=object)
        current_length = len(probs_array)
        print(f"Padding array. Current length: {current_length}")

        if current_length >= 6:
            print("No padding needed.")
            return probs_array[:6]
        else:
            padding_length = 6 - current_length
            padding = np.full(padding_length, 'NA', dtype=object)
            print(f"Padding with {padding_length} 'NA's")
            return np.concatenate((probs_array, padding))

    def pred_with_lables(img_path, labels):
        print(f"pred_with_lables input: {img_path}, {labels}")
        probs, labels = preprocess(img_path, labels)

        if probs is None:
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'

        print(f"Original probs: {probs}")
        print(f"Original labels: {labels}")

        label_probs = list(zip(probs, labels))
        print(f"Zipped label_probs: {label_probs}")

        label_probs.sort(key=lambda x: x[0], reverse=True)
        print(f"Sorted label_probs: {label_probs}")

        sorted_probs, sorted_labels = zip(*label_probs)
        print(f"Unzipped sorted_probs: {sorted_probs}")
        print(f"Unzipped sorted_labels: {sorted_labels}")

        sorted_probs = list(sorted_probs)
        sorted_labels = list(sorted_labels)

        if not sorted_probs:
            print("Warning: No probabilities to process.")
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'

        sorted_probs = [prob * 100 for prob in sorted_probs]
        sorted_probs = [round(prob, 2) if isinstance(prob, (int, float)) else prob for prob in sorted_probs]

        padded_labels = pad_to_length_6(sorted_labels)
        padded_probs = pad_to_length_6(sorted_probs)

        print(f"Padded labels: {padded_labels}")
        print(f"Padded probs: {padded_probs}")

        label_1 = padded_labels[0]
        prob_1 = padded_probs[0]
        label_2 = padded_labels[1]
        prob_2 = padded_probs[1]
        label_3 = padded_labels[2]
        prob_3 = padded_probs[2]
        label_4 = padded_labels[3]
        prob_4 = padded_probs[3]
        label_5 = padded_labels[4]
        prob_5 = padded_probs[4]

        return label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5

    label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5 = pred_with_lables(img, labels)

    return label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5




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
                img=selected_image_path, type=cancer_type)

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
            label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5 = predict(img=demo_images[selected_image_key],  type=cancer_type)

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
