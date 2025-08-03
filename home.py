import tempfile
import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt


# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap(model, image_array, last_conv_layer_name, pred_index=None):

    # Create a model that outputs the last conv layer and predictions
    # Using model.inputs instead of [model.inputs] to avoid nested lists
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record the operations for automatic differentiation.
    # The gradient tape will watch the operations on the inputs
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image_array)
        # If the model has multiple outputs, preds will be a list.
        # We are interested in the first prediction output.
        if isinstance(preds, list):
            preds = preds[0]

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Handle binary vs. categorical models
        if preds.shape[1] == 1: # Binary classification
             class_channel = preds[0]
        else: # Categorical classification
             class_channel = preds[:, pred_index]


    # This is the gradient of the output neuron (top predicted or chosen)
    # with respect to the output of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients over the feature map dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get the output of the last conv layer
    conv_outputs = last_conv_layer_output[0]

    # Compute the heatmap by multiplying the pooled gradients with the conv outputs
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam_on_image(img, heatmap, alpha=0.4):
    """Overlays the heatmap on the original image."""
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    return superimposed_img


def predict(img, type):
    import os
    import numpy as np
    import pandas as pd
    import cv2
    import random
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adamax  # Import Adamax

    # Set a seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    model_path = 'na'

    if type == 'Brain':
        labels = ['Glioma', 'Menin', 'Tumor']
        model_path = 'models/brain_cancer_model.keras'

    elif type == 'Breast':
        labels = ['Benign', 'Malignant']
        model_path = 'models/breast_cancer_model.keras'

    elif type == 'Cervical':
        labels = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        model_path = 'models/cervix_cancer_model.keras'

    elif type == 'Kidney':
        labels = ['Normal', 'Tumor']
        model_path = 'models/kidney_cancer_model.keras'

    elif type == 'Lung/Colon':
        labels = ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue',
                  'Lung Squamous Cell Carcinoma']
        model_path = 'models/lung_colon_cancer_model.keras'

    elif type == 'Lymphoma':
        labels = ['Chronic Lymphocytic Leukemia', 'Follicular Lymphoma', 'Mantle Cell Lymphoma']
        model_path = 'models/lymphoma_cancer_model.keras'

    elif type == 'Oral':
        labels = ['Normal', 'Oral Squamous Cell Carcinoma']
        model_path = 'models/oral_cancer_model.keras'

    if model_path == 'na':
        print(f"Error: Unsupported cancer type: {type}")
        return None, None, None, None, None, None, None, None, None, None  # added to stop error.

    model = tf.keras.models.load_model(model_path)

    def preprocess(img, labels):
        nonlocal model
        img = cv2.imread(img)
        if img is None:
            print("Error: Could not read image.")
            return None, None  # added none, none

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_array = np.expand_dims(img_resized, axis=0)

        prediction = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(prediction[0])

        # Get the predicted label
        predicted_label = labels[predicted_class_index]

        # Get the probabilities
        probabilities = prediction[0].tolist()
        

        return probabilities, labels  # corrected return statement.

    def pad_to_length_6(array):
        probs_array = np.array(array, dtype=object)
        current_length = len(probs_array)
        

        if current_length >= 6:
            
            return probs_array[:6]
        else:
            padding_length = 6 - current_length
            padding = np.full(padding_length, 'NA', dtype=object)
            
            return np.concatenate((probs_array, padding))

    def pred_with_lables(img_path, labels):
        
        probs, labels = preprocess(img_path, labels)

        if probs is None:
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'

        

        label_probs = list(zip(probs, labels))
        

        label_probs.sort(key=lambda x: x[0], reverse=True)
        
        sorted_probs, sorted_labels = zip(*label_probs)
        

        sorted_probs = list(sorted_probs)
        sorted_labels = list(sorted_labels)

        if not sorted_probs:
            print("Warning: No probabilities to process.")
            return 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'

        sorted_probs = [prob * 100 for prob in sorted_probs]
        sorted_probs = [round(prob, 2) if isinstance(prob, (int, float)) else prob for prob in sorted_probs]

        padded_labels = pad_to_length_6(sorted_labels)
        padded_probs = pad_to_length_6(sorted_probs)

       

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

    return label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5, model_path

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

      


    cancer_type = st.selectbox("Select Cancer Area",
                               ["Brain", "Breast", "Cervical", "Kidney", "Lung/Colon", "Lymphoma", "Oral"])
    if "selected_image_key" in st.session_state:
        st.write(f"Currently Selected Image: {st.session_state.selected_image_key}")
    st.write(f"Currently Selected Area: {cancer_type}")
    if st.button("Run Detection"):
        with st.spinner("Running Detection..."):  # Loading box

            selected_image_path = temp_file_path
            st.session_state.demo_image_selected = temp_file_path
            label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5, model_path = predict(
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
                
            model = tf.keras.models.load_model(model_path)
            image_array = tf.expand_dims(cv2.resize(cv2.imread(temp_file_path), (224, 224)), axis=0)
            preds = model.predict(image_array)
            
            # For binary classification with single output, argmax is always 0
            if preds.shape[1] == 1:
                predicted_class = 0
            else:
                predicted_class = tf.argmax(preds[0])

            # --- Grad-CAM Generation ---
            last_conv_layer_name = "conv_1_bn" 
            heatmap = generate_gradcam_heatmap(model, image_array, last_conv_layer_name, pred_index=predicted_class)
            
            # Overlay heatmap on original image
            original_img = cv2.cvtColor(cv2.imread(temp_file_path), cv2.COLOR_BGR2RGB)
            superimposed_img = overlay_gradcam_on_image(original_img, heatmap)

            # Display Grad-CAM
            st.subheader("Grad-CAM Heatmap")
            st.image(superimposed_img, caption="Model Activation Heatmap", use_container_width=True)


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
    st.title("📋 Demo 🖼️")
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
                st.image(img, caption=key, use_container_width=False, output_format="PNG")
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
            label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5, model_path = predict(img=demo_images[selected_image_key],  type=cancer_type)

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
            
            model = tf.keras.models.load_model(model_path)
            image_array = tf.expand_dims(cv2.resize(cv2.imread(temp_file_path), (224, 224)), axis=0)
            preds = model.predict(image_array)
            
            # For binary classification with single output, argmax is always 0
            if preds.shape[1] == 1:
                predicted_class = 0
            else:
                predicted_class = tf.argmax(preds[0])

            # --- Grad-CAM Generation ---
            last_conv_layer_name = "conv_1_bn" 
            heatmap = generate_gradcam_heatmap(model, image_array, last_conv_layer_name, pred_index=predicted_class)
            
            # Overlay heatmap on original image
            original_img = cv2.cvtColor(cv2.imread(temp_file_path), cv2.COLOR_BGR2RGB)
            superimposed_img = overlay_gradcam_on_image(original_img, heatmap)

            # Display Grad-CAM
            st.subheader("Grad-CAM Heatmap")
            st.image(superimposed_img, caption="Model Activation Heatmap", use_column_width=True)


def demo_page():
    st.title("📋 Demo 🖼️")
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
                st.image(img, caption=key, use_container_width=False, output_format="PNG")
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
            label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5, model_path = predict(img=demo_images[selected_image_key],  type=cancer_type)

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
            
            model = tf.keras.models.load_model(model_path)
            image_array = tf.expand_dims(cv2.resize(cv2.imread(demo_images[selected_image_key]), (224, 224)), axis=0)
            preds = model.predict(image_array)

            # For binary classification with single output, argmax is always 0
            if preds.shape[1] == 1:
                predicted_class = 0
            else:
                predicted_class = tf.argmax(preds[0])
                
            # --- Grad-CAM Generation ---
            last_conv_layer_name = "conv_1_bn"
            heatmap = generate_gradcam_heatmap(model, image_array, last_conv_layer_name, pred_index=predicted_class)

            # Overlay heatmap on original image
            original_img = cv2.cvtColor(cv2.imread(demo_images[selected_image_key]), cv2.COLOR_BGR2RGB)
            superimposed_img = overlay_gradcam_on_image(original_img, heatmap)

            # Display Grad-CAM
            st.subheader("Grad-CAM Heatmap")
            st.image(superimposed_img, caption="Model Activation Heatmap", use_column_width=True)
                
            

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
