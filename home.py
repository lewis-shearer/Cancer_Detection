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
    st.title("🩺 Welcome to the CNN Cancer Detection App 🔬")
    st.write("""
    👋 **Hello! Welcome to an advanced tool for cancer detection.**

    This application leverages the power of Convolutional Neural Networks (CNNs) to analyze medical images and identify potential signs of various types of cancer. Our goal is to provide a user-friendly platform that can serve as an educational and preliminary screening tool.
    """)
    
    st.info("Please use the sidebar on the left to navigate through the app's features. 🧭")

    st.write("---")

    st.subheader("🚀 How It Works")
    st.write("""
    The process is simple:
    1.  **Navigate**: Use the sidebar to go to the **Detection** or **Demo** page.
    2.  **Upload**: On the **Detection** page, upload a medical image you want to analyze.
    3.  **Select**: Choose the corresponding cancer type from the dropdown menu.
    4.  **Analyze**: Click "Run Detection" to let the CNN model process the image.
    5.  **Review**: The app will display the prediction, confidence score, and a Grad-CAM heatmap showing which parts of the image the model focused on.
    """)

    st.write("---")

    st.subheader("🌟 Key Features")
    st.write("""
    - **Multi-Cancer Detection**: Supports various cancer types including Brain, Breast, and more.
    - **Instant Predictions**: Get results in seconds.
    - **Visual Feedback**: Grad-CAM heatmaps provide insight into the model's decision-making process.
    - **In-Depth Information**: Explore the "Backend Information" section to learn about the model architecture, training, and datasets.
    """)

    st.warning("⚠️ **Disclaimer**: This application is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any medical concerns. 👩‍⚕️👨‍⚕️")

def detection_page():
    st.title("🔬 Cancer Detection 📊")
    st.write("**Upload an image and select a cancer type to get a prediction.**")
    st.write("The model will analyze the image and provide a prediction along with a confidence score and a heatmap indicating the areas of interest.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    temp_file_path = None
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

    cancer_type = st.selectbox("Select Cancer Area",
                               ["Brain", "Breast", "Cervical", "Kidney", "Lung/Colon", "Lymphoma", "Oral"])
    
    st.write(f"**Selected Area:** {cancer_type}")

    if st.button("Run Detection"):
        if temp_file_path is None:
            st.error("Please upload an image first.")
            return
            
        with st.spinner("🔍 Analyzing Image... Please wait."):
            label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5, model_path = predict(
                img=temp_file_path, type=cancer_type)

            st.subheader("📈 Prediction Results")
            
            # Display main prediction
            st.markdown(f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center;">
                <h2>Prediction: <span style="color: #FF4B4B;">{label_1}</span></h2>
                <h3>Confidence: <span style="color: #FF4B4B;">{prob_1}%</span></h3>
            </div>
            """, unsafe_allow_html=True)

            # Display other class probabilities in an expander
            with st.expander("View Other Class Probabilities"):
                if cancer_type == 'Brain':
                    st.write(f"Confidence of {label_2}: {prob_2}%")
                    st.write(f"Confidence of {label_3}: {prob_3}%")
                elif cancer_type == 'Breast' or cancer_type == 'Kidney' or cancer_type == 'Oral':
                    st.write(f"Confidence of {label_2}: {prob_2}%")
                elif cancer_type == 'Cervical' or cancer_type == 'Lung/Colon':
                    st.write(f"Confidence of {label_2}: {prob_2}%")
                    st.write(f"Confidence of {label_3}: {prob_3}%")
                    st.write(f"Confidence of {label_4}: {prob_4}%")
                    st.write(f"Confidence of {label_5}: {prob_5}%")
                elif cancer_type == 'Lymphoma':
                    st.write(f"Confidence of {label_2}: {prob_2}%")
                    st.write(f"Confidence of {label_3}: {prob_3}%")

            # --- Grad-CAM Generation ---
            model = tf.keras.models.load_model(model_path)
            image_array = tf.expand_dims(cv2.resize(cv2.imread(temp_file_path), (224, 224)), axis=0)
            preds = model.predict(image_array)
            
            if isinstance(preds, list):
                preds = preds[0]
            
            if preds.shape[1] == 1:
                predicted_class = 0
            else:
                predicted_class = tf.argmax(preds[0])

            last_conv_layer_name = "conv_1_bn" 
            heatmap = generate_gradcam_heatmap(model, image_array, last_conv_layer_name, pred_index=predicted_class)
            
            original_img = cv2.cvtColor(cv2.imread(temp_file_path), cv2.COLOR_BGR2RGB)
            superimposed_img = overlay_gradcam_on_image(original_img, heatmap)

            st.subheader("🔥 Grad-CAM Heatmap")
            st.image(superimposed_img, caption="This heatmap shows the regions the model focused on for its prediction.", use_column_width=True)

        # Clean up the temporary file
        if temp_file_path:
            os.remove(temp_file_path)

def model_architecture_page():
    st.title("🏗️ Model Architecture: MobileNetV3Large")
    st.write("""
    The deep learning models used in this application are built upon the **MobileNetV3Large** architecture. This is a state-of-the-art, lightweight, and powerful convolutional neural network designed for high performance on mobile and embedded devices, making it perfect for efficient and fast predictions.
    """)

    st.subheader("Key Features of MobileNetV3")
    st.write("""
    - **Efficiency**: It uses a combination of depthwise separable convolutions and inverted residual blocks with linear bottlenecks to drastically reduce the number of parameters and computational cost compared to traditional CNNs.
    - **Squeeze-and-Excitation Blocks**: It incorporates lightweight attention modules (Squeeze-and-Excitation) that allow the network to learn the importance of different channels in the feature maps, improving accuracy with minimal overhead.
    - **h-swish Activation Function**: It utilizes a novel, computationally efficient approximation of the swish activation function, which provides better accuracy than standard ReLU.
    - **Platform-Aware NAS**: The architecture was discovered through a combination of hardware-aware Network Architecture Search (NAS) and manual refinements, ensuring it is optimized for real-world performance.
    """)
    
    st.info("For this project, the pre-trained MobileNetV3Large model was used as a base, and a custom classifier head was added and fine-tuned on our specific cancer datasets. This technique, known as **transfer learning**, allows us to leverage the powerful features learned by the model on a large dataset (ImageNet) and adapt them to our medical imaging task.")

def training_process_page():
    st.title("🚂 Training Process")
    st.write("The models were trained using a systematic process to ensure robustness and high accuracy.")

    st.subheader("1. Data Preprocessing and Augmentation")
    st.write("""
    - **Resizing**: All images were resized to a standard dimension of 224x224 pixels to match the input size of the MobileNetV3 architecture.
    - **Normalization**: Pixel values were scaled to a range of [0, 1] to help stabilize the training process.
    - **Data Augmentation**: To prevent overfitting and make the model more robust to variations in images, the training data was augmented with random transformations such as:
        - Rotations
        - Horizontal and vertical flips
        - Zooming
        - Brightness adjustments
    """)

    st.subheader("2. Transfer Learning")
    st.write("""
    - The **MobileNetV3Large** model, pre-trained on the ImageNet dataset, was used as the base.
    - The original classification layer of the model was removed.
    - A new classifier head was added on top, consisting of a Global Average Pooling layer followed by Dense (fully connected) layers and a final output layer with a Softmax (for multi-class) or Sigmoid (for binary) activation function.
    """)

    st.subheader("3. Fine-Tuning")
    st.write("""
    - **Optimizer**: The models were trained using the **Adamax** optimizer, which is a variant of Adam that is well-suited for models with sparse parameter updates.
    - **Loss Function**: **Categorical Cross-Entropy** was used for multi-class classification tasks, and **Binary Cross-Entropy** for binary tasks.
    - **Epochs**: The models were trained for a set number of epochs, with early stopping implemented to prevent overfitting. The training would halt if the validation loss did not improve for a certain number of consecutive epochs.
    - **Batch Size**: A suitable batch size was chosen to balance training speed and memory usage.
    """)

def datasets_page():
    st.title("💾 Datasets")
    st.write("The models were trained on a comprehensive collection of medical images for various cancer types.")
    st.write("The primary dataset used is the **Multi-Cancer Image Dataset**, which is a large-scale, curated collection of images for different cancer classifications.")
    
    st.info("You can find the dataset on Kaggle: [Multi-Cancer Image Dataset](https://www.kaggle.com/datasets/obulisainaren/multi-cancer)")

    st.subheader("Dataset Details")
    st.write("""
    This dataset contains thousands of images categorized into the following cancer types, which correspond to the models in this app:
    - **Brain Cancer**: Images classified into Glioma, Meningioma, and Pituitary tumors.
    - **Breast Cancer**: Histopathological images classified as Benign or Malignant.
    - **Cervical Cancer**: Images of different cell types.
    - **Kidney Cancer**: CT scan images showing Normal kidney tissue and Tumors.
    - **Lung and Colon Cancer**: Histopathological images for Colon Adenocarcinoma, Benign Colon tissue, Lung Adenocarcinoma, Benign Lung tissue, and Lung Squamous Cell Carcinoma.
    - **Lymphoma**: Images of different lymphoma subtypes.
    - **Oral Cancer**: Images of Normal tissue and Oral Squamous Cell Carcinoma.
    """)

def results_page():
    st.title("📊 Model Performance & Results")
    st.write("The performance of the trained models was evaluated using several standard classification metrics to ensure their reliability and accuracy.")

    st.subheader("Evaluation Metrics")
    st.write("""
    - **Accuracy**: The proportion of correct predictions out of the total number of predictions. It's a good general measure of performance.
    - **Precision**: Measures the accuracy of positive predictions. It answers the question: "Of all the predictions that were 'cancer', how many were actually correct?"
    - **Recall (Sensitivity)**: Measures the model's ability to find all the actual positive samples. It answers: "Of all the actual cancer cases, how many did the model correctly identify?"
    - **F1-Score**: The harmonic mean of Precision and Recall, providing a single score that balances both concerns.
    - **Confusion Matrix**: A table that visualizes the performance of the model, showing the counts of true positive, true negative, false positive, and false negative predictions.
    """)

    st.subheader("Overall Performance")
    st.write("The models achieved high accuracy across all cancer types, demonstrating their effectiveness in distinguishing between different classes. Fine-tuning the MobileNetV3Large architecture proved to be a successful strategy, leading to robust models that can generalize well to new, unseen images.")
    st.info("For detailed, per-class metrics and confusion matrices, please refer to the original model training notebooks.")

def demo_page():
    st.title("🖼️ Demo")
    st.write("**Try out the model with our pre-selected demo images.**")
    st.write("Click on an image preview, select the corresponding cancer type, and then press 'Run Detection' to see the model in action.")

    demo_images = {
        "Brain Glioma": "images/brain_glioma_0038.jpg",
        "Brain Menin": "images/brain_menin_0039.jpg",
        "Brain Tumor": "images/brain_tumor_0021.jpg",
        "Breast Benign": "images/breast_benign_0003.jpg",
        "Breast Malignant": "images/breast_malignant_0002.jpg",
    }

    selected_image_key = st.session_state.get("selected_image_key", list(demo_images.keys())[0])

    cols = st.columns(len(demo_images))
    for i, (key, path) in enumerate(demo_images.items()):
        with cols[i]:
            try:
                img = Image.open(path)
                # Add a border if the image is selected
                border_style = "border: 3px solid #FF4B4B;" if selected_image_key == key else ""
                st.image(img, caption=key, use_column_width=True)
                if st.button("Select", key=f"btn_{key}"):
                    st.session_state.selected_image_key = key
                    st.rerun() # Rerun to update the selection visually
            except FileNotFoundError:
                st.error(f"Image not found: {path}.")

    st.write("---")
    st.write(f"**Selected Image:** {st.session_state.get('selected_image_key', 'None')}")
    
    # Automatically select the cancer type based on the image key
    selected_key = st.session_state.get('selected_image_key', '')
    if 'Brain' in selected_key:
        cancer_type_index = 0
    elif 'Breast' in selected_key:
        cancer_type_index = 1
    else:
        cancer_type_index = 0 # Default

    cancer_type = st.selectbox("Select Cancer Area", 
                               ["Brain", "Breast", "Cervical", "Kidney", "Lung/Colon", "Lymphoma", "Oral"],
                               index=cancer_type_index)
    
    st.write(f"**Selected Area:** {cancer_type}")

    if st.button("Run Detection", key="demo_run"):
        with st.spinner("🔍 Analyzing Image... Please wait."):
            selected_image_path = demo_images[selected_image_key]
            
            label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5, model_path = predict(img=selected_image_path,  type=cancer_type)

            st.subheader("📈 Prediction Results")
            st.markdown(f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center;">
                <h2>Prediction: <span style="color: #FF4B4B;">{label_1}</span></h2>
                <h3>Confidence: <span style="color: #FF4B4B;">{prob_1}%</span></h3>
            </div>
            """, unsafe_allow_html=True)

            # --- Grad-CAM Generation ---
            model = tf.keras.models.load_model(model_path)
            image_array = tf.expand_dims(cv2.resize(cv2.imread(selected_image_path), (224, 224)), axis=0)
            preds = model.predict(image_array)
            
            if isinstance(preds, list):
                preds = preds[0]

            if preds.shape[1] == 1:
                predicted_class = 0
            else:
                predicted_class = tf.argmax(preds[0])
                
            last_conv_layer_name = "conv_1_bn"
            heatmap = generate_gradcam_heatmap(model, image_array, last_conv_layer_name, pred_index=predicted_class)

            original_img = cv2.cvtColor(cv2.imread(selected_image_path), cv2.COLOR_BGR2RGB)
            superimposed_img = overlay_gradcam_on_image(original_img, heatmap)

            st.subheader("🔥 Grad-CAM Heatmap")
            st.image(superimposed_img, caption="This heatmap shows the regions the model focused on for its prediction.", use_column_width=True)

def main():
    st.sidebar.title("🧭 Navigation")
    
    # Main pages
    pages = {
        "🩺 Welcome": welcome_page,
        "🔬 Detection": detection_page,
        "🖼️ Demo": demo_page,
        "🧠 Backend Information": None, # Placeholder
    }
    
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Backend Information sub-pages
    backend_pages = {
        "Model Architecture": model_architecture_page,
        "Training Process": training_process_page,
        "Datasets": datasets_page,
        "Results": results_page,
    }

    if selected_page == "🧠 Backend Information":
        st.sidebar.markdown("---")
        backend_selected_page = st.sidebar.radio("Backend Details", list(backend_pages.keys()))
        backend_pages[backend_selected_page]()
    else:
        pages[selected_page]()

if __name__ == "__main__":
    main()
