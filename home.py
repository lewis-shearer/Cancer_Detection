import streamlit as st
import random
from PIL import Image
import os
import time

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

    cancer_type = st.selectbox("Select Cancer Type", ["Breast Cancer", "Lung Cancer", "Skin Cancer", "Other"])

    if uploaded_file is not None:
        st.write("✅ Image uploaded successfully! Processing... ⏳")

        # Placeholder for your CNN model prediction logic
        # Replace with actual model prediction and report generation.
        prediction = "[Your Model Prediction Here] 📈"  # Example Prediction
        confidence = "[Your Model Confidence Here] 💯" # Example Confidence
        detailed_report = "Detailed report placeholder." # Example Report

        st.markdown(f"""
        <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
            <strong>Prediction:</strong> {prediction}<br>
            <strong>Confidence:</strong> {confidence}
        </div>
        """, unsafe_allow_html=True)

        detailed_report_requested = st.checkbox("Generate Detailed Report?")

        if detailed_report_requested:
            st.write("---")
            st.subheader("Detailed Report")
            st.write(detailed_report)
        st.write("Cancer type selected: ", cancer_type)

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
        "Brain Tumor 1": "demo_images/brain_tumor_0001.jpg",
        "Brain Tumor 2": "demo_images/brain_tumor_0001.jpg",
        "Lung Cancer 1": "demo_images/brain_tumor_0001.jpg",
        "Skin Melanoma 1": "demo_images/brain_tumor_0001.jpg",
        "Normal Image 1": "demo_images/brain_tumor_0001.jpg",
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

    if "selected_image_key" in st.session_state:
        st.write(f"Currently Selected: {st.session_state.selected_image_key}")

    if st.button("Run Detection"):
        with st.spinner("Running Detection..."): #Loading box
            time.sleep(10) #wait for 10 seconds.
            selected_image_path = demo_images[selected_image_key]
            st.session_state.demo_image_selected = selected_image_path
            prediction = random.choice(["Benign", "Malignant"])
            confidence = random.randint(70, 99)
            report_content = f"Prediction: {prediction}\nConfidence: {confidence}%\nImage Path: {selected_image_path}"

            st.markdown(f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px;">
                <strong>Prediction:</strong> {prediction}<br>
                <strong>Confidence:</strong> {confidence}%
            </div>
            """, unsafe_allow_html=True)

            def download_report():
                return report_content.encode()

            st.download_button(
                label="Download Full Report",
                data=download_report(),
                file_name="detection_report.txt",
                mime="text/plain",
            )
        if "demo_image_selected" in st.session_state:
            st.write(f"Selected Image Path: {st.session_state.demo_image_selected}")


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