# 🧬 Cancer Detection

A deep learning application for detecting cancer from medical images using **TensorFlow**. The project provides an easy-to-use web interface powered by **Streamlit**, allowing users to upload images and receive AI-driven diagnostic insights.  

🔗 **Live Demo:** [Cancer Detection App](https://cancerdetection-lewis.streamlit.app/)  
📊 **Dataset:** [Multi-Cancer Image Dataset (Kaggle)](https://www.kaggle.com/datasets/obulisainaren/multi-cancer)  

---

## 📑 Table of Contents
- [Introduction](#-introduction)  
- [Features](#-features)  
- [Tech Stack](#-tech-stack)  
- [Dataset](#-dataset)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Project Structure](#-project-structure)  
- [Examples](#-examples)  
- [Troubleshooting](#-troubleshooting)  
- [Contributors](#-contributors)  
- [License](#-license)  

---

## 🌟 Introduction
This project leverages **Convolutional Neural Networks (CNNs)** to classify medical images for cancer detection. It is designed to:  
- Provide a quick and accessible way to test cancer detection models.  
- Showcase model explainability with **Grad-CAM visualizations**.  
- Serve as a proof-of-concept for applying AI to real-world healthcare challenges.  

---

## ✨ Features
- Upload medical images directly via the web interface.  
- Get real-time cancer predictions with confidence scores.  
- Visualize which image regions influenced the model’s decision using Grad-CAM.  
- Simple deployment via **Streamlit Cloud**.  

---

## 🛠 Tech Stack
- **Python 3.9+**  
- **TensorFlow / Keras** (model creation & training)  
- **NumPy, Pandas, Matplotlib** (data processing & visualization)  
- **Grad-CAM** (explainability)  
- **Streamlit** (interactive UI)  

---

## 📊 Dataset
The model is trained using the [**Multi-Cancer Image Dataset**](https://www.kaggle.com/datasets/obulisainaren/multi-cancer), which contains medical images across multiple cancer types.  

⚠️ **Note:** The dataset is used strictly for research and educational purposes. This application is **not a medical diagnostic tool** and should not be relied upon for clinical decisions.  

---

## ⚙️ Installation
Clone the repository and install dependencies:  

```bash
git clone https://github.com/lewis-shearer/Cancer_Detection.git
cd Cancer_Detection
pip install -r requirements.txt
 ```

## 🚀 Usage
Run the app locally:  

```bash
streamlit run app.py
 ```
Then open http://localhost:8501 in your browser.

For deployment, push your repo to Streamlit Cloud or another hosting service.

## 📂 Project Structure

```bash
Cancer_Detection/
│
├── app.py                 # Streamlit app entry point
├── requirements.txt       # Dependencies
├── model_creation/        # Scripts for training models
├── grad_cam/              # Explainability visualizations
├── .devcontainer/         # Dev environment setup
└── README.md              # Project documentation
 ```

## 🖼 Examples
- Upload an X-ray or histopathology image.  
- The model outputs **Cancer Detected** or **No Cancer** with a probability score.  
- Grad-CAM heatmaps highlight critical regions of the image.  

---

## 🛠 Troubleshooting
- **Module not found errors** → Run `pip install -r requirements.txt` again.  
- **Streamlit app won’t launch** → Ensure you’re in the correct project directory.  
- **TensorFlow issues on M1 Macs** → Install `tensorflow-macos` and `tensorflow-metal`.  

---

## 👥 Contributors
- [Lewis Shearer](https://github.com/lewis-shearer)  

Contributions are welcome! Feel free to fork, submit issues, or open pull requests.  

---

## 📜 License
This project is licensed under the **MIT License**.  

