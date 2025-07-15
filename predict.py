import os
import numpy as np
import pandas as pd
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import load_model

def classes(type):
    if type == 'Brain':
        labels = ['Glioma', 'Menin', 'Tumor']
        model_path = r"C:\Users\lshea\Desktop\Cancer_Detection\models\Brain_model_final.h5"

    elif type == 'Breast':
        labels = ['Benign', 'Malignant']
        model_path = r'models/Breast_model_final.h5'

    elif type == 'Cervical':
        labels = ['Dyskeratotic', 'Koilocytotic', 'Metaplastic', 'Parabasal', 'Superficial-Intermediate']
        model_path = r'models/Cervix_model_final.h5'

    elif type == 'Kidney':
        labels = ['Normal', 'Tumor']
        model_path = r'models/Kidney_model_final.h5'

    elif type == 'Lung/Colon':
        labels = ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue',
                  'Lung Squamous Cell Carcinoma']
        model_path = r'models/Lung_model_final.h5'

    elif type == 'Lymphoma':
        labels = ['Chronic Lymphocytic Leukemia', 'Follicular Lymphoma', 'Mantle Cell Lymphoma']
        model_path = r'models/Lymphoma_model_final.h5'

    elif type == 'Oral':
        labels = ['Normal', 'Oral Squamous Cell Carcinoma']
        model_path = r'models/Oral_model_final.h5'

    if model_path == 'na':
        print(f"Error: Unsupported cancer type: {type}")


    else:
        print(model_path)
        print('CLASSES LOADED SUCCESSFULLY 🎉')
        return labels, model_path

def preproccess(img):
    img = cv2.imread(img)
    if img is None:
        print("Error: Could not read image.")
        return None, None  # added none, none

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    print('IMAGE PREPROCESSED SUCCESSFULLY 🎉')
    return img_array

def model_load(model_path):
    # Load the model from the specified path and then return the model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    model = load_model(model_path)
    print('MODEL LOADED SUCCESSFULLY 🎉')
    return model


def predict(image, model):
    prediction = model.predict(image)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction[0])

    # Get the predicted label
    predicted_label = labels[predicted_class_index]

    # Get the probabilities
    probabilities = prediction[0].tolist()
    print(predicted_label, probabilities)
    print('MODEL PREDICTED SUCCESSFULLY 🎉')
    return probabilities, labels  # corrected return statement.


# --- Example run ---
image = r"C:\Users\lshea\Desktop\Cancer_Detection\images\brain_glioma_0038.jpg"
type = 'Brain'

labels, model_path = classes(type=type)
if labels is not None and model_path is not None:
    image_array = preproccess(img=image)
    if image_array is not None:
        model = model_load(model_path=model_path)
        if model is not None:
            predict(image=image_array, model=model, labels=labels)
