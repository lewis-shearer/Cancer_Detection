import os
import numpy as np
import pandas as pd
import cv2
import random
import tensorflow as tf

def classes(type):
    if type == 'Brain':
        labels = ['Glioma', 'Menin', 'Tumor']
        model_path = 'models/brain_model_final.h5'

    elif type == 'Breast':
        labels = ['Benign', 'Malignant']
        model_path = 'models/breast_model_final.h5'

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
    
    else:
        return labels, model_path

def preproccess(img):
    img = cv2.imread(img)
    if img is None:
        print("Error: Could not read image.")
        return None, None  # added none, none

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)

    return img_array

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def predict(image, model):
    prediction = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(prediction[0])

    # Get the predicted label
    predicted_label = labels[predicted_class_index]

    # Get the probabilities
    probabilities = prediction[0].tolist()
    print(predicted_label, probabilities)

    return probabilities, labels  # corrected return statement.


image = 'images/images/brain_glioma_0038.jpg'
type = 'Brain'

labels, model_path = classes(type = type)
image_array = preproccess(image)
model = load_model(model_path= model_path)
predict(image=image_array, model=model)
