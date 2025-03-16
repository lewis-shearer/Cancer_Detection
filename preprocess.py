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

    model = tf.keras.models.load_model(model_path, compile=False)
    optimizer = Adamax(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss='BinaryCrossentropy', metrics=['accuracy'])

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
        print(predicted_label, probabilities)

        return probabilities, labels  # corrected return statement.

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

label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5 = predict('images/brain_glioma_0038.jpg', 'Brain')
print(label_1, prob_1, label_2, prob_2, label_3, prob_3, label_4, prob_4, label_5, prob_5)