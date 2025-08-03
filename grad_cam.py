import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def generate_gradcam_heatmap(model, image_array, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap.
    """
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(image_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Handle binary vs. categorical models
        if preds.shape[1] == 1: # Binary classification
             class_channel = preds[0]
        else: # Categorical classification
             class_channel = preds[:, pred_index]


    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Saves and displays the Grad-CAM heatmap overlaid on the original image.
    """
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Matplotlib uses RGB

    # Resize the heatmap to be the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Apply the heatmap to the original image
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = np.array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    print(f"Saved Grad-CAM image to {cam_path}")


def get_model_and_layer(image_filename):
    """
    Determines the cancer type from the filename and returns the corresponding model path and conv layer name.
    """
    if 'brain' in image_filename:
        return 'models/brain_cancer_model.keras', "conv_1_bn"
    if 'breast' in image_filename:
        return 'models/breast_cancer_model.keras', "conv_1_bn"
    # Add other model mappings here if needed
    # e.g., if 'cervical' in image_filename: return 'models/cervical_model.h5', "some_layer"
    return None, None

def test_all_images():
    """
    Generates Grad-CAM for all images in the 'images' folder.
    """
    images_dir = 'images'
    output_dir = 'grad_cam_outputs'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(images_dir):
        print(f"Error: Directory '{images_dir}' not found.")
        return

    for image_filename in os.listdir(images_dir):
        if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(images_dir, image_filename)
            print(f"\nProcessing {img_path}...")

            model_path, last_conv_layer = get_model_and_layer(image_filename)

            if not model_path:
                print(f"  - Skipping: No model mapping found for {image_filename}")
                continue
            
            if not os.path.exists(model_path):
                print(f"  - Skipping: Model file not found at {model_path}")
                continue

            # Load model
            model = tf.keras.models.load_model(model_path)

            # Preprocess image
            img = cv2.imread(img_path)
            img_array = tf.expand_dims(cv2.resize(img, (224, 224)), axis=0)

            # Generate heatmap
            heatmap = generate_gradcam_heatmap(model, img_array, last_conv_layer)

            # Save overlay
            output_filename = f"{os.path.splitext(image_filename)[0]}_gradcam.jpg"
            output_path = os.path.join(output_dir, output_filename)
            save_and_display_gradcam(img_path, heatmap, output_path)

if __name__ == '__main__':
    test_all_images()