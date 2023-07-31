import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import os

# Read image labels from a CSV file
labels = pd.read_csv('labels.csv')
unique_labels = np.unique(labels["labels"])

# Define image size
IMG_SIZE = 224

# Function for preprocessing images
def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

# Function to create data batches
def create_data_batches(X, y=None, batch_size=32, val_data=False, test_data=False):
    if test_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(process_image).batch(batch_size)
        return data_batch
    elif val_data:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch
    else:
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch

# Function to load the saved model
def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})

# Function to get the predicted label
def get_pred_label(prediction_probabilities):
    return unique_labels[np.argmax(prediction_probabilities)]

# Function to display custom image predictions and labels
def display_custom_predictions(custom_paths, predictions):
    for i, custom_path in enumerate(custom_paths):
        print(f"Image: {custom_path}")
        print(f"Predicted label: {get_pred_label(predictions[i])}")
        print()

# Load the saved model
model = load_model("model.h5")

# Get custom image filepaths
custom_path = "path/to/saved/images"
custom_image_paths = [os.path.join(custom_path, fname) for fname in os.listdir(custom_path)]

# Turn custom images into batch datasets
custom_data = create_data_batches(custom_image_paths, test_data=True)

# Make predictions on the custom data
custom_preds = model.predict(custom_data)

# Display custom image predictions and labels
display_custom_predictions(custom_image_paths, custom_preds)
