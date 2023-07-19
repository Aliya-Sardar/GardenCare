import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import base64

def recognize(base64_image_data):
    class_names = ['Grape___Black_rot', 'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Rose___Black_spot', 'Rose___healthy', 'Tomato___Late_blight', 'Tomato___healthy']

    # Load the h5 model and allocate tensors.
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, 'Resnet50.h5')
    model = load_model(file_path)

    # Define the target image size
    target_size = (224, 224)

    # Decode the base64 image data and convert it to a NumPy array
    decoded_image = base64.b64decode(base64_image_data)
    image_array = np.frombuffer(decoded_image, dtype=np.uint8)

    # Convert the image data to a NumPy array and reshape it to the target size
    image_array = np.reshape(image_array, target_size + (3,))

    # Expand the image array to create a batch of size 1 and preprocess it
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Make predictions on the image
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])

    # Print the predicted class index
    print("Predicted Class Index:", predicted_class_index)
    predicted_class_label = class_names[predicted_class_index]
    return predicted_class_label
