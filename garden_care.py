import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
#import cv2


def recognize(file):
    class_names = ['Grape___Black_rot', 'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Rose___Black_spot', 'Rose___healthy', 'Tomato___Late_blight', 'Tomato___healthy']

    
    # Load the h5 model and allocate tensors.
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, 'Resnet50.h5')
    model = load_model(file_path)

    # Define the target image size
    target_size = (224, 224)

    # Load and preprocess the image

    #image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    #image = cv2.resize(image, target_size)
    #image_array = img_to_array(image)

    image = load_img(file_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Make predictions on the image
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])

    # Print the predicted class index
    print("Predicted Class Index:", predicted_class_index)
    predicted_class_label = class_names[predicted_class_index]
    return predicted_class_label
