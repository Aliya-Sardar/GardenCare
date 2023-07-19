import tensorflow as tf
import numpy as np
import os

def recognize(base64_image_data):
    class_names = ['Grape___Black_rot', 'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Rose___Black_spot', 'Rose___healthy', 'Tomato___Late_blight', 'Tomato___healthy']

    # Load the h5 model and allocate tensors.
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, 'Resnet50.h5')
    model = tf.keras.models.load_model(file_path)

    # Define the target image size
    target_size = (224, 224)

    # Decode the base64 image data and convert it to a TensorFlow tensor
    decoded_image = tf.io.decode_base64(base64_image_data)

    # Decode the image using tf.io.decode_image
    image = tf.io.decode_image(decoded_image, channels=3)  # Set channels=1 for grayscale images

    # Resize the image to the target size
    image = tf.image.resize(image, target_size)

    # Expand the image dimensions to create a batch of size 1 and preprocess it
    image = tf.expand_dims(image, axis=0)
    image_array = tf.keras.applications.resnet50.preprocess_input(image)

    # Make predictions on the image
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions[0])

    # Print the predicted class index
    print("Predicted Class Index:", predicted_class_index)
    predicted_class_label = class_names[predicted_class_index]
    return predicted_class_label
