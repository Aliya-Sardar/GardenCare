import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os

def recognize(file):
    class_names = ['Grape___Black_rot', 'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Rose___Black_spot', 'Rose___healthy', 'Tomato___Late_blight', 'Tomato___healthy']

    # Load the TFLite model and allocate tensors.
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, 'Resnet50.tflite')
    interpreter = tf.lite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()

    # Define the target image size
    target_size = (224, 224)

    # Convert the FileStorage object to a NumPy array
    image = Image.open(file)
    image_array = np.array(image)

    # Resize and preprocess the image
    image_array = tf.image.resize(image_array, target_size)
    image_array = img_to_array(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Get input and output details of the TFLite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor to the image data
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor and predicted class index
    output_tensor = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_tensor[0])

    # Print the predicted class index
    print("Predicted Class Index:", predicted_class_index)
    predicted_class_label = class_names[predicted_class_index]
    return predicted_class_label
