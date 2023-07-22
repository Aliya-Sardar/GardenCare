import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from PIL import Image

def recognize(image_data):
    class_names = ['Grape___Black_rot', 'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Rose___Black_spot', 'Rose___healthy', 'Tomato___Late_blight', 'Tomato___healthy']

    # Load the TFLite model and allocate tensors.
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, 'resnet50_tfmodel.tflite')
    interpreter = tf.lite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()

    # Open the image from image_data using PIL
    pil_image = Image.open(image_data)

    # Resize the image to the target size
    target_size = (224, 224)
    pil_image = pil_image.resize(target_size)

    # Convert the PIL Image to a NumPy array
    image_array = img_to_array(pil_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    # Get input and output details of the TFLite model
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor to the image data
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor and make a writable copy
    output_tensor = interpreter.get_tensor(output_details[0]['index']).copy()
    predicted_class_index = np.argmax(output_tensor[0])

    confidence = float(output_tensor[0][predicted_class_index])

    # Return the predicted class
    predicted_class_label = class_names[predicted_class_index]
    return predicted_class_label , (confidence*100)
