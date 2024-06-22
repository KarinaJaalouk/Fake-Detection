import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

class RealVsFakePredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.class_names = ['fake', 'real']

    def make_prediction(self, img_path):
        """Make a prediction on a single image"""
        test_image = image.load_img(img_path, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0  # Normalize the image
        
        result = self.model.predict(test_image)
        predicted_class = np.argmax(result, axis=1)
        return predicted_class, result

    def display_prediction(self, img_path):
        """Display the prediction result for a given image path"""
        predicted_class, result = self.make_prediction(img_path)
        print(result)
        print(
            "This image is {} with a {:.2f}% confidence."
            .format(self.class_names[predicted_class[0]], 100 * np.max(result))
        )

# Define the model path
model_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/save_model/model.h5'

# Create the predictor object
predictor = RealVsFakePredictor(model_path)

# Test images
test_image_paths = [
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00016.jpg',
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/valid/real/00008.jpg',
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/fake/00F8LKY6JC.jpg',
    '222.jpg',
    '2222.jpg'
]

# Display predictions for test images
for path in test_image_paths:
    predictor.display_prediction(path)
"""
# Additional test cases
additional_test_image_paths = [
    '1.jpg',
    '2.jpg',
    'raquel.jpg',
    'adrian.jpg',
    'karina.jpg',
    'jorge.jpg',
    'gorge.jpg',
    '777.jpg',
    '888.jpg',
    '999.jpg',
    'dada.jpg',
    'nono.jpg',
    'net.jpg'
]

# Display predictions for additional test images
for path in additional_test_image_paths:
    predictor.display_prediction(path)
"""
