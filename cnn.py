import os
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import pydotplus
import pydot

# Print the number of available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class RealVsFakeClassifier:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.train_dir = os.path.join(base_dir, 'train')
        self.test_dir = os.path.join(base_dir, 'test')
        self.valid_dir = os.path.join(base_dir, 'valid')
        self.class_names = ['fake', 'real']
        self._check_directories()
        self._create_generators()
        self._build_model()
        
    def _check_directories(self):
        """Check if train, test, and validation directories exist"""
        if os.path.exists(self.train_dir) and os.path.exists(self.test_dir) and os.path.exists(self.valid_dir):
            print("All directories exist.")
        else:
            print("One or more directories do not exist.")
            if not os.path.exists(self.train_dir):
                print(f"Directory {self.train_dir} does not exist.")
            if not os.path.exists(self.test_dir):
                print(f"Directory {self.test_dir} does not exist.")
            if not os.path.exists(self.valid_dir):
                print(f"Directory {self.valid_dir} does not exist.")
    
    def _create_generators(self):
        """Create data generators for training, validation, and testing"""
        self.train_datagen = ImageDataGenerator(rescale=1.0/255.0)
        self.test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        self.valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        self.train_generator = self.train_datagen.flow_from_directory(
            self.train_dir, batch_size=100, class_mode='binary', target_size=(150, 150))
        self.validation_generator = self.valid_datagen.flow_from_directory(
            self.valid_dir, batch_size=100, class_mode='binary', target_size=(150, 150))
        self.test_generator = self.test_datagen.flow_from_directory(
            self.test_dir, batch_size=100, class_mode='binary', target_size=(150, 150))
    
    def _build_model(self):
        """Build the CNN model"""
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1064, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        self.model.summary()
        
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    def train(self, epochs=10):
        """Train the model"""
        self.history = self.model.fit(self.train_generator, validation_data=self.validation_generator, epochs=epochs, validation_steps=50, verbose=1)
        self._plot_history()
    
    def _plot_history(self):
        """Plot the training history"""
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.show()
    
    def evaluate(self):
        """Evaluate the model on the test set"""
        test_loss, test_acc = self.model.evaluate(self.test_generator)
        print(f'Test accuracy: {test_acc}')
    
    def predict(self, image_path):
        """Make a prediction on a single image"""
        test_image = image.load_img(image_path, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = self.model.predict(test_image)
        print(result)
        print(
            "This image is {} with a {:.2f}% confidence."
            .format(self.class_names[np.argmax(result)], 100 * np.max(result))
        )
    
    def save_model(self, path):
        """Save the model to the specified path"""
        self.model.save(path)

# Define the base directory
base_dir = "C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake"

# Create the classifier object
classifier = RealVsFakeClassifier(base_dir)

# Train the model
classifier.train(epochs=10)

# Evaluate the model
classifier.evaluate()

# Predict on test images
test_image_paths = [
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00016.jpg',
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/fake/00F8LKY6JC.jpg',
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00001.jpg',
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/fake/00MZYXAT77.jpg',
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00001.jpg',
    'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00099.jpg'
]

for path in test_image_paths:
    classifier.predict(path)

# Save the model
classifier.save_model("C:/Users/Lenovo/Desktop/archive/real_vs_fake/save_model/model.h5")
classifier.save_model('my_model.keras')
