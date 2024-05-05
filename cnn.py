import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from PIL import Image

# Function to load and preprocess the dataset
def load_dataset(data_dir, image_size):
    images = []
    labels = []
    
    label_encoder = LabelEncoder()
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            image = Image.open(image_path).convert("RGB")
            image = image.resize(image_size)
            image = np.array(image) / 255.0  # Normalize pixel values
            images.append(image)
            labels.append(label)
    
    labels = label_encoder.fit_transform(labels)  # Convert labels to numerical values
    return np.array(images), np.array(labels)

# Define model architecture
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output layer, binary classification
    ])
    return model

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_images, train_labels, val_images, val_labels, epochs, batch_size):
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_images, val_labels))
    return history

# Evaluate the model
def evaluate_model(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels)
    return loss, accuracy

# Perform inference
def predict_image(model, image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    if prediction > 0.5:
        return "Fake"
    else:
        return "Real"


data_dir = "test"
image_size = (128, 128)
images, labels = load_dataset(data_dir, image_size)

# Split the dataset into training, validation, and test sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Define model architecture
model = create_model(input_shape=train_images[0].shape)

# Compile the model
model = compile_model(model)

# Train the model
history = train_model(model, train_images, train_labels, val_images, val_labels, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = evaluate_model(model, test_images, test_labels)
print("Test Accuracy:", accuracy)

data_dir = "test"
image_path = "test/fake/ZW468TZZF2.jpg"
prediction = predict_image(model, image_path, image_size)
print("Prediction:", prediction)
