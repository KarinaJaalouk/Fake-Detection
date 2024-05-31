import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image, ImageFilter
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import os

# Define the path to the dataset
data_path = 'data'

# List all files in the dataset directory
image_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

# Create a DataFrame with paths and labels
data = []
for file in image_files:
    label = 'real' if 'real' in file else 'fake'
    data.append((os.path.join(data_path, file), label))

train_dataset = pd.DataFrame(data, columns=['path', 'label_str'])

# Check if the DataFrame is created correctly
print(train_dataset.head())

# Count the number of real and fake images
label_counts = train_dataset['label_str'].value_counts()

# Plot the counts
plt.figure(figsize=(10, 6))
plt.bar(label_counts.index, label_counts.values, color=['blue', 'orange'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Dataset distribution')
plt.show()

# Save the plot to a file if it doesn't display in your environment
plt.savefig('dataset_distribution.png')
