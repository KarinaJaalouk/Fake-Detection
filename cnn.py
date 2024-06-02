import os
import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from PIL import Image
import matplotlib.pyplot as plt
import pydotplus
import pydot

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

base_dir = "C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

# Проверяем, существуют ли директории
if os.path.exists(train_dir) and os.path.exists(test_dir) and os.path.exists(valid_dir):
    print("Все директории существуют.")
else:
    print("Одна или несколько директорий не существуют.")
    if not os.path.exists(train_dir):
        print(f"Папка {train_dir} не существует.")
    if not os.path.exists(test_dir):
        print(f"Папка {test_dir} не существует.")
    if not os.path.exists(valid_dir):
        print(f"Папка {valid_dir} не существует.")

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir, batch_size=100, class_mode='binary', target_size=(150, 150))
validation_generator = valid_datagen.flow_from_directory(
    valid_dir, batch_size=100, class_mode='binary', target_size=(150, 150))
test_generator = test_datagen.flow_from_directory(
    test_dir, batch_size=100, class_mode='binary', target_size=(150, 150))

# Визуализация первых 9 изображений из генератора данных
plt.figure(figsize=(10, 10))
for i in range(9):
    img, label = next(train_generator)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(img[0])
    plt.title("Fake" if label[0] == 0.0 else "Real")
    plt.axis("off")
plt.show()

model = tf.keras.models.Sequential([
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

tf.keras.utils.pydot = pydot

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=validation_generator, epochs=10, validation_steps=50, verbose=1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

class_names = ['fake', 'real']

import numpy as np
from tensorflow.keras.preprocessing import image

# Замена пути на ваш тестовый файл изображения
test_image_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/fake/0C28G3DC1O.jpg'
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
print(
    "This image is {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)

# Замена пути на ваш тестовый файл изображения
test_image_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/fake/00F8LKY6JC.jpg'
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
print(
    "This image is {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)

# Замена пути на ваш тестовый файл изображения
test_image_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00001.jpg'
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
print(
    "This image is {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)

# Замена пути на ваш тестовый файл изображения
test_image_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/fake/00MZYXAT77.jpg'
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
print(
    "This image is {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)

# Замена пути на ваш тестовый файл изображения
test_image_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00001.jpg'
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
print(
    "This image is {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)

# Замена пути на ваш тестовый файл изображения
test_image_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00099.jpg'
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
print(
    "This image is {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)

model.save("C:/Users/Lenovo/Desktop/archive/real_vs_fake/save_model/model.h5")
model.save('my_model.keras')