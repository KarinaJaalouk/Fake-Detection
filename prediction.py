import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

model_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/save_model/model.h5' 
model = load_model(model_path)

def make_prediction(img_path, model):
    test_image = image.load_img(img_path, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.0  # Normalizing Images
    
    result = model.predict(test_image)
    
    predicted_class = np.argmax(result, axis=1)
    return predicted_class, result

class_names = ['fake', 'real']

# Замена пути на ваш тестовый файл изображения
test_image_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test/real/00016.jpg'
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
print(
    "This image is {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)

class_names = ['fake', 'real']

# Замена пути на ваш тестовый файл изображения
test_image_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/valid/real/00008.jpg'
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
test_image_path = '222.jpg'
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
test_image_path = '2222.jpg'
test_image = image.load_img(test_image_path, target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

print(result)
print(
    "This image is {} with a {:.2f}% confidence."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)


# img_path = '1.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class: {predicted_label}")

# img_path = '2.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class: {predicted_label}")

# img_path = 'raquel.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class Raquel: {predicted_label}")

# img_path = 'adrian.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class Adrian: {predicted_label}")

# img_path = 'karina.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class Karina: {predicted_label}")

# img_path = 'jorge.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class Jorge: {predicted_label}")

# img_path = 'gorge.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class REAL Gorge: {predicted_label}")

# img_path = '777.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class 7: {predicted_label}")

# img_path = '888.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class 7: {predicted_label}")

# img_path = '999.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class 7: {predicted_label}")


# img_path = 'dada.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class dada: {predicted_label}")

# img_path = 'nono.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class nono: {predicted_label}")

# img_path = 'net.jpg'  
# class_names = ['fake', 'real']  

# predicted_class, prediction = make_prediction(img_path, model)
# predicted_label = class_names[predicted_class[0]]

# print(f"Predicted Class net: {predicted_label}")