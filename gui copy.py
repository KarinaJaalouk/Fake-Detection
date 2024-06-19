import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np

class ImageGameApp:
    def __init__(self, root, image_folder):
        self.root = root
        self.root.title("Real vs Fake Images")
        self.image_folder = image_folder

        self.frame1 = Frame(self.root, padx=10, pady=10)
        self.frame1.grid(row=0, column=0, padx=10, pady=10)

        self.frame2 = Frame(self.root, padx=10, pady=10)
        self.frame2.grid(row=0, column=1, padx=10, pady=10)

        self.current_round = 1
        self.score = 0
        self.image1_path = ""
        self.image2_path = ""

        self.load_initial_images()

        self.create_widgets()

    def load_image(self, path, size=(150, 150)):
        img = Image.open(path)
        img = img.resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(img), path

    def choose_random_images(self):
        real_images = os.listdir(os.path.join(self.image_folder, 'real'))
        fake_images = os.listdir(os.path.join(self.image_folder, 'fake'))
        
        real_image = random.choice(real_images)
        fake_image = random.choice(fake_images)
        
        return os.path.join(self.image_folder, 'real', real_image), os.path.join(self.image_folder, 'fake', fake_image)

    def update_score_label(self):
        self.score_label.config(text=f"Score: {self.score}/10")

    def load_initial_images(self):
        real_image_path, fake_image_path = self.choose_random_images()
        self.real_image, self.real_image_path = self.load_image(real_image_path)
        self.fake_image, self.fake_image_path = self.load_image(fake_image_path)

        self.image1_path = self.real_image_path
        self.image2_path = self.fake_image_path

    def button_action_is_real_picture1(self):
        print(self.image1_path)
        if 'test\\real' in self.image1_path:  # Используем 'test\\real' вместо r'test\real'
            print("Correct: Real image button1")
            self.score += 1
        else:
            print("Incorrect: Fake image button1")
        
        self.update_score_label()
        print(f"Round {self.current_round} Score: {self.score}/10")
        
        if self.current_round < 10:
            self.load_new_round()
        else:
            self.display_final_score()

    def button_action_is_real_picture2(self):
        print(self.image2_path)
        if 'test\\real' in self.image2_path:  # Используем 'test\\real' вместо r'test\real'
            print("Correct: Real image button2")
            self.score += 1
        else:
            print("Incorrect: Fake image button2")
        
        self.update_score_label()
        print(f"Round {self.current_round} Score: {self.score}/10")
        
        if self.current_round < 10:
            self.load_new_round()
        else:
            self.display_final_score()

    def load_new_round(self):
        self.current_round += 1
        real_image_path, fake_image_path = self.choose_random_images()
        self.real_image, self.real_image_path = self.load_image(real_image_path)
        self.fake_image, self.fake_image_path = self.load_image(fake_image_path)
        
        if random.choice([True, False]):
            self.image1_label.config(image=self.real_image)
            self.image2_label.config(image=self.fake_image)
            self.image1_path = self.real_image_path
            self.image2_path = self.fake_image_path
        else:
            self.image1_label.config(image=self.fake_image)
            self.image2_label.config(image=self.real_image)
            self.image1_path = self.fake_image_path
            self.image2_path = self.real_image_path
        
        self.round_label.config(text=f"Round {self.current_round}")
        self.update_score_label()

        # Check if it's the last round
        if self.current_round > 10:
            self.display_final_score()

    def display_final_score(self):
        # Hide unnecessary widgets
        self.frame1.grid_remove()
        self.frame2.grid_remove()
        self.round_label.grid_remove()
        self.score_label.grid_remove()

        # Display final score
        final_score_label = tk.Label(self.root, text=f"Game Over. Final Score: {self.score}", font=("Helvetica", 20))
        final_score_label.grid(row=0, column=0, padx=100, pady=100)

    def create_widgets(self):
        Label(self.frame1, text="Image 1").pack()
        self.image1_label = Label(self.frame1, image=self.real_image)
        self.image1_label.pack()
        Button(self.frame1, text="This is real", command=self.button_action_is_real_picture1).pack(pady=5)

        Label(self.frame2, text="Image 2").pack()
        self.image2_label = Label(self.frame2, image=self.fake_image)
        self.image2_label.pack()
        Button(self.frame2, text="This is real", command=self.button_action_is_real_picture2).pack(pady=5)

        self.round_label = tk.Label(self.root, text=f"Round {self.current_round}", font=("Helvetica", 16))
        self.round_label.grid(row=1, column=0, columnspan=2)

        self.score_label = tk.Label(self.root, text=f"Score: {self.score}/10", font=("Helvetica", 16))
        self.score_label.grid(row=2, column=0, columnspan=2)

if __name__ == "__main__":
    # Path to the folder containing images
    image_folder = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test'

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
    img_path = '1.jpg'

    predicted_class, prediction = make_prediction(img_path, model)
    predicted_label = class_names[predicted_class[0]]

    print(f"Predicted Class: {predicted_label}")
    root = tk.Tk()
    app = ImageGameApp(root, image_folder)
    root.mainloop()
