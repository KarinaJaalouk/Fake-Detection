import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

class ImageGameApp:
    def __init__(self, root, image_folder, model_path):
        self.root = root
        self.root.title("Real vs Fake Images")
        self.image_folder = image_folder
        self.model = load_model(model_path)
        self.class_names = ['fake', 'real']

        self.current_round = 1
        self.player_score = 0
        self.ai_score = 0
        self.image1_path = ""
        self.image2_path = ""

        self.create_intro_screen()

    def create_intro_screen(self):
        self.intro_frame = Frame(self.root, padx=20, pady=20)
        self.intro_frame.grid(row=0, column=0, padx=20, pady=20)

        Label(self.intro_frame, text="Ready to compete with AI?", font=("Helvetica", 16)).grid(row=0, column=0, pady=20)
        Button(self.intro_frame, text="Compete", command=self.start_game).grid(row=1, column=0, pady=10)

    def start_game(self):
        self.intro_frame.grid_forget()
        self.create_widgets()
        self.load_initial_images()

    def create_widgets(self):
        self.frame1 = Frame(self.root, padx=10, pady=10)
        self.frame1.grid(row=0, column=0, padx=10, pady=10)

        self.frame2 = Frame(self.root, padx=10, pady=10)
        self.frame2.grid(row=0, column=1, padx=10, pady=10)

        self.round_label = tk.Label(self.root, text=f"Round {self.current_round}", font=("Helvetica", 16))
        self.round_label.grid(row=1, column=0, columnspan=2)

        self.score_label = tk.Label(self.root, text=f"Player Score: {self.player_score}/10 | AI Score: {self.ai_score}/10", font=("Helvetica", 16))
        self.score_label.grid(row=2, column=0, columnspan=2)

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
        self.score_label.config(text=f"Player Score: {self.player_score}/10 | AI Score: {self.ai_score}/10")

    def load_initial_images(self):
        real_image_path, fake_image_path = self.choose_random_images()
        self.real_image, self.real_image_path = self.load_image(real_image_path)
        self.fake_image, self.fake_image_path = self.load_image(fake_image_path)

        self.image1_path = self.real_image_path
        self.image2_path = self.fake_image_path

        self.show_images()

    def show_images(self):
        if random.choice([True, False]):
            self.image1_label = Label(self.frame1, image=self.real_image)
            self.image2_label = Label(self.frame2, image=self.fake_image)
            self.image1_path = self.real_image_path
            self.image2_path = self.fake_image_path
        else:
            self.image1_label = Label(self.frame1, image=self.fake_image)
            self.image2_label = Label(self.frame2, image=self.real_image)
            self.image1_path = self.fake_image_path
            self.image2_path = self.real_image_path

        self.image1_label.pack()
        self.image2_label.pack()

        Button(self.frame1, text="This is real", command=self.button_action_is_real_picture1).pack(pady=5)
        Button(self.frame2, text="This is real", command=self.button_action_is_real_picture2).pack(pady=5)

    def button_action_is_real_picture1(self):
        print(f"Image 1 Path: {self.image1_path}")
        if self.image1_path == self.real_image_path:
            self.player_score += 1
            print("This is real button 1")
        else:
            print("This is not real button 1")
        self.update_score_label()
        self.check_game_status()

    def button_action_is_real_picture2(self):
        print(f"Image 2 Path: {self.image2_path}")
        if self.image2_path == self.real_image_path:
            self.player_score += 1
            print("This is real button 2")
        else:
            print("This is not real button 2")
        self.update_score_label()
        self.check_game_status()

    def make_prediction(self, img_path):
        test_image = image.load_img(img_path, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.0

        result = self.model.predict(test_image)
        predicted_class = np.argmax(result, axis=1)
        return self.class_names[predicted_class[0]]

    def load_new_round(self):
        self.current_round += 1
        real_image_path, fake_image_path = self.choose_random_images()
        self.real_image, self.real_image_path = self.load_image(real_image_path)
        self.fake_image, self.fake_image_path = self.load_image(fake_image_path)

        # Predict both images
        prediction1 = self.make_prediction(self.real_image_path)
        prediction2 = self.make_prediction(self.fake_image_path)
        print("__________________________________________________________________________")
        print(f"Image 1 Path: {self.real_image_path}, Predicted Class: {prediction1}")
        print(f"Image 2 Path: {self.fake_image_path}, Predicted Class: {prediction2}")

        # Update AI score
        if prediction1 == 'real':
            self.ai_score += 1
        if prediction2 == 'real':
            self.ai_score += 1

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

    def check_game_status(self):
        if self.current_round < 10:
            self.load_new_round()
        else:
            self.display_final_score()

    def display_final_score(self):
        self.frame1.grid_remove()
        self.frame2.grid_remove()
        self.round_label.grid_remove()
        self.score_label.grid_remove()

        final_score_label = tk.Label(self.root, text=f"Game Over. Player Score: {self.player_score}, AI Score: {self.ai_score}", font=("Helvetica", 20))
        final_score_label.grid(row=0, column=0, padx=100, pady=100)

if __name__ == "__main__":
    image_folder = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test'
    model_path = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/save_model/model.h5'

    root = tk.Tk()
    app = ImageGameApp(root, image_folder, model_path)
    root.mainloop()
