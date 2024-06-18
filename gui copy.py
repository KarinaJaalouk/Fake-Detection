import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import os
import random

# Path to the folder containing images
image_folder = 'C:/Users/Lenovo/Desktop/archive/real_vs_fake/real-vs-fake/test'

# Function to load and preprocess image
def load_image(path, size=(150, 150)):
    img = Image.open(path)
    img = img.resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(img), path

# Function to choose a random real and fake image
def choose_random_images():
    real_images = os.listdir(os.path.join(image_folder, 'real'))
    fake_images = os.listdir(os.path.join(image_folder, 'fake'))
    
    real_image = random.choice(real_images)
    fake_image = random.choice(fake_images)
    
    return os.path.join(image_folder, 'real', real_image), os.path.join(image_folder, 'fake', fake_image)

# Function to update score label
def update_score_label():
    score_label.config(text=f"Score: {score}/10")

# Create the main window
root = tk.Tk()
root.title("Real vs Fake Images")

# Create frames for images and buttons
frame1 = Frame(root, padx=10, pady=10)
frame1.grid(row=0, column=0, padx=10, pady=10)

frame2 = Frame(root, padx=10, pady=10)
frame2.grid(row=0, column=1, padx=10, pady=10)

# Initialize variables
current_round = 1
score = 0

# Define global variables for image paths
image1_path = ""
image2_path = ""

# Load initial images
real_image_path, fake_image_path = choose_random_images()
real_image, real_image_path = load_image(real_image_path)
fake_image, fake_image_path = load_image(fake_image_path)

image1_path = real_image_path
image2_path = fake_image_path

def button_action_is_real_picture1():
    global current_round, score
    print(image1_path)
    if 'test\\real' in image1_path:  # Используем 'test\\real' вместо r'test\real'
        print("Correct: Real image button1")
        score += 1
    else:
        print("Incorrect: Fake image button1")
    
    update_score_label()  # Обновляем метку счета
    print(f"Round {current_round} Score: {score}/10")
    
    if current_round < 10:
        load_new_round()
    else:
        print("Game Over. Final Score:", score)
        root.destroy()

def button_action_is_real_picture2():
    global current_round, score
    print(image2_path)
    if 'test\\real' in image2_path:  # Используем 'test\\real' вместо r'test\real'
        print("Correct: Real image button2")
        score += 1
    else:
        print("Incorrect: Fake image button2")
    
    update_score_label()  # Обновляем метку счета
    print(f"Round {current_round} Score: {score}/10")
    
    if current_round < 10:
        load_new_round()
    else:
        print("Game Over. Final Score:", score)
        root.destroy()

# Function to load new images for the next round
def load_new_round():
    global real_image, fake_image, real_image_path, fake_image_path, current_round
    global image1_path, image2_path

    # Load new images
    real_image_path, fake_image_path = choose_random_images()
    real_image, real_image_path = load_image(real_image_path)
    fake_image, fake_image_path = load_image(fake_image_path)
    
    if random.choice([True, False]):
        image1_label.config(image=real_image)
        image2_label.config(image=fake_image)
        image1_path = real_image_path
        image2_path = fake_image_path
    else:
        image1_label.config(image=fake_image)
        image2_label.config(image=real_image)
        image1_path = fake_image_path
        image2_path = real_image_path
    
    round_label.config(text=f"Round {current_round}")
    update_score_label()  # Обновляем метку счета

    current_round += 1  # Увеличиваем номер текущего раунда

    if current_round > 10:
        print("Game Over. Final Score:", score)
        root.destroy()

# Add labels for images
Label(frame1, text="Image 1").pack()
image1_label = Label(frame1, image=real_image)
image1_label.pack()
Button(frame1, text="This is real", command=button_action_is_real_picture1).pack(pady=5)

Label(frame2, text="Image 2").pack()
image2_label = Label(frame2, image=fake_image)
image2_label.pack()
Button(frame2, text="This is real", command=button_action_is_real_picture2).pack(pady=5)

# Label for round number
round_label = tk.Label(root, text=f"Round {current_round}", font=("Helvetica", 16))
round_label.grid(row=1, column=0, columnspan=2)

score_label = tk.Label(root, text=f"Score: {score}/10", font=("Helvetica", 16))
score_label.grid(row=2, column=0, columnspan=2)

# Load initial round
load_new_round()

# Run the application
root.mainloop()
