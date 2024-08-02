from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img)
        noise = np.random.normal(self.mean, self.std, img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255)
        return Image.fromarray(noisy_img.astype(np.uint8))

# Define individual transformations
horizontal_flip = transforms.RandomHorizontalFlip()
vertical_flip = transforms.RandomVerticalFlip()
rotation = transforms.RandomRotation(30)
color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
resized_crop = transforms.RandomResizedCrop(224)
gaussian_noise = GaussianNoise(mean=0, std=25)

def apply_probabilistic_transform(image):
    if random.random() < 0.5:
        image = horizontal_flip(image)
    if random.random() < 0.5:
        image = vertical_flip(image)
    if random.random() < 0.5:
        image = rotation(image)
    if random.random() < 0.5:
        image = color_jitter(image)
    if random.random() < 0.5:
        image = resized_crop(image)
    if random.random() < 0.5:
        image = gaussian_noise(image)
    return image

def is_image_duplicate(image_path, existing_images):
    return os.path.basename(image_path) in existing_images

input_dir = './img'
output_dir = './folder_that_is_Augmented_version_small_testing_dataset'

os.makedirs(output_dir, exist_ok=True)

for label_folder in os.listdir(input_dir):
    label_folder_path = os.path.join(input_dir, label_folder)
    if os.path.isdir(label_folder_path):
        print(f"Processing folder: {label_folder}")  # Debug statement
        output_label_folder_path = os.path.join(output_dir, label_folder)
        os.makedirs(output_label_folder_path, exist_ok=True)
        
        existing_images = set(os.listdir(output_label_folder_path))
        
        for filename in os.listdir(label_folder_path):
            img_path = os.path.join(label_folder_path, filename)
            print(f"Processing image: {filename}")  # Debug statement
            image = Image.open(img_path)
            original_image_path = os.path.join(output_label_folder_path, filename)
            image.save(original_image_path)
            
            existing_images.add(filename)

            for i in range(5):
                augmented_image = apply_probabilistic_transform(image)
                augmented_filename = f'{os.path.splitext(filename)[0]}_aug_{i}.jpeg'
                augmented_image_path = os.path.join(output_label_folder_path, augmented_filename)
                
                if not is_image_duplicate(augmented_filename, existing_images):
                    augmented_image.save(augmented_image_path)
                    existing_images.add(augmented_filename)
                else:
                    print(f"Duplicate found and skipped: {augmented_filename}")
