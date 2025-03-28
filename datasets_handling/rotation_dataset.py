import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import config

class RotationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, img) 
            for img in os.listdir(image_dir) 
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Randomly rotate the image
        angle = random.randint(-config.MAX_ANGLE, config.MAX_ANGLE)
        rotated_image = image.rotate(angle)

        # Apply transforms
        if self.transform:
            rotated_image = self.transform(rotated_image)

        # Convert angle to sin/cos representation
        angle_rad = np.deg2rad(angle)  # Convert degrees to radians
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)

        # Create a tensor of shape (2,)
        angle_tensor = torch.tensor([angle_sin, angle_cos], dtype=torch.float32)

        return rotated_image, angle_tensor  # Now (batch_size, 2)

class ImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]
