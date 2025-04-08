import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


from image_rotation_correction.utils import rotate_image_symmetry
import image_rotation_correction.config as config

class RotationDataset(Dataset):
    """
    A PyTorch Dataset class for creating a rotated image datasets.

    Args:
        image_dir (str): Path to the directory containing image files.
        transform (callable, optional): A function/transform to apply to the images. Default is None.
        rotation_type (str, optional): Type of rotation to apply. 
            - 'blank': Applies a standard rotation.
            - 'sym': Applies a rotation with symmetry handling. Default is 'blank'.

    Attributes:
        image_paths (list): List of file paths to the images in the specified directory.
        transform (callable): Transform function to apply to the images.
        rotation_type (str): Type of rotation to apply.

    Methods:
        __len__():
            Returns the total number of images in the dataset.

        __getitem__(idx):
            Retrieves the image and its corresponding rotation angle tensor at the specified index.

            Args:
                idx (int): Index of the image to retrieve.

            Returns:
                tuple: A tuple containing:
                    - rotated_image (PIL.Image.Image): The rotated image.
                    - angle_tensor (torch.Tensor): A tensor of shape (2,) representing the rotation angle 
                      in sine and cosine form.
    """
    def __init__(self, image_dir, transform=None, rotation_type='blank'):
        self.image_paths = [
            os.path.join(image_dir, img) 
            for img in os.listdir(image_dir) 
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform
        self.rotation_type = rotation_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Randomly rotate the image
        angle = random.randint(-config.MAX_ANGLE, config.MAX_ANGLE)
        if self.rotation_type == 'blank':
            rotated_image = image.rotate(angle)
        elif self.rotation_type == 'sym':
            rotated_image = rotate_image_symmetry(image, angle)

        # Apply transforms, used to fit to model input size
        if self.transform:
            rotated_image = self.transform(rotated_image)

        # Convert angle to sin/cos representation
        angle_rad = np.deg2rad(angle)
        angle_sin = np.sin(angle_rad)
        angle_cos = np.cos(angle_rad)

        # Create a tensor of shape (2,)
        angle_tensor = torch.tensor([angle_sin, angle_cos], dtype=torch.float32)

        return rotated_image, angle_tensor  # Now (batch_size, 2)

class ImageDataset(Dataset):
    """
    A standard PyTorch Dataset class for loading images without rotation.

    Args:
        image_dir (str): Path to the directory containing image files.
        transform (callable, optional): A function/transform to apply to the images.

    Attributes:
        image_paths (list): List of file paths to the images in the specified directory.
        transform (callable): Transform function to apply to the images.
    """
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

        if self.transform:
            image = self.transform(image)


        image_name = os.path.basename(img_path)
        return image, image_name