import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from image_rotation_correction.config import DEVICE, CHECKPOINT_PATH, LOAD_MODEL
from image_rotation_correction.models.efficientnet import RotationEfficientNet
from image_rotation_correction.utils import load_checkpoint

class RotationEfficientNetSingleton:
    """
    Singleton class to load and store a single instance of RotationEfficientNet. This prevents loading multiple time the model.
    """
    _instance = None

    def __new__(cls, device=DEVICE, load_model=True):
        """
        Creates or retrieves the singleton instance of RotationEfficientNet.

        Args:
            device (str, optional): The device to load the model on ("cuda" or "cpu"). Defaults to DEVICE.

        Returns:
            RotationEfficientNetSingleton: The singleton instance of the RotationEfficientNet model.
        """
        if cls._instance is None:
            print(f"Loading RotationEfficientNet model on {device}...")
            cls._instance = super(RotationEfficientNetSingleton, cls).__new__(cls)
            cls._instance.model = RotationEfficientNet().to(device)
            if LOAD_MODEL and load_model:
                load_checkpoint(CHECKPOINT_PATH, cls._instance.model)
                print("Model loaded from checkpoint.")
            cls._instance.device = device
            cls._instance.model.eval()
        return cls._instance

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_rotation(image, device=DEVICE):
    """
    Predicts the rotation angle of an image using the trained neural network.

    Args:
        image (PIL.Image): The input image to predict the rotation for.
        device (str, optional): The device to perform computation on ("cuda" or "cpu"). Defaults to DEVICE.

    Returns:
        float: The predicted rotation angle in degrees (0-360°).
    """
    image = transform(image).unsqueeze(0).to(device)  # Apply transforms and add batch dim
    instance_model = RotationEfficientNetSingleton(device)
    with torch.no_grad():
        output = instance_model.model(image)

    sin_val, cos_val = output.cpu().numpy().flatten()
    predicted_angle = np.arctan2(sin_val, cos_val) * (180 / np.pi)

    return predicted_angle % 360


def predict_rotation_batch(data_loader, device=DEVICE, verbose=False):
    """
    Predicts the rotation angles for a batch of images.

    Args:
        data_loader (torch.utils.data.DataLoader): A DataLoader object containing image batches.
        device (str, optional): The device to perform computation on ("cuda" or "cpu"). Defaults to DEVICE.
        verbose (bool, optional): If True, prints the predicted rotation angles. Defaults to False.

    Returns:
        list of tuples: A list containing tuples where each tuple is (image_name, predicted_angle).
    """
    results = []
    instance_model = RotationEfficientNetSingleton(device)
    with torch.no_grad():
        for images, image_names in data_loader:
            images = images.to(device)
            outputs = instance_model.model(images)

            for i, output in enumerate(outputs):
                sin_val, cos_val = output.cpu().numpy()
                predicted_angle = np.arctan2(sin_val, cos_val) * (180 / np.pi)
                predicted_angle = predicted_angle % 360
                results.append((image_names[i], predicted_angle))
                if verbose:
                    print(f"Predicted Rotation Angle for {image_names[i]}: {predicted_angle:.2f}°")
    return results
