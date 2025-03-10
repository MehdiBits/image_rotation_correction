import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from models.efficientnet import RotationEfficientNet
from utils import load_checkpoint
import config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device is set to : {device}')

# Load model
model = RotationEfficientNet().to(device)
load_checkpoint(config.CHECKPOINT_PATH, model)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def list_files_in_directory(directory_path):
    """List all files in a given directory."""
    return [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]

def apply_gradual_rotation(image, max_angle=40):
    """
    Apply gradual rotations from -max_angle to max_angle.

    Returns:
    - List of tuples (rotated_image, applied_angle)
    """
    image_stack = []
    for angle in range(-max_angle, max_angle):
        rotated_image = image.rotate(angle)  # Rotate using PIL
        image_stack.append((rotated_image, angle))

    return image_stack

def predict_rotation(image):
    """
    Predict the rotation of an image using the trained neural network.
    """
    image = transform(image).unsqueeze(0).to(device)  # Apply transforms and add batch dim

    with torch.no_grad():
        output = model(image)

    # Convert sin/cos to angle
    sin_val, cos_val = output.cpu().numpy().flatten()
    predicted_angle = np.arctan2(sin_val, cos_val) * (180 / np.pi)
    # Ensure angle is in range [0, 360]
    return predicted_angle % 360

def test_rotation_gradual(input_dir, max_angle=20, verbose=False):
    """
    Apply gradual rotations to images and estimate their angles using the trained NN.
    
    Returns:
    - pd.DataFrame with columns: "filename", "applied angle", "predicted angle".
    """
    processed_data = []
    
    for path in list_files_in_directory(input_dir):
        image = Image.open(path).convert("RGB")  # Open image with PIL

        if verbose:
            print(f'Processing image {os.path.basename(path)}')

        rotated_images = apply_gradual_rotation(image, max_angle=max_angle)

        for rotated_image, angle in rotated_images:
            predicted_angle = predict_rotation(rotated_image)
            processed_data.append({
                "filename": os.path.basename(path),
                "applied angle": angle,
                "predicted angle": predicted_angle
            })

    return pd.DataFrame(processed_data)

if __name__ == "__main__":
    input_dir = os.path.join('data', 'test')
    output_dir = os.path.join('data', 'test')

    df = test_rotation_gradual(input_dir, verbose=True)
    df.to_csv(os.path.join(output_dir, 'results_test_nn.csv'), index=False)

    print(f'Results saved to {os.path.join(output_dir, "results_test_nn.csv")}')
