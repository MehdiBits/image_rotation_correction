import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from models.efficientnet import RotationEfficientNet
from utils import load_checkpoint
import config
import os

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

def predict_rotation(image_path):
    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
    
    # Convert sin/cos to angle
    sin_val, cos_val = output.cpu().numpy().flatten()
    predicted_angle = np.arctan2(sin_val, cos_val) * (180 / np.pi)
    
    # Ensure angle is in range [0, 360]
    predicted_angle = predicted_angle % 360

    print(f"Predicted Rotation Angle: {predicted_angle:.2f}°")

    return predicted_angle

def predict_applied_rotation(image_path, angle):
    image = Image.open(image_path).convert("RGB")
    image = image.rotate(angle)
    image = transform(image).unsqueeze(0).to(device)
    

    with torch.no_grad():
        output = model(image)
    
    # Convert sin/cos to angle
    sin_val, cos_val = output.cpu().numpy().flatten()
    predicted_angle = np.arctan2(sin_val, cos_val) * (180 / np.pi)
    
    # Ensure angle is in range [0, 360]
    predicted_angle = predicted_angle % 360

    print(f"Predicted Rotation Angle: {predicted_angle:.2f}°")

    return predicted_angle

# Test on an image
image_name = '38318119_85ff9cd53c_29_73293249@N00.jpg'
image_path = os.path.join('data', 'train', image_name)
predict_rotation(image_path=image_path)
