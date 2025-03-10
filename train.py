import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.efficientnet import RotationEfficientNet
from datasets_handling.rotation_dataset import RotationDataset
from utils import save_checkpoint, load_checkpoint
import config
from tqdm import tqdm  
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = RotationDataset(config.TRAIN_DIR, transform=transform)
val_dataset = RotationDataset(config.VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

# Load model
model = RotationEfficientNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()  # Use MSE for regression
optimizer = optim.AdamW(model.parameters(), lr=config.LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Load checkpoint if resuming training
if config.LOAD_MODEL:
    load_checkpoint(config.CHECKPOINT_PATH, model, optimizer)



def train():
    print("🚀 Starting Training...\n")
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        angle_errors = []  # Store angle errors

        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

        for batch_idx, (images, angles) in loop:
            images, angles = images.to(device), angles.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # outputs -> (batch, 2) -> (sinθ, cosθ)

            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Convert sin/cos to degrees for error tracking
            pred_sin, pred_cos = outputs[:, 0], outputs[:, 1]
            true_sin, true_cos = angles[:, 0], angles[:, 1]

            pred_angles = torch.atan2(pred_sin, pred_cos) * (180 / np.pi)
            true_angles = torch.atan2(true_sin, true_cos) * (180 / np.pi)

            angle_error = torch.abs(true_angles - pred_angles)
            angle_errors.extend(angle_error.cpu().detach().numpy())

            # Update progress bar description
            loop.set_description(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        avg_angle_error = np.mean(angle_errors)  # Mean Absolute Error in degrees
        print(f"📉 Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
        print(f"🔹 Avg Training Angle Error: {avg_angle_error:.2f}°\n")

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_angle_errors = []
        with torch.no_grad():
            for images, angles in val_loader:
                images, angles = images.to(device), angles.to(device)
                outputs = model(images)

                loss = criterion(outputs, angles)
                val_loss += loss.item()

                pred_sin, pred_cos = outputs[:, 0], outputs[:, 1]
                true_sin, true_cos = angles[:, 0], angles[:, 1]

                pred_angles = torch.atan2(pred_sin, pred_cos) * (180 / np.pi)
                true_angles = torch.atan2(true_sin, true_cos) * (180 / np.pi)

                angle_error = torch.abs(true_angles - pred_angles)
                val_angle_errors.extend(angle_error.cpu().detach().numpy())

        avg_val_loss = val_loss / len(val_loader)
        avg_val_angle_error = np.mean(val_angle_errors)
        print(f"📉 Validation Loss: {avg_val_loss:.4f}")
        print(f"🔹 Avg Validation Angle Error: {avg_val_angle_error:.2f}°\n")

        # Save checkpoint
        if epoch % config.SAVE_EVERY == 0 or epoch == config.NUM_EPOCHS - 1:
            save_checkpoint(model, optimizer, config.CHECKPOINT_PATH)

    print("✅ Training Complete!")

if __name__ == "__main__":
    train()
