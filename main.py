import torch
from torch.utils.data import DataLoader
from models.efficientnet import RotationEfficientNet
from utils import load_checkpoint
import config
import os
import argparse
import csv
import numpy as np
from datasets_handling.rotation_dataset import ImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device is set to : {device}')

# Load model
model = RotationEfficientNet().to(device)
load_checkpoint(config.CHECKPOINT_PATH, model)
model.eval()

def predict_rotation_batch(data_loader):
    results = []
    with torch.no_grad():
        for images, image_names in data_loader:
            images = images.to(device)
            outputs = model(images)

            for i, output in enumerate(outputs):
                sin_val, cos_val = output.cpu().numpy()
                predicted_angle = np.arctan2(sin_val, cos_val) * (180 / np.pi)
                predicted_angle = predicted_angle % 360
                results.append((image_names[i], predicted_angle))
                print(f"Predicted Rotation Angle for {image_names[i]}: {predicted_angle:.2f}°")

    return results

def main(input_folder, output_csv):
    dataset = ImageDataset(image_folder=input_folder)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    results = predict_rotation_batch(data_loader)

    # Write results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Predicted Angle"])
        writer.writerows(results)

    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict rotation angles for images in a folder.")
    parser.add_argument('--input', required=True, help='Path to the input image folder')
    parser.add_argument('--output', required=True, help='Path to the output CSV file')

    args = parser.parse_args()

    main(args.input, args.output)
