import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import csv

from image_rotation_correction.predict import predict_rotation_batch
import image_rotation_correction.config as config
from image_rotation_correction.utils import load_checkpoint
from image_rotation_correction.models.efficientnet import RotationEfficientNet
from image_rotation_correction.datasets_handling.datasets import ImageDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device is set to : {device}')

transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = RotationEfficientNet().to(device)
load_checkpoint(config.CHECKPOINT_PATH, model)
model.eval()


def main(input_folder, output_csv):
    dataset = ImageDataset(image_dir=input_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    results = predict_rotation_batch(data_loader, verbose=True)

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
