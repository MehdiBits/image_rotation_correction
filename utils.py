import torch
from PIL import Image

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer=None, device='cpu'):
    # map_location argument allows to switch between cpu-gpu inference models
    checkpoint = torch.load(filename, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Checkpoint loaded: {filename}")

def rotate_image_symmetry(image, angle):
    """
    Rotates an image randomly while maintaining symmetry by extending it through mirroring.

    Args:
        image (PIL.Image): Input image.
        angle (float): Rotation angle in degrees.

    Returns:
        tuple: (Rotated PIL.Image, float angle used for rotation)
    """
    width, height = image.size

    extended_image = Image.new("RGB", (width * 3, height * 3))
    extended_image.paste(image, (width, height))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (0, height))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT), (width * 2, height))
    extended_image.paste(image.transpose(Image.FLIP_TOP_BOTTOM), (width, 0))
    extended_image.paste(image.transpose(Image.FLIP_TOP_BOTTOM), (width, height * 2))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (0, 0))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (width * 2, 0))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (0, height * 2))
    extended_image.paste(image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM), (width * 2, height * 2))

    rotated_image = extended_image.rotate(angle, resample=Image.BICUBIC, center=(width * 3 // 2, height * 3 // 2))
    rotated_image = rotated_image.crop((width, height, width * 2, height * 2))

    return rotated_image