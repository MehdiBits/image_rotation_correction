import os
import tempfile
import torch
import pytest
from PIL import Image
from image_rotation_correction.utils import save_checkpoint, load_checkpoint, rotate_image_symmetry

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

def test_save_and_load_checkpoint():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Save original weights for comparison
    original_weights = model.linear.weight.clone().detach()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        checkpoint_path = tmp_file.name

    # Save checkpoint
    save_checkpoint(model, optimizer, filename=checkpoint_path)
    assert os.path.exists(checkpoint_path)

    # Modify weights
    with torch.no_grad():
        model.linear.weight += 1.0

    # Load checkpoint
    load_checkpoint(checkpoint_path, model, optimizer=optimizer)

    # Check if weights were restored
    assert torch.allclose(model.linear.weight, original_weights, atol=1e-6)

    os.remove(checkpoint_path)

def test_rotate_image_symmetry():
    # Create a simple 3x3 red square image
    img = Image.new("RGB", (10, 10), color="red")
    angle = 45

    rotated_img = rotate_image_symmetry(img, angle)
    rotated_img_90 = rotate_image_symmetry(img, 90)
    rotated_img_90_2 = rotate_image_symmetry(rotated_img_90, -90)

    assert(img == rotated_img_90_2)  # Check if rotating back gives the original image

    assert isinstance(rotated_img, Image.Image)
    assert rotated_img.size == img.size  # Ensure same size after crop



