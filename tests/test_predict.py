import pytest
import torch
from PIL import Image

from image_rotation_correction.predict import predict_rotation, predict_rotation_batch, RotationEfficientNetSingleton

def dummy_image(size=(224, 224), color=(255, 255, 255)):
    return Image.new("RGB", size, color)


def test_rotation_singleton_is_single_instance():
    # Instantiate the singleton twice
    instance1 = RotationEfficientNetSingleton(device="cpu", load_model=False)
    instance2 = RotationEfficientNetSingleton(device="cpu", load_model=False)
    
    # Check that both instances are the same (singleton behavior)
    assert instance1 is instance2, "RotationEfficientNetSingleton should return the same instance every time"


def test_predict_rotation_output_type_and_range():
    img = dummy_image()
    angle = predict_rotation(img, device="cpu")
    
    assert isinstance(angle, (float, int)), "Angle should be a number"
    assert 0.0 <= angle < 360.0, "Angle should be in range [0, 360)"

def test_predict_rotation_batch_output_structure():
    class DummyLoader:
        def __iter__(self):
            # Simulate a single batch with two images and names
            batch = (torch.randn(2, 3, 224, 224), ["img1.jpg", "img2.jpg"])
            return iter([batch])

    results = predict_rotation_batch(DummyLoader(), device="cpu")

    assert isinstance(results, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in results)
    assert all(isinstance(name, str) for name, _ in results)
    assert all(isinstance(angle, (float, int)) and 0.0 <= angle < 360.0 for _, angle in results)
