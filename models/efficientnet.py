import torch
import torch.nn as nn
import timm

class RotationEfficientNet(nn.Module):
    def __init__(self):
        super(RotationEfficientNet, self).__init__()
        self.model = timm.create_model("efficientnet_b3a", pretrained=True)
        
        # Change output layer: Predict two values (cosθ, sinθ)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)

    def forward(self, x):
        return self.model(x)
