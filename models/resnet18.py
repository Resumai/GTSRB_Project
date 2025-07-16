import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_resnet18(num_classes: int = 43, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
