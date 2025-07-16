# from typing import TYPE_CHECKING
from models.simple_cnn import SimpleCNN
from models.resnet18 import build_resnet18
import torch.nn as nn

# if TYPE_CHECKING:

def get_torch_model(name: str) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN()
    elif name == "resnet18":
        return build_resnet18()
    else:
        raise ValueError(f"Unknown model name: {name}")
