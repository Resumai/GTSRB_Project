import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from configs import cnn_config as cfg

def build_mobilnet_v2():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)

    # Adjust the classifier for 43 output classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, cfg.NUM_CLASSES)
    return model
