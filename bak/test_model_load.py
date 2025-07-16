import torch

from torchvision.utils import make_grid
import matplotlib.pyplot as plt



def test_model_load(model, val_loader, device):
    # Take a small batch from val_loader
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)

    # Predict
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Show the first 5 predictions
    print("Predicted:", predicted[:5].cpu().numpy())
    print("Actual:   ", labels[:5].cpu().numpy())
