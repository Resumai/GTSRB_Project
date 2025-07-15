from torchvision import transforms
from configs import cnn_config as cfg


# Settings for GTSRB Image Loader
def get_default_transform():
    transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE), # Resize heigh and width, duh
        transforms.ToTensor(),  # PIL → Tensor. Scales rgb values to [0,1]. Changes from HWC to CHW. CHW divides into channels from 2d array to 3d.
        transforms.Normalize(cfg.NORMALIZE_MEAN, cfg.NORMALIZE_STD)  # Normalization. Center to [-1,1] (adjust if needed)
    ])
    return transform
