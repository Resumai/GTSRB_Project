import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def get_augmented_transform():
    augmented_transforms = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.RandomRotation(cfg.RANDOM_ROTATION),
        transforms.RandomAffine(0, translate=(cfg.MAX_WIDTH_SHIFT, cfg.MAX_HEIGHT_SHIFT)),
        transforms.ColorJitter(brightness=cfg.RANDOM_BRIGHTNESS, contrast=cfg.RANDOM_CONTRAST, saturation=cfg.RANDOM_SATURATION),
        transforms.ToTensor(),
        transforms.Normalize(cfg.NORMALIZE_MEAN, cfg.NORMALIZE_STD)
    ])
    return augmented_transforms




# get_augmented_transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.RandomRotation(15),
#     transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.9, 1.1)),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#     transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])
