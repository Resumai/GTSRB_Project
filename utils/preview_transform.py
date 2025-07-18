from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from image_transforms import get_augmented_transform, get_default_transform

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Path to one sample image
image_path = "data/GTSRB/Training/00014/00014_00000.ppm"  # Adjust as needed

# Load and apply transform
img = Image.open(image_path).convert("RGB")
transform = get_augmented_transform()
transformed_tensor = transform(img)

# Convert back to PIL for display (undo Normalize)
unnormalize = transforms.Normalize(
    mean=[-m / s for m, s in zip((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    std=[1 / s for s in (0.5, 0.5, 0.5)]
)
img_for_display = unnormalize(transformed_tensor).clamp(0, 1)  # Clamp to [0, 1] range

# Display using matplotlib
plt.imshow(img_for_display.permute(1, 2, 0))  # [C, H, W] -> [H, W, C]
plt.title("Transformed Image")
plt.axis("off")
plt.show()
