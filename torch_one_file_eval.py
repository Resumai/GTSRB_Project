from PIL import Image
import torch
from torchvision import transforms
from utils import get_default_transform, get_torch_model, id_to_label
from configs import cnn_config as cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your image
img = Image.open("data/GTSRB/42.ppm").convert("RGB")

# Apply transform
transform = get_default_transform()
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension [1, 3, 32, 32]


# Send to correct device
img_tensor = img_tensor.to(device)
model = get_torch_model(cfg.MODEL_NAME).to(device)
model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
model.eval()

with torch.no_grad():
    output = model(img_tensor)
    _, predicted_class = torch.max(output, 1)

result_id = predicted_class.item()
print("Predicted class ID:", result_id)
print("Predicted label:", id_to_label(result_id))