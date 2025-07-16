from PIL import Image
import torch
from torchvision import transforms
from utils import get_default_transform, get_torch_model
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
model.load_state_dict(torch.load("models/saved/best_cnn_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    output = model(img_tensor)
    _, predicted_class = torch.max(output, 1)

print("Predicted class ID:", predicted_class.item())


# id_to_label = {
#     0: "Speed limit (20km/h)",
#     1: "Speed limit (30km/h)",
#     2: "Speed limit (50km/h)",
#     # and so on
# }
# print("Predicted label:", id_to_label[predicted_class.item()])
