import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms
from utils import get_default_transform, get_torch_model
from configs import cnn_config as cfg

app = Flask(__name__)
UPLOAD_FOLDER = "web_ui/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_torch_model(cfg.MODEL_NAME).to(device)
model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
model.eval()

transform = get_default_transform()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Load + transform image
            img = Image.open(filepath).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                _, pred = torch.max(output, 1)
                prediction = pred.item()

            image_url = f"/static/uploads/{file.filename}"

    return render_template("index.html", prediction=prediction, image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True)