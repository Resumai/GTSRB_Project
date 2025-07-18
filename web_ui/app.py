import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms
from utils import get_default_transform, get_torch_model, id_to_label
from configs import cfg_current as cfg

app = Flask(__name__)
UPLOAD_FOLDER = "web_ui/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from models.simple_cnn import SimpleCNN
# model = SimpleCNN().to("cpu")
# model.load_state_dict(torch.load("models/saved/last_cnn_model_run11.pth", map_location="cpu"))
model = get_torch_model(cfg.MODEL_NAME).to(device)
model.load_state_dict(torch.load(cfg.MODEL_SAVE_PATH, map_location=device))
model.eval()

transform = get_default_transform()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None
    label = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Convert .ppm to .png
            if file.filename.lower().endswith(".ppm"):
            # Force .ppm to .png conversion
                img = Image.open(filepath)
                new_filename = file.filename.rsplit(".", 1)[0] + ".png"
                filepath = os.path.join(UPLOAD_FOLDER, new_filename)
                img.save(filepath)
            else:
                img = Image.open(filepath)

            img = img.convert("RGB")
            # img_tensor = transform(img).unsqueeze(0).to(device)
            img_tensor = transform(img).unsqueeze(0).to("cpu")

            with torch.no_grad():
                output = model(img_tensor)
                _, pred = torch.max(output, 1)
                prediction = pred.item()
            label = id_to_label(prediction)
            image_url = f"/static/uploads/{os.path.basename(filepath)}"

            print(image_url)

    return render_template("index.html", prediction=prediction, image_url=image_url, label=label)


if __name__ == "__main__":
    app.run(debug=True)