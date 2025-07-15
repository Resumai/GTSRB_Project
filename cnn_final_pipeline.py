from torch.utils.data import DataLoader
import torch
from models.simple_cnn import SimpleCNN
from utils.GTSRB_Loader import GTSRBImageLoader
from utils.df_compare import df_compare
import os, csv
import pandas as pd

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_to_csv(filename, class_id, path="output.csv"):
    write_header = not os.path.exists(path)

    with open(path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        if write_header:
            writer.writerow(["Filename", "ClassId"])
        writer.writerow([filename, class_id])


# Load unlabeled test dataset
test_folder = "data/GTSRB/Final_Test"
final_dataset = GTSRBImageLoader(test_folder, unlabeled=True)
final_loader = DataLoader(final_dataset, batch_size=32, shuffle=False)

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("best_cnn_model.pth", map_location=device))
model.eval()

# Predict

predictions = []
with torch.no_grad():
    for images, paths in final_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for path, pred in zip(paths, predicted):
            file_name = os.path.basename(path)
            pred_item = pred.item()
            predictions.append((file_name, pred_item))
            # append_to_csv(path, pred_item)
            print(f"{file_name} => Predicted Class: {pred_item}")


df = pd.DataFrame(predictions, columns=["Filename", "ClassId"])



df_compare(df)