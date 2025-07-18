import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from utils import GTSRBImageLoader, df_compare, get_torch_model, append_to_csv, get_default_transform, generate_class_summary

# Different config - different model
from configs import cfg_current as cfg 

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load unlabeled test dataset
final_dataset = GTSRBImageLoader(cfg.FINAL_TEST_PATH, transform=get_default_transform(), unlabeled=True)
final_loader = DataLoader(final_dataset, batch_size=32, shuffle=False)

# Load model
model = get_torch_model(cfg.MODEL_NAME).to(device)
model.load_state_dict(torch.load(cfg.LOAD_MODEL, map_location=device))
# model.load_state_dict(torch.load("models/saved/last_cnn_model3.pth", map_location=device))
model.eval()

# Predict
def torch_predict(model, final_loader, device):
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
                # print(f"{file_name} => Predicted Class: {pred_item}")
    return predictions


predictions = torch_predict(model, final_loader, device)

df_pred_list = pd.DataFrame(predictions, columns=["Filename", "ClassId"])
df_compare(df_pred_list, cfg.GROUND_TRUTH_CSV)

generate_class_summary(df_pred_list, cfg.GROUND_TRUTH_CSV)
