import torch
from torch.utils.data import DataLoader, random_split
from models.simple_cnn import SimpleCNN
from utils.GTSRB_Loader import GTSRBImageLoader
from utils.cnn_graph import plot_cnn_training
from training.torch_cnn_train import train_torch_model
from configs import cnn_config as cfg

# Device (CPU fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Init and transform training dataset
dataset = GTSRBImageLoader(cfg.TRAIN_DATA)

# Total dataset size, to determine split sizes
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Load dataset, train/val split
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Init model
model = SimpleCNN().to(device)

train_losses, val_accuracies = train_torch_model(
    model, 
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True), 
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False), 
    epochs = cfg.EPOCHS, 
    criterion = torch.nn.CrossEntropyLoss(), 
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE), 
    device=device, 
    file_name = cfg.MODEL_SAVE_PATH
)


plot_cnn_training(train_losses, val_accuracies)





# from utils.test_model_load import test_model_load
# test_model_load(model, val_loader, device)


# from training.sklearn_train import train_sklearn_model
# from sklearn.ensemble import RandomForestClassifier

# rf_model = RandomForestClassifier(n_estimators=100)
# acc = train_sklearn_model(rf_model, train_loader, val_loader)


