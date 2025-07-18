import torch
from torch.utils.data import DataLoader, random_split
from training.torch_train import train_torch_model
from utils import GTSRBImageLoader, plot_cnn_training, get_torch_model, save_training_log, get_augmented_transform, get_default_transform, log_experiment, display_log

# Different config - different model
from configs import cnn_config as cfg

# Device (CPU fallback)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Init and transform training dataset
dataset = GTSRBImageLoader(cfg.TRAIN_DATA)

# Total dataset size, to determine split sizes
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Load dataset, train/val split
train_data, val_data = random_split(dataset, [train_size, val_size])

# Apply transformations
if cfg.AUGMENTED_TRANSFORM:
    train_data.dataset.transform = get_augmented_transform()
else:
    train_data.dataset.transform = get_default_transform()
val_data.dataset.transform =  get_default_transform()

# Prepare loaders
# "batch_size" defines how many images are processed at once during training 
# can be set higher for speed on val_data, it shouldn't affect outcome(should test)
train_loader = DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=True) 
val_loader = DataLoader(val_data, batch_size=64, shuffle=False) 




# Model, criterion for loss, optimizer
model = get_torch_model(cfg.MODEL_NAME).to(device)
criterion = torch.nn.CrossEntropyLoss()
if cfg.OPTIMIZER == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
# TODO: Try MuonClip Optimizer if possible, recently got nice results, in LLM's tho


# Training loop, and saving results
train_losses, val_accuracies = train_torch_model(
    model, train_loader, val_loader,
    epochs=cfg.EPOCHS,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    model_save_path=cfg.MODEL_SAVE_PATH
)
save_training_log(train_losses, val_accuracies, filename=cfg.LOG_VA_TL)
log_experiment(
    log_path="logs/hparam_experiments.csv",
    model_name=cfg.MODEL_NAME,
    optimizer_name=cfg.OPTIMIZER,
    learning_rate=cfg.LEARNING_RATE,
    batch_size=cfg.BATCH_SIZE,
    dropout_enabled=cfg.USE_DROPOUT,
    dropout_rate=cfg.DROPOUT_RATE,
    epochs=cfg.EPOCHS,
    train_augments=cfg.AUGMENTED_TRANSFORM,
    r_brightness=cfg.RANDOM_BRIGHTNESS,
    r_contrast=cfg.RANDOM_CONTRAST,
    r_saturation=cfg.RANDOM_SATURATION,
    r_rotation=cfg.RANDOM_ROTATION,
    width_shift=cfg.MAX_WIDTH_SHIFT,
    height_shift=cfg.MAX_HEIGHT_SHIFT,
    train_loss=train_losses,
    val_accuracy=val_accuracies,
    notes="None"
)


display_log("logs/hparam_experiments.csv", max_rows=10)
plot_cnn_training(train_losses, val_accuracies)






# from utils.test_model_load import test_model_load
# test_model_load(model, val_loader, device)


# from training.sklearn_train import train_sklearn_model
# from sklearn.ensemble import RandomForestClassifier

# rf_model = RandomForestClassifier(n_estimators=100)
# acc = train_sklearn_model(rf_model, train_loader, val_loader)


