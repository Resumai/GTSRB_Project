from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def train_sklearn_model(model, train_loader, val_loader):
    # Flatten train data from PyTorch loaders into arrays for sklearn
    X_train, y_train = [], []
    for images, labels in train_loader:
        X_batch = images.view(images.size(0), -1).numpy()
        y_batch = labels.numpy()
        X_train.extend(X_batch)
        y_train.extend(y_batch)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model.fit(X_train, y_train)

    # Validation
    X_val, y_val = [], []
    for images, labels in val_loader:
        X_batch = images.view(images.size(0), -1).numpy()
        y_batch = labels.numpy()
        X_val.extend(X_batch)
        y_val.extend(y_batch)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Random Forest Validation Accuracy: {acc:.4f}")

    return acc

    

