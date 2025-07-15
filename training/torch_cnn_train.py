import torch
from tqdm import tqdm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
   import torch.optim.optimizer as optimizer



def train_torch_model(model : torch.nn.Module, train_loader, val_loader, epochs, criterion , optimizer : torch.optim.Optimizer, device, file_name=None):

    train_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss : torch.nn.CrossEntropyLoss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()



        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total

        # Saving best current model
        if file_name:
            best_acc = 0.0
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), file_name)
                
        print(f"Epoch {epoch+1}: Loss = {avg_train_loss:.4f}, Val Accuracy = {val_acc:.4f}")

        # For graph
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
    # Quality of life for sanity
    if file_name:
        print(f"Best current model saved to file {file_name}.")
    return train_losses, val_accuracies