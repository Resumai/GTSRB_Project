import torch
from tqdm import tqdm



def train_torch_model(model : torch.nn.Module, train_loader, val_loader, epochs, criterion , optimizer : torch.optim.Optimizer, device, model_save_path=None):
    train_losses = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        # tqdm is for progress bar in loops
        # One loop consists of whole batch size of data
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Moves to computing device of choice
            images, labels = images.to(device), labels.to(device)
            
            # 1. Clears gradients from previous iteration
            optimizer.zero_grad()
            # 2. Feeds batch of images through the model (forward pass)
            outputs = model(images)
            # 3. Compares predictions to true labels (computes batch loss)
            loss : torch.nn.CrossEntropyLoss = criterion(outputs, labels)
            # 4. Computes gradients via backward propagation of errors(backpropagation)
            loss.backward()
            # 5. Updates model parameters using computed gradients
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
        if model_save_path:
            best_acc = 0.0
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                
        print(f"Epoch {epoch+1}: Loss = {avg_train_loss:.4f}, Val Accuracy = {val_acc:.4f}")

        # For graph
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
    # Quality of life for sanity
    if model_save_path:
        print(f"Best current model saved to file {model_save_path}.")
    return train_losses, val_accuracies