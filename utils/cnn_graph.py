# Graph
import matplotlib.pyplot as plt

def plot_cnn_training(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # share x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Val Accuracy', color=color)
    ax2.plot(epochs, val_accuracies, color=color, label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Loss & Validation Accuracy")
    fig.tight_layout()
    plt.show()