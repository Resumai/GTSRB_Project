# Graph
# import matplotlib.pyplot as plt

# def plot_cnn_training(train_losses, val_accuracies):
#     epochs = range(1, len(train_losses) + 1)

#     fig, ax1 = plt.subplots(figsize=(10, 5))

#     color = 'tab:red'
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Train Loss', color=color)
#     ax1.plot(epochs, train_losses, color=color, label='Train Loss', marker='o')
#     ax1.tick_params(axis='y', labelcolor=color)

#     # Optional: zoom in to better see the changes in small loss values
#     ax1.set_ylim(0, max(train_losses) * 1.1)  # Or use ax1.set_ylim(0, 0.1) for tighter view

#     ax2 = ax1.twinx()  # share x-axis
#     color = 'tab:blue'
#     ax2.set_ylabel('Val Accuracy', color=color)
#     ax2.plot(epochs, val_accuracies, color=color, label='Val Accuracy', marker='o')
#     ax2.tick_params(axis='y', labelcolor=color)

#     plt.title("Training Loss & Validation Accuracy")
#     fig.tight_layout()
#     plt.show()



import matplotlib.pyplot as plt
import os

def plot_cnn_training(train_losses, val_accuracies, save_dir="logs/graphs"):
    os.makedirs(save_dir, exist_ok=True)

    # Find the next available numbered filename
    base_name = "tl_va_graph_"
    i = 3
    while os.path.exists(os.path.join(save_dir, f"{base_name}{i}.png")):
        i += 1
    file_path = os.path.join(save_dir, f"{base_name}{i}.png")

    # Plot as before
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, color=color, label='Train Loss', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(train_losses) * 1.1)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Val Accuracy', color=color)
    ax2.plot(epochs, val_accuracies, color=color, label='Val Accuracy', marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Loss & Validation Accuracy")
    fig.tight_layout()

    # Save figure
    plt.savefig(file_path)
    print(f"[INFO] Saved training graph: {file_path}")
    plt.close()
