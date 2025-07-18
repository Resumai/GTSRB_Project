import csv

def save_training_log(train_losses, val_accuracies, filename="training_log.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["Epoch", "T Loss", "Val Acc"])
        for i, (loss, acc) in enumerate(zip(train_losses, val_accuracies), start=1):
            writer.writerow([i, round(loss, 6), round(acc, 6)])