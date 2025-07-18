import os, csv

def append_to_csv(filename, class_id, path="predictions.csv"):
    write_header = not os.path.exists(path)

    with open(path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        if write_header:
            writer.writerow(["Filename", "ClassId"])
        writer.writerow([filename, class_id])


def save_epoch_log(train_losses, val_accuracies, filename="training_log.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(["Epoch", "T Loss", "Val Acc"])
        for i, (loss, acc) in enumerate(zip(train_losses, val_accuracies), start=1):
            writer.writerow([i, round(loss, 6), round(acc, 6)])