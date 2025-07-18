import csv
import os

def log_experiment(
    log_path,
    model_name,
    optimizer_name,
    learning_rate,
    batch_size,
    dropout_enabled,
    dropout_rate,
    epochs,
    train_augments,
    r_brightness,
    r_contrast,
    r_saturation,
    r_rotation,
    width_shift,
    height_shift,
    train_loss,
    val_accuracy,
    notes=""
):
    log_exists = os.path.isfile(log_path)

    if not dropout_enabled:
        dropout_rate = "N/A"
    if not train_augments:
        r_brightness = "N/A"
        r_contrast = "N/A"
        r_saturation = "N/A"
        r_rotation = "N/A"
        width_shift = "N/A"
        height_shift = "N/A"

    # Determine current run number
    run_number = 1
    if log_exists:
        with open(log_path, mode="r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            run_number = len(rows) 

    with open(log_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not log_exists:
            writer.writerow([
                "#", "Model", "Optimizer", "LR", 
                "BatchSize", "DropoutRate", "Epochs",
                "RBrighness","RContrast", "RSaturation",
                "RRotation","WidthShift", "HeightShift",
                "FinalTrainLoss", "FinalValAcc", "Notes"
            ])



        writer.writerow([
            run_number,
            model_name,
            optimizer_name,
            learning_rate,
            batch_size,
            dropout_rate,
            epochs,
            r_brightness,
            r_contrast,
            r_saturation,
            r_rotation,
            width_shift,
            height_shift,
            round(train_loss[-1], 5) if train_loss else "N/A",
            round(val_accuracy[-1], 5) if val_accuracy else "N/A",
            notes
        ])



import pandas as pd

def display_log(path="logs/hparam_experiments.csv", max_rows=20):
    try:
        df = pd.read_csv(path)
        print(df.tail(max_rows).to_string(index=False))
    except FileNotFoundError:
        print("⚠️ Log file not found.")
    except Exception as e:
        print(f"⚠️ Failed to read log: {e}")
