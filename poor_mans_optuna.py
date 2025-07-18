# When you don't have time to learn Optuna, but need to run few automated tests

from configs import cnn_config as cfg
import importlib
import torch_train_pipeline

# Param grid
learning_rates = [0.0012, 0.0008, 0.0005]
dropouts = [0.0, 0.2, 0.4]
batch_sizes = [48, 64, 128]

run_id = 5
for lr in learning_rates:
    for dr in dropouts:
        for bs in batch_sizes:
            print(f"\n=== RUN {run_id} | LR={lr}, Dropout={dr}, Batch={bs} ===")
            cfg.LEARNING_RATE = lr
            cfg.DROPOUT_RATE = dr
            cfg.USE_DROPOUT = dr > 0
            cfg.BATCH_SIZE = bs
            cfg.EPOCHS = 10  # or whatever you're testing
            cfg.OPTIMIZER = "Adam"
            cfg.LOG_VA_TL = f"logs/last_cnn_log_run{run_id}.csv"
            cfg.MODEL_SAVE_PATH = f"models/saved/last_cnn_model_run{run_id}.pth"

            try:
                importlib.reload(torch_train_pipeline)
                # importlib.reload(importlib.import_module("torch_train_pipeline"))
            except Exception as e:
                print(f"Failed on run {run_id}: {e}")
            run_id += 1

# In case it would start crashing, i could: 
# import torch
# torch.cuda.empty_cache()
# or
# del model
# torch.cuda.empty_cache()
