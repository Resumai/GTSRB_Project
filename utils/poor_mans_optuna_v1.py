# When you don't have time to learn Optuna, but need to run few automated tests

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from configs import cnn_config as cfg
import importlib
import torch_train_pipeline  # initial import to register module

for i, lr in enumerate([0.001, 0.0005, 0.0003]):
    cfg.LEARNING_RATE = lr
    # cfg.BATCH_SIZE = 64
    # cfg.DROPOUT_RATE = 0.2 + i * 0.1
    # cfg.USE_DROPOUT = True
    # cfg.EPOCHS = 5

    print(f"\n[RUN {i+1}] LR={lr}, Dropout={cfg.DROPOUT_RATE}")
    importlib.reload(torch_train_pipeline)
