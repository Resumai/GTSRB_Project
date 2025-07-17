MODEL_NAME = "simple_cnn"

# Paths
TRAIN_DATA = "data/GTSRB/Training"
FINAL_TEST_PATH = "data/GTSRB/Final_Test"
GROUND_TRUTH_CSV = "data/GTSRB/GT-Final_test.csv"
MODEL_SAVE_PATH = "models/saved/last_cnn_model.pth"

# Model / Training
BATCH_SIZE = 64
EPOCHS = 6
LEARNING_RATE = 0.001
NUM_CLASSES = 43

# Image Transform
IMG_SIZE = (32, 32)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
