MODEL_NAME = "resnet18"

# Paths
TRAIN_DATA = "data/GTSRB/Training"
FINAL_TEST_PATH = "data/GTSRB/Final_Test"
GROUND_TRUTH_CSV = "data/GTSRB/GT-Final_test.csv"
MODEL_SAVE_PATH = "models/saved/last_resnet_model.pth"
# Validation Accuracy and Train Loss log
LOG_VA_TL = "logs/last_resnet18_log.csv"


# Model / Training
BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 0.001
NUM_CLASSES = 43

# Image Transform
IMG_SIZE = (32, 32)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
