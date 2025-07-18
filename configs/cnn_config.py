MODEL_NAME = "simple_cnn"

# Paths
TRAIN_DATA = "data/GTSRB/Training"
FINAL_TEST_PATH = "data/GTSRB/Final_Test"
GROUND_TRUTH_CSV = "data/GTSRB/GT-Final_test.csv"
MODEL_SAVE_PATH = "models/saved/last_cnn_model.pth"
# Validation Accuracy and Train Loss log
LOG_VA_TL = "logs/last_cnn_log.csv"


OPTIMIZER = "Adam"

# Model / Training
NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


# Dropout
USE_DROPOUT = False
DROPOUT_RATE = 0.03


# Image Transform
IMG_SIZE = (32, 32)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


AUGMENTED_TRANSFORM = True

# Additional Image Augments to train data
RANDOM_BRIGHTNESS = 0.2 # 0.2 = up to ±20%
RANDOM_CONTRAST = 0.2
RANDOM_SATURATION = 0.2

MAX_WIDTH_SHIFT = 0.1 # 0.1 = up to ±10%, off center
MAX_HEIGHT_SHIFT = 0.1

RANDOM_ROTATION = 10 # 10 = 10 degrees clockwise or counter-clockwise

