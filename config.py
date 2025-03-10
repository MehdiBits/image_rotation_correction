DATASET_DIR = "data/all_images"
TRAIN_DIR = "data/train/"
VAL_DIR = "data/val/"
TEST_DIR = "data/test/"


BATCH_SIZE = 32
LR = 1e-4
NUM_EPOCHS = 10
SAVE_EVERY = 2
LOAD_MODEL = True
MAX_ANGLE = 30
CHECKPOINT_PATH = f"checkpoints/rotation_model_{MAX_ANGLE}.pth"
