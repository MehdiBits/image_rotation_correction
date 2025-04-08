DATASET_DIR = "image_rotation_correction/data/all_images"
TRAIN_DIR = "image_rotation_correction/data/train/"
VAL_DIR = "image_rotation_correction/data/val/"
TEST_DIR = "image_rotation_correction/data/test/"

DEVICE = 'cpu'
BATCH_SIZE = 32
LR = 1e-4
NUM_EPOCHS = 200
SAVE_EVERY = 2
LOAD_MODEL = True
MAX_ANGLE = 30
CHECKPOINT_PATH = f"image_rotation_correction/checkpoints/rotation_model_{MAX_ANGLE}.pth"
