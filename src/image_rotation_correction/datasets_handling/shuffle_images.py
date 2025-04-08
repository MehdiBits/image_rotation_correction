

import os
import random
import shutil
import sys 

import config

# Paths
DATASET_DIR = config.DATASET_DIR
TRAIN_DIR = config.TRAIN_DIR
VAL_DIR = config.VAL_DIR
TEST_DIR = config.TEST_DIR

# Create train/val/test folders
for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)

# Get all image filenames
all_images = [f for f in os.listdir(DATASET_DIR) if f.endswith((".jpg", ".png"))]

# Shuffle images randomly
random.shuffle(all_images)

# Define split sizes
train_size = int(0.7 * len(all_images))
val_size = int(0.2 * len(all_images))
test_size = len(all_images) - train_size - val_size  # Ensure total = 100%

# Assign images
train_images = all_images[:train_size]
val_images = all_images[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]

# Move images
for img in train_images:
    shutil.copy(os.path.join(DATASET_DIR, img), os.path.join(TRAIN_DIR, img))

for img in val_images:
    shutil.copy(os.path.join(DATASET_DIR, img), os.path.join(VAL_DIR, img))

for img in test_images:
    shutil.copy(os.path.join(DATASET_DIR, img), os.path.join(TEST_DIR, img))

print(f"✅ Split complete: {train_size} train, {val_size} val, {test_size} test.")
