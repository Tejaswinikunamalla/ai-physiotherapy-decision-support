import os
import shutil
import random

SOURCE_DIR = "new_bone_dataset"
TRAIN_DIR = "Bone_Dataset/train"
VAL_DIR = "Bone_Dataset/valid"

SPLIT_RATIO = 0.8  # 80% train

for category in ["Fractured", "Non_fractured"]:
    source_folder = os.path.join(SOURCE_DIR, category)
    images = os.listdir(source_folder)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(TRAIN_DIR, category), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, category), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(source_folder, img),
                    os.path.join(TRAIN_DIR, category, img))

    for img in val_images:
        shutil.copy(os.path.join(source_folder, img),
                    os.path.join(VAL_DIR, category, img))

print("✅ Dataset split completed correctly.")