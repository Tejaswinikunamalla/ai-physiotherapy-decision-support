import os
from PIL import Image

DATASET_DIR = "Bone_Dataset"

def remove_corrupted_images(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                print("Removing corrupted:", path)
                os.remove(path)

remove_corrupted_images(DATASET_DIR)

print("✅ Corrupted images removed.")