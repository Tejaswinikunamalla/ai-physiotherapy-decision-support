import os
import cv2

INPUT_DIR = "data/raw"
OUTPUT_DIR = "data/processed"
IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in os.listdir(INPUT_DIR):
    label_path = os.path.join(INPUT_DIR, label)
    output_label_path = os.path.join(OUTPUT_DIR, label)
    os.makedirs(output_label_path, exist_ok=True)

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(output_label_path, img_name), img)

print("Preprocessing completed successfully.")
