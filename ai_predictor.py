
import os
import numpy as np
import tensorflow as tf
from PIL import ImageFile 

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Safe aliases (same style as your ResNet code)
load_model = tf.keras.models.load_model
image = tf.keras.preprocessing.image

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fracture_resnet_best.h5")

model = load_model(MODEL_PATH)

# ---------------- IMPORTANT ----------------
# SET THIS BASED ON YOUR train_data.class_indices OUTPUT 
# Example:
# If you saw: {'fracture': 0, 'non_fracture': 1}
# then FRACTURE_CLASS_INDEX = 0
#
# If you saw: {'fracture': 1, 'non_fracture': 0}
# then FRACTURE_CLASS_INDEX = 1

FRACTURE_CLASS_INDEX = 0   # 🔴 CHANGE THIS if needed


# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------------- PREDICTION FUNCTION ----------------
def predict_fracture(img_path):
    img_array = preprocess_image(img_path)

    # Sigmoid output (single value between 0 and 1)
    raw_output = model.predict(img_array)[0][0]

    print("Raw model output:", raw_output)

    # -------- Correct Probability Mapping --------
    if FRACTURE_CLASS_INDEX == 1:
        fracture_prob = raw_output
    else:
        fracture_prob = 1 - raw_output

    non_fracture_prob = 1 - fracture_prob

    print("Fracture probability:", fracture_prob)
    print("Non-fracture probability:", non_fracture_prob)

    # -------- Scores --------
    ai_risk_score = int(fracture_prob * 100)
    confidence = int(max(fracture_prob, non_fracture_prob) * 100)

    # -------- Decision Threshold --------
    THRESHOLD = 0.5

    if fracture_prob >= THRESHOLD:
        structural_status = "Abnormal"
    else:
        structural_status = "Normal"

    return ai_risk_score, confidence, structural_status