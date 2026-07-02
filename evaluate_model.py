# import os
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


# # ---------------- PATHS ----------------

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(BASE_DIR, "models", "fracture_resnet_best.h5")

# VAL_DIR = os.path.join(BASE_DIR, "Bone_Dataset", "valid")


# # ---------------- LOAD MODEL ----------------

# model = tf.keras.models.load_model(MODEL_PATH)

# print("Model loaded successfully")


# # ---------------- DATA GENERATOR ----------------

# IMG_SIZE = (224,224)
# BATCH_SIZE = 16

# val_gen = ImageDataGenerator(rescale=1./255)

# val_data = val_gen.flow_from_directory(
#     VAL_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary",
#     shuffle=False
# )

# print("Class indices:", val_data.class_indices)


# # ---------------- TRUE LABELS ----------------

# y_true = val_data.classes


# # ---------------- PREDICTIONS ----------------

# y_pred_prob = model.predict(val_data)

# threshold = 0.45
# y_pred = (y_pred_prob > threshold).astype(int).flatten()


# # ---------------- CONFUSION MATRIX ----------------

# cm = confusion_matrix(y_true, y_pred)

# print("\nConfusion Matrix:")
# print(cm)


# # ---------------- ACCURACY ----------------

# accuracy = accuracy_score(y_true, y_pred)

# print("\nAccuracy:", accuracy)


# # ---------------- CLASSIFICATION REPORT ----------------

# print("\nClassification Report:\n")

# print(
#     classification_report(
#         y_true,
#         y_pred,
#         target_names=["Fractured", "Non_fractured"]
#     )
# )


# # ---------------- PLOT MATRIX ----------------

# plt.figure(figsize=(6,5))

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Fractured","Non_fractured"],
#     yticklabels=["Fractured","Non_fractured"]
# )

# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix - ResNet")

# plt.show()
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,roc_curve,auc
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
from PIL import ImageFile

# Fix corrupted image issue
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "Bone_Dataset")
VAL_DIR = os.path.join(DATASET_DIR, "valid")

MODEL_PATH = os.path.join(BASE_DIR, "models", "fracture_resnet_best.h5")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# -----------------------------
# Load Model
# -----------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully\n")

# -----------------------------
# Load Validation Data
# -----------------------------
val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class indices:", val_data.class_indices)
print()

# -----------------------------
# Predict
# -----------------------------
print("Running predictions...\n")

pred_probs = model.predict(val_data)

# Convert probabilities → binary predictions
predictions = (pred_probs > 0.5).astype(int)

true_labels = val_data.classes

# -----------------------------
# Metrics
# -----------------------------
cm = confusion_matrix(true_labels, predictions)

accuracy = accuracy_score(true_labels, predictions)

report = classification_report(
    true_labels,
    predictions,
    target_names=list(val_data.class_indices.keys())
)

print("Confusion Matrix:\n")
print(cm)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(report)

# -----------------------------
# Plot Confusion Matrix
# -----------------------------
plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(val_data.class_indices.keys()),
    yticklabels=list(val_data.class_indices.keys())
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
# -----------------------------
# ROC Curve
# -----------------------------

# true labels
y_true = val_data.classes

# predicted probabilities
y_scores = pred_probs

# compute ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# compute AUC
roc_auc = auc(fpr, tpr)

print("\nAUC Score:", roc_auc)

# plot ROC curve
plt.figure()

plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)

plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.show()