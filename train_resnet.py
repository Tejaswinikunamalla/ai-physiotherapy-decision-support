import os
import tensorflow as tf
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
# Shortcuts
ResNet50 = tf.keras.applications.ResNet50
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dropout = tf.keras.layers.Dropout
Model = tf.keras.models.Model
Adam = tf.keras.optimizers.Adam

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

BASE_DIR = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "Bone_Dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "valid")

print("Train exists:", os.path.exists(TRAIN_DIR))
print("Val exists:", os.path.exists(VAL_DIR))

# -------------------------------
# Data Augmentation
# -------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("Class indices:", train_data.class_indices)

# -------------------------------
# Load ResNet50
# -------------------------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# -------------------------------
# Custom Classification Head
# -------------------------------
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# -------------------------------
# Compile Model
# -------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# Callbacks
# -------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/fracture_resnet_best.h5",
    monitor="val_accuracy",
    save_best_only=True
)

# -------------------------------
# Train Model
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# -------------------------------
# Save Final Model
# -------------------------------
model.save("models/fracture_resnet_final.h5")

print("Training complete. Model saved.")