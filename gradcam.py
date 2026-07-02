# import numpy as np
# import tensorflow as tf
# import cv2
# import matplotlib.pyplot as plt
# #from tensorflow.keras.preprocessing import image
# image = tf.keras.preprocessing.image
# MODEL_PATH = "models/fracture_resnet_best.h5"
# IMG_PATH = "temp_xray.jpg"

# IMG_SIZE = (224,224)

# model = tf.keras.models.load_model(MODEL_PATH)

# # Last convolution layer of ResNet
# last_conv_layer_name = "conv5_block3_out"

# # preprocess image
# img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
# img_array = image.img_to_array(img)/255.0
# img_array = np.expand_dims(img_array, axis=0)

# # prediction
# pred = model.predict(img_array)
# print("Prediction probability:", pred[0][0])

# # create grad model
# grad_model = tf.keras.models.Model(
#     [model.inputs],
#     [model.get_layer(last_conv_layer_name).output, model.output]
# )

# with tf.GradientTape() as tape:
#     conv_outputs, predictions = grad_model(img_array)
#     loss = predictions[:,0]

# grads = tape.gradient(loss, conv_outputs)

# pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

# conv_outputs = conv_outputs[0]

# heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
# heatmap = tf.squeeze(heatmap)

# heatmap = np.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

# heatmap = heatmap.numpy()

# # resize heatmap
# #heatmap = cv2.resize(heatmap, (224,224))
# heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
# # convert original image
# img = cv2.imread(IMG_PATH)
# img = cv2.resize(img,(224,224))

# heatmap = np.uint8(255 * heatmap)
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# superimposed_img = heatmap * 0.4 + img

# # show result
# plt.figure(figsize=(10,4))

# plt.subplot(1,3,1)
# plt.title("Original")
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# plt.subplot(1,3,2)
# plt.title("Heatmap")
# plt.imshow(heatmap)

# plt.subplot(1,3,3)
# plt.title("GradCAM Result")
# plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))

# plt.show()
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Alias (same style as your other files)
image = tf.keras.preprocessing.image

MODEL_PATH = "models/fracture_resnet_best.h5"
IMG_PATH = "frac.jpeg"

IMG_SIZE = (224,224)

model = tf.keras.models.load_model(MODEL_PATH)

last_conv_layer_name = "conv5_block3_out"

# -----------------------------
# Load Image
# -----------------------------
img = image.load_img(IMG_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# -----------------------------
# Prediction
# -----------------------------
draw_box=False
pred = model.predict(img_array)
print("Prediction probability:", pred[0][0])
if pred[0][0]>0.5:
    draw_box=True
else:
    draw_box=False
# -----------------------------
# GradCAM Model
# -----------------------------
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)

# -----------------------------
# Gradients
# -----------------------------
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    loss = predictions[:,0]

grads = tape.gradient(loss, conv_outputs)

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)

heatmap = np.maximum(heatmap,0)
heatmap = heatmap / np.max(heatmap)

heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / np.max(heatmap)

# DO NOT add .numpy() here

# -----------------------------
# Resize heatmap
# -----------------------------
heatmap = cv2.resize(heatmap,(224,224),interpolation=cv2.INTER_CUBIC)
heatmap = cv2.GaussianBlur(heatmap,(11,11),0)

# -----------------------------
# Load original image
# -----------------------------
img = cv2.imread(IMG_PATH)
img = cv2.resize(img,(224,224))

# Convert heatmap to color
heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

# Overlay
overlay = cv2.addWeighted(img,0.6,heatmap_color,0.4,0)

# -----------------------------
# Find high attention region
# -----------------------------
threshold = 0.75
mask = heatmap > threshold

mask = mask.astype("uint8") * 255

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours and draw_box:
    largest = max(contours, key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(largest)

    cv2.rectangle(
        overlay,
        (x,y),
        (x+w,y+h),
        (0,255,0),
        2
    )

# -----------------------------
# Display
# -----------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original X-ray")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Heatmap")
plt.imshow(heatmap)
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Fracture Detection Region")
plt.imshow(cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()