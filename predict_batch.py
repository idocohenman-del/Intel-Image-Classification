import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# -------- CONFIG --------
IMAGE_DIR = "seg_pred"
MODEL_PATH = "saved_models/improved_cnn.keras"   # change if needed
IMG_SIZE = (150, 150)
MAX_IMAGES = 100

CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
# ------------------------

model = keras.models.load_model(MODEL_PATH)
print("Model loaded\n")

images = [f for f in os.listdir(IMAGE_DIR)
          if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:MAX_IMAGES]

for img_name in images:
    img_path = os.path.join(IMAGE_DIR, img_name)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    label = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    print(f"{img_name} -> {label} ({confidence:.2f})")
