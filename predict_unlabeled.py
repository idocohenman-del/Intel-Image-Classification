import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

IMAGE_DIR = "seg_pred"
MODEL_PATH = "saved_models/baseline_cnn.keras"
IMG_SIZE = (150, 150)

class_names = sorted([
    d.name for d in os.scandir("seg_train") if d.is_dir()
])

print("Classes:", class_names)

model = keras.models.load_model(MODEL_PATH)
print("Model loaded\n")

for img_name in os.listdir(IMAGE_DIR):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(IMAGE_DIR, img_name)

        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # ❗ NO /255 HERE

        preds = model.predict(img_array, verbose=0)
        idx = np.argmax(preds)
        conf = float(np.max(preds))

        print(f"{img_name} -> {class_names[idx]} ({conf:.2f})")
