import tensorflow as tf
from tensorflow import keras

# -------- Paths --------
TEST_DIR = "seg_pred"
MODEL_PATH = "saved_models/improved_cnn.keras"  # change per model

# -------- Parameters --------
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# -------- Load test data --------
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------- Normalization --------
normalization_layer = keras.layers.Rescaling(1./255)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# -------- Load model --------
model = keras.models.load_model(MODEL_PATH)

# -------- Evaluate --------
test_loss, test_acc = model.evaluate(test_ds)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
