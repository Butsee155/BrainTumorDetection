import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
import os

os.makedirs("models", exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 20
CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]

print("[INFO] Preparing data...")

# Data augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    validation_split=0.2
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    "data/Training",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    classes=CLASSES
)
val_data = train_gen.flow_from_directory(
    "data/Training",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    classes=CLASSES
)
test_data = test_gen.flow_from_directory(
    "data/Testing",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES
)

print(f"[INFO] Classes: {train_data.class_indices}")
print(f"[INFO] Training samples  : {train_data.samples}")
print(f"[INFO] Validation samples: {val_data.samples}")

# ── Build model using EfficientNetB0 transfer learning ────────────────────────
print("[INFO] Building model...")
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # freeze base layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")  # 4 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint("models/brain_tumor_model.h5",
                    save_best_only=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("[INFO] Training model...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ── Fine-tune ─────────────────────────────────────────────────────────────────
print("[INFO] Fine-tuning top layers...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    callbacks=callbacks
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(test_data)
print(f"\n[SUCCESS] Test Accuracy : {acc:.1%}")
print(f"[SUCCESS] Test Loss     : {loss:.4f}")
print("[SUCCESS] Model saved to models/brain_tumor_model.h5")