import os, json, random, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_pre
from tensorflow.keras import layers, models, callbacks

# Reproducibility (optional)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Config
IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS_HEAD = 10  # stage 1 (frozen base)
EPOCHS_FINE_TUNE = 5  # stage 2 (unfreeze last layers)
DATA_DIR_TRAIN = "dataset/training_set"
DATA_DIR_VAL = "dataset/test_set"
OUT_MODEL = "models/mobilenetv2.h5"
OUT_CLASSES = "models/class_index.json"

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Data
train_gen = ImageDataGenerator(
    preprocessing_function=mv2_pre,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_gen = ImageDataGenerator(preprocessing_function=mv2_pre)

train = train_gen.flow_from_directory(
    DATA_DIR_TRAIN, target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary"
)
print("Class mapping:", train.class_indices)

val = val_gen.flow_from_directory(
    DATA_DIR_VAL,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False,
)

# Model (MobileNetV2 base)
base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False  # Stage 1: freeze base

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dense(128, activation="relu", name="embed")(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation="sigmoid", name="out")(x)
model = models.Model(inputs=base.input, outputs=out)

# Train - Stage 1 (frozen base)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

es = callbacks.EarlyStopping(
    patience=3, restore_best_weights=True, monitor="val_accuracy"
)
rlr = callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss")

history_head = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS_HEAD,
    callbacks=[es, rlr],
)

# Fine-tune - Stage 2 (unfreeze tail)
base.trainable = True
# Unfreeze only the last ~20 layers for gentle fine-tuning
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS_FINE_TUNE,
    callbacks=[es],
)

# Save
model.save(OUT_MODEL)
with open(OUT_CLASSES, "w") as f:
    json.dump(train.class_indices, f, indent=2)
