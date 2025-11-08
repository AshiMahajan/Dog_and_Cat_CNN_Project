import json, os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 20
DATA_DIR_TRAIN = "dataset/training_set"
DATA_DIR_VAL = "dataset/test_set"
OUT_MODEL = "models/basic_cnn.h5"
OUT_CLASSES = "models/class_index.json"

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
val_gen = ImageDataGenerator(rescale=1.0 / 255)

train = train_gen.flow_from_directory(
    DATA_DIR_TRAIN, target_size=IMG_SIZE, batch_size=BATCH, class_mode="binary"
)
val = val_gen.flow_from_directory(
    DATA_DIR_VAL,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False,
)

model = models.Sequential(
    [
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

es = callbacks.EarlyStopping(
    patience=4, restore_best_weights=True, monitor="val_accuracy"
)
rlr = callbacks.ReduceLROnPlateau(patience=2, factor=0.5)

history = model.fit(train, validation_data=val, epochs=EPOCHS, callbacks=[es, rlr])

model.save(OUT_MODEL)
with open(OUT_CLASSES, "w") as f:
    json.dump(train.class_indices, f, indent=2)
