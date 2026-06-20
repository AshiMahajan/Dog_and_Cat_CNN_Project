import json, os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import f1_score, classification_report

IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS = 35  # bumped up - previous run hadn't plateaued by epoch 20
DATA_DIR_TRAIN = "dataset/training_set"
DATA_DIR_VAL = "dataset/test_set"
OUT_MODEL = "models/basic_cnn.h5"           # final model, saved at the end
CKPT_MODEL = "models/basic_cnn_best.h5"     # best checkpoint, saved during training
OUT_CLASSES = "models/class_index.json"
OUT_REPORT = "reports/classification_report.txt"
OUT_HISTORY = "reports/history.json"

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
    shuffle=False,  # must stay False so predictions line up with val.classes
)

# --- Resume support ---
# If a checkpoint from a previous run already exists, load it and keep
# training from there instead of starting over from random weights.
if os.path.exists(CKPT_MODEL):
    print(f"Found existing checkpoint at {CKPT_MODEL} - resuming training from it.")
    model = tf.keras.models.load_model(CKPT_MODEL)
else:
    print("No checkpoint found - building a fresh model.")
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
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


class F1ValCallback(callbacks.Callback):
    """Computes true val F1 at the end of every epoch using sklearn."""

    def __init__(self, val_data):
        super().__init__()
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs=None):
        self.val_data.reset()
        y_true = self.val_data.classes
        y_prob = self.model.predict(self.val_data, verbose=0).ravel()
        y_pred = (y_prob > 0.5).astype(int)
        f1 = f1_score(y_true, y_pred)
        logs = logs or {}
        logs["val_f1"] = f1
        print(f" - val_f1: {f1:.4f}")


es = callbacks.EarlyStopping(
    patience=4, restore_best_weights=True, monitor="val_accuracy"
)
rlr = callbacks.ReduceLROnPlateau(patience=2, factor=0.5)
f1_cb = F1ValCallback(val)

# Saves the best model EVERY time val_accuracy improves - this is what lets
# you resume training later instead of starting from scratch, and protects
# you if the script crashes mid-run.
ckpt = callbacks.ModelCheckpoint(
    CKPT_MODEL,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

history = model.fit(
    train,
    validation_data=val,
    epochs=EPOCHS,
    callbacks=[es, rlr, f1_cb, ckpt],
)

# Final save (this reflects the best weights thanks to restore_best_weights,
# if EarlyStopping triggered - otherwise it's just the last epoch's weights,
# so CKPT_MODEL is your true "best" copy regardless)
model.save(OUT_MODEL)
with open(OUT_CLASSES, "w") as f:
    json.dump(train.class_indices, f, indent=2)

with open(OUT_HISTORY, "w") as f:
    json.dump(history.history, f, indent=2)

# Final, definitive evaluation report on the validation/test set
val.reset()
y_true = val.classes
y_prob = model.predict(val, verbose=0).ravel()
y_pred = (y_prob > 0.5).astype(int)

class_names = [k for k, v in sorted(train.class_indices.items(), key=lambda kv: kv[1])]
report = classification_report(y_true, y_pred, target_names=class_names)
final_f1 = f1_score(y_true, y_pred)

print("\nFinal validation F1 score:", final_f1)
print(report)

with open(OUT_REPORT, "w") as f:
    f.write(f"Final validation F1 score: {final_f1:.4f}\n\n")
    f.write(report)