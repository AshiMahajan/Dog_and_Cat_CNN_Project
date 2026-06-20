import os, json, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_pre
from tensorflow.keras import callbacks
from sklearn.metrics import f1_score, classification_report

IMG_SIZE = (128, 128)
BATCH = 32
EPOCHS_FINE_TUNE = 5
DATA_DIR_TRAIN = "dataset/training_set"
DATA_DIR_VAL = "dataset/test_set"
MODEL_PATH = "models/mobilenetv2.h5"   # the model saved from your last run
OUT_REPORT = "reports/transfer_classification_report.txt"
OUT_HISTORY = "reports/transfer_finetune_resume_history.json"

os.makedirs("reports", exist_ok=True)

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
val = val_gen.flow_from_directory(
    DATA_DIR_VAL, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode="binary", shuffle=False,
)

# Load the model exactly as it was saved at the end of your last run
# (base already unfrozen for the last 20 layers, since that happened
# before training in the previous script)
model = tf.keras.models.load_model(MODEL_PATH)

# Re-compile is required after load if you want to be 100% sure optimizer
# state matches the fine-tune learning rate (1e-5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)


class F1ValCallback(callbacks.Callback):
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


# FRESH EarlyStopping instance - this is the fix. Starts with no memory
# of stage 1's val_accuracy peak, so it judges fine-tuning fairly.
es_ft = callbacks.EarlyStopping(
    patience=3, restore_best_weights=True, monitor="val_accuracy"
)
f1_cb = F1ValCallback(val)

history_ft = model.fit(
    train, validation_data=val, epochs=EPOCHS_FINE_TUNE,
    callbacks=[es_ft, f1_cb],
)

model.save(MODEL_PATH)  # overwrite with the (correctly) further fine-tuned version

with open(OUT_HISTORY, "w") as f:
    json.dump(history_ft.history, f, indent=2)

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