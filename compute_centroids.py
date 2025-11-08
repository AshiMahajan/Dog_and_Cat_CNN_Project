import os, json, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_pre

MODEL_PATH = "models/mobilenetv2.h5"
TRAIN_DIR = "dataset/training_set"
IMG_SIZE = (128, 128)
BATCH = 32
OUT_PATH = "models/centroids.json"

model = tf.keras.models.load_model(MODEL_PATH)
embed_model = tf.keras.Model(model.input, model.get_layer("embed").output)

gen = ImageDataGenerator(preprocessing_function=mv2_pre)
flow = gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False,
)

embeds = []
labels = []
for _ in range(len(flow)):
    x, y = next(flow)
    e = embed_model.predict(x, verbose=0)
    embeds.append(e)
    labels.append(y)
embeds = np.vstack(embeds)
labels = np.concatenate(labels).astype(int)

centroids = {}
for cls_idx in [0, 1]:
    cls_emb = embeds[labels == cls_idx]
    centroids[str(cls_idx)] = np.mean(cls_emb, axis=0).tolist()

with open(OUT_PATH, "w") as f:
    json.dump(centroids, f)
print("Saved:", OUT_PATH)
