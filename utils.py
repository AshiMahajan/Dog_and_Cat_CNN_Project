import json, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing import image
import io, base64
from PIL import Image


def load_class_index(path="models/class_index.json"):
    with open(path) as f:
        d = json.load(f)  # {'cat': 0, 'dog': 1}
    return {v: k for k, v in d.items()}  # {0: 'cat', 1: 'dog'}


def prepare_array(img, target):
    img = img.resize(target)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def predict_image_voted(
    model, pil_img, img_size, class_map, preproc=None, threshold=0.5
):
    # crops from the PIL image
    im = pil_img.convert("RGB")
    W, H = im.size
    crops = [im]
    s = min(W, H)
    cx, cy = (W - s) // 2, (H - s) // 2
    crops += [
        im.crop((cx, cy, cx + s, cy + s)),
        im.crop((0, 0, s, s)),
        im.crop((W - s, 0, W, s)),
        im.crop((0, H - s, s, H)),
        im.crop((W - s, H - s, W, H)),
    ]

    probs = []
    for c in crops:
        x = _prepare_x_from_pil(c, img_size, preproc)
        p = float(model.predict(x, verbose=0)[0][0])
        probs.append(p)

    probs = np.array(probs, dtype=float)
    mean_p = float(probs.mean())
    std_p = float(probs.std())
    pred_idx = int(mean_p >= threshold)
    label = class_map.get(pred_idx, f"class{pred_idx}")
    confidence = mean_p if pred_idx == 1 else (1.0 - mean_p)
    disagree = (probs >= 0.5).any() and (probs < 0.5).any()
    high_var = std_p > 0.12
    ambiguous = bool(disagree or high_var)
    return label, confidence, mean_p, std_p, ambiguous


def _cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def load_centroids(path="models/centroids.json"):
    with open(path) as f:
        d = json.load(f)  # {"0":[...], "1":[...]}
    return {int(k): np.array(v, dtype=np.float32) for k, v in d.items()}


def get_embed_model(model):
    try:
        layer = model.get_layer("embed")
    except ValueError:
        # fallback: penultimate layer
        layer = model.layers[-2]
    return tf.keras.Model(model.input, layer.output)


def predict_image_with_ood(
    model,
    pil_img,
    img_size,
    class_map,
    preproc=None,
    prob_threshold=0.5,
    centroids=None,
):
    im = pil_img.convert("RGB")
    x = _prepare_x_from_pil(im, img_size, preproc)

    p = float(model.predict(x, verbose=0)[0][0])
    pred_idx = int(p >= prob_threshold)
    conf = p if pred_idx == 1 else (1.0 - p)
    label = class_map.get(pred_idx, f"class{pred_idx}")

    if not centroids:
        return label, conf, p, False, 0.0, 0.0

    embed_model = get_embed_model(model)
    e = embed_model.predict(x, verbose=0).reshape(-1)

    sims_items = sorted(
        ((k, _cosine(e, v)) for k, v in centroids.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    best_cls, best_sim = sims_items[0]
    second_sim = sims_items[1][1] if len(sims_items) > 1 else -1.0
    margin = float(best_sim - second_sim)
    return label, conf, p, False, float(best_sim), margin


def pil_to_data_url(pil_img, format="JPEG", quality=90):
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"


def _prepare_x_from_pil(pil_img, target, preproc=None):
    arr = image.img_to_array(pil_img.resize(target))
    x = np.expand_dims(arr, axis=0)
    return preproc(x) if preproc is not None else (x / 255.0)
