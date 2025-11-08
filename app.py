import os, json, glob
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
)
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_pre
from utils import load_class_index, load_centroids
import io, base64
from PIL import Image, ImageStat, ImageFilter
from utils import (
    load_centroids,
    predict_image_voted,
    predict_image_with_ood,
    pil_to_data_url,
)

# App config
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "uploads")
app.config["MAX_CONTENT_LENGTH"] = int(
    os.environ.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024)
)  # 10 MB
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change_me")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Model selection
MODEL_NAME = os.environ.get("MODEL_NAME", "mobilenetv2").lower()
if MODEL_NAME == "basic":
    MODEL_PATH = "models/basic_cnn.h5"
    IMG_SIZE = (128, 128)
    PREPROC = None  # utils handles /255.0 when PREPROC is None
else:
    MODEL_NAME = "mobilenetv2"
    MODEL_PATH = "models/mobilenetv2.h5"
    IMG_SIZE = (128, 128)
    PREPROC = mv2_pre

# Load model & class map
model = tf.keras.models.load_model(MODEL_PATH)

# Hardcode mapping to avoid pluralized labels from folder names
class_map = {0: "cat", 1: "dog"}

# Optionally attempt to load centroids for OOD rejection
centroids = None
try:
    centroids = load_centroids("models/centroids.json")
except Exception:
    centroids = None

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Reports helpers
def _load_report(tag):
    metrics_path = f"reports/metrics_{tag}.json"
    cm_path = f"reports/cm_{tag}.png"
    roc_path = f"reports/roc_{tag}.png"
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return {
        "tag": tag,
        "metrics": metrics,
        "cm_path": cm_path if os.path.exists(cm_path) else None,
        "roc_path": roc_path if os.path.exists(roc_path) else None,
    }


def is_sketch_like(path, sat_thresh=25, edge_thresh=20):
    from PIL import Image, ImageStat, ImageFilter

    im = Image.open(path).convert("RGB")

    # Saturation (S channel of HSV)
    s_mean = ImageStat.Stat(im.convert("HSV").split()[1]).mean[0]

    # Edge density (FIND_EDGES on grayscale)
    edges = im.convert("L").filter(ImageFilter.FIND_EDGES)
    e_mean = ImageStat.Stat(edges).mean[0]

    return (s_mean < sat_thresh) and (e_mean > edge_thresh)


def is_sketch_like_pil(pil_img, sat_thresh=25, edge_thresh=20):
    im = pil_img.convert("RGB")
    s_mean = ImageStat.Stat(im.convert("HSV").split()[1]).mean[0]
    edges = im.convert("L").filter(ImageFilter.FIND_EDGES)
    e_mean = ImageStat.Stat(edges).mean[0]
    return (s_mean < sat_thresh) and (e_mean > edge_thresh)


# Routes
@app.route("/", methods=["GET"])
def index():
    # Always pass result/image_url so template never fails on first load
    return render_template(
        "index.html", model_name=MODEL_NAME, result=None, image_url=None
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("index"))

    if "image" not in request.files or request.files["image"].filename == "":
        flash("Please select an image.")
        return redirect(url_for("index"))

    f = request.files["image"]
    if not allowed_file(f.filename):
        flash("Only JPG/PNG images are allowed.")
        return redirect(url_for("index"))

    # ---- in-memory read ----
    img_bytes = f.read()
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        flash("Could not read the image file.")
        return redirect(url_for("index"))

    # Data URL for preview (no disk write)
    image_data_url = pil_to_data_url(pil_img, format="JPEG")

    # 1) Multi-crop voting (ambiguity)
    label_vote, conf_vote, raw_prob_vote, prob_std, ambiguous_vote = (
        predict_image_voted(model, pil_img, IMG_SIZE, class_map, PREPROC)
    )

    # 2) Embedding metrics for OOD
    label_ood, conf_ood, raw_prob_ood, _ignored, best_sim, margin = (
        predict_image_with_ood(
            model, pil_img, IMG_SIZE, class_map, PREPROC, centroids=centroids
        )
    )

    # ----- Combined decision rule (ORDER MATTERS) -----
    sim_hi = 0.90
    sim_lo = 0.86
    hard_low = 0.84
    conf_hi = 0.80
    conf_lo = 0.60
    margin_lo = 0.05

    sketchy = is_sketch_like_pil(pil_img)
    if sketchy:
        sim_hi = 0.87
        sim_lo = 0.80
        hard_low = 0.72  # allow much lower similarity for sketches
        sketch_conf_ok = 0.65
    else:
        sketch_conf_ok = None

    is_ambiguous = bool(ambiguous_vote or (margin < margin_lo))

    # 0) HARD OOD VETO (with sketch override)
    if best_sim < hard_low:
        if sketchy and (conf_vote >= sketch_conf_ok):
            final_label = label_vote
            is_ood = False
            is_ambiguous = True
        else:
            final_label = "unknown animal"
            is_ood = True

    # 1) Reject when BOTH weak (photos only)
    elif (best_sim < sim_lo) and (conf_vote < conf_lo) and (not sketchy):
        final_label = "unknown animal"
        is_ood = True

    # 2) Accept on strong similarity
    elif best_sim >= sim_hi:
        final_label = label_vote
        is_ood = False

    # 3) Accept on high confidence if similarity isn't very low
    elif (conf_vote >= conf_hi) and (
        best_sim >= (sim_lo if not sketchy else (sim_lo - 0.03))
    ):
        final_label = label_vote
        is_ood = False

    # 4) Sketch fallback: reasonably confident sketch => accept
    elif sketchy and (conf_vote >= sketch_conf_ok):
        final_label = label_vote
        is_ood = False
        is_ambiguous = True

    else:
        final_label = label_vote
        is_ood = False

    confidence = 0.0 if is_ood else conf_vote
    raw_prob = raw_prob_vote

    # Debug log
    print(
        {
            "vote": {
                "label": label_vote,
                "conf": conf_vote,
                "raw": raw_prob_vote,
                "std": prob_std,
                "ambiguous": ambiguous_vote,
            },
            "embed": {"best_sim": best_sim, "margin": margin},
            "sketchy": sketchy,
            "final": {
                "label": final_label,
                "is_ood": is_ood,
                "ambiguous": is_ambiguous,
            },
        }
    )

    result = {
        "label": final_label,
        "confidence": round(confidence * 100, 2),
        "probability": round(raw_prob, 6),
        "std": round(prob_std, 4),
        "ambiguous": is_ambiguous,
        "is_ood": is_ood,
        "similarity": round(best_sim, 3),
        "margin": round(margin, 3),
        "sketchy": sketchy,
    }

    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        result=result,
        image_url=image_data_url,  # <--- data URL preview (no disk)
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/accuracy", methods=["GET"])
def accuracy_page():
    tags = ["basic", "vgg16", "mobilenetv2"]
    reports = [r for t in tags if (r := _load_report(t)) is not None]
    for path in glob.glob("reports/metrics_*.json"):
        tag = os.path.basename(path).replace("metrics_", "").replace(".json", "")
        if not any(r["tag"] == tag for r in reports):
            r = _load_report(tag)
            if r:
                reports.append(r)
    return render_template("reports.html", reports=reports)


@app.route("/api/reports", methods=["GET"])
def reports_api():
    tags = ["basic", "vgg16", "mobilenetv2"]
    payload = []
    for t in tags:
        r = _load_report(t)
        if r:
            payload.append(r)
    return {"reports": payload}, 200


@app.route("/report-file")
def static_report_file():
    path = request.args.get("path", "")
    base = os.path.abspath("reports")
    target = os.path.abspath(path)
    if not target.startswith(base) or not os.path.exists(target):
        return {"error": "Not found"}, 404
    dir_name = os.path.dirname(target)
    file_name = os.path.basename(target)
    return send_from_directory(dir_name, file_name)


@app.route("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}, 200


def delete_later(path, seconds=30):
    Timer(seconds, lambda: os.remove(path) if os.path.exists(path) else None).start()


if __name__ == "__main__":
    app.run(debug=True)
