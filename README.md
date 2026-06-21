# 🐾 AniLens — Cat vs Dog Classifier (Flask + MobileNetV2 + OOD Detection)

A production-ready image classifier that identifies **Cat**, **Dog**, or **Unknown Animal** — built and compared across a from-scratch CNN and a MobileNetV2 transfer-learning model.

**🔗 Live demo:** [huggingface.co/spaces/Leonopteryx/AniLens](https://huggingface.co/spaces/Leonopteryx/AniLens)

---

## 📸 Screenshots

### Prediction page
![Model reports](screenshots/Model%20Report.png)

### Model comparison reports
![Prediction page](screenshots/Prediction%20Page.png)
---

## 📊 Model performance

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| CNN (from scratch) | 88.0% | 0.880 | 0.950 |
| MobileNetV2 (transfer learning, fine-tuned) | 97.9% | 0.979 | 0.997 |

Trained and evaluated on a 16K-image cat/dog dataset (8,000 train / 8,000 test, perfectly class-balanced, no image overlap between splits).

---

## 🚀 Features

### Real-time prediction
Upload any image → model predicts **Cat**, **Dog**, or **Unknown Animal** (out-of-distribution rejection).

### Advanced ML techniques
- Two-stage transfer learning: frozen-base head training, then fine-tuning the last 20 MobileNetV2 layers
- Global average pooling + dedicated embedding layer
- Centroid-based cosine similarity for out-of-distribution detection
- Multi-crop inference voting (6 crops per image, batched into a single forward pass)
- Sketch-detection gating rules for non-photographic inputs
- Per-epoch F1/precision/recall tracking via custom Keras callbacks, beyond plain accuracy

### Clean Flask backend
- Fully in-memory image processing — no files written to disk
- Data URL image preview, no upload storage
- Config via environment variables
- Cached embedding model (built once at startup, not per-request)

### Deployment
- Dockerized, deployed on Hugging Face Spaces (free CPU tier)
- Also portable to Render, Railway, or any Docker-compatible host

---

## 🧠 Project structure

```
.
├── train_cnn.py          # From-scratch CNN training (with checkpoint/resume support)
├── train_transfer.py     # MobileNetV2 transfer learning (staged fine-tuning)
├── compute_centroids.py  # Generates per-class embedding centroids for OOD detection
├── evaluate.py           # Generates accuracy/F1/ROC-AUC reports for both models
├── app.py                # Flask inference app
├── utils.py               # Prediction helpers (multi-crop voting, OOD, embeddings)
├── templates/             # index.html, reports.html
├── models/                # Trained model weights + class index + centroids
└── reports/               # Evaluation metrics, confusion matrices, ROC curves
```

---

## 🛠️ Tech stack

TensorFlow / Keras · Flask · scikit-learn · Docker · Hugging Face Spaces

---

## 🏃 Running locally

```bash
pip install -r requirements.txt
python app.py
```

Or via Docker:

```bash
docker build -t anilens .
docker run -p 7860:7860 anilens
```