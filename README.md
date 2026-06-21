---
title: Cat-Dog Classifier + OOD
emoji: 🐾
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🐾 Real-Time Cat vs Dog Classifier (Flask + MobileNetV2 + OOD Detection)

A production-ready image classifier that identifies **Cat**, **Dog**, or **Unknown Animal** — with:

- ✅ MobileNetV2 Transfer Learning
- ✅ Multi-crop voting for stability
- ✅ OOD (Out-of-Distribution) detection
- ✅ Sketch-detection safety net
- ✅ No file saving — in-memory image processing
- ✅ Clean dark UI
- ✅ Reports (Accuracy, ROC-AUC, Confusion Matrix)

---

## 🚀 Features

### ✅ 1. Real-time prediction

Upload any image → Model predicts:

- **Cat**
- **Dog**
- **Unknown Animal** (OOD detection)

### ✅ 2. Advanced ML Enhancements

- Transfer learning with MobileNetV2
- Global average pooling + embedding layer
- Centroid-based similarity for OOD
- Multi-crop voting at inference
- Sketch-friendly gating rules

### ✅ 3. Clean Flask Backend

- In-memory image processing (no disk required)
- Data URL image preview
- Config via environment variables
- Flash & validation

### ✅ 4. Fully production ready

- Works on Render, Railway, AWS EC2, Heroku
- Supports Docker
- No user-upload files stored
- Secure with SECRET_KEY

---

Free hosted web app on Hugging Face Spaces. Includes transfer learning,
multi-crop voting, OOD detection via embeddings, and in-memory uploads.