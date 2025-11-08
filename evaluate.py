import os, json, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mv2_pre

VAL_DIR = "dataset/test_set"  # <- your validation/test directory
BATCH = 32


def evaluate(model_path, img_size, preproc=None, tag="model"):
    os.makedirs("reports", exist_ok=True)

    # Build validation generator with correct preprocessing
    if preproc is None:
        val_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    else:
        val_gen = ImageDataGenerator(preprocessing_function=preproc)

    val = val_gen.flow_from_directory(
        VAL_DIR,
        target_size=img_size,
        batch_size=BATCH,
        class_mode="binary",
        shuffle=False,
    )

    # Load model and predict probabilities
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(val, verbose=0).ravel()

    # Prepare labels/predictions
    ytrue = val.classes
    classes = list(val.class_indices.keys())
    preds = (probs >= 0.5).astype(int)

    # Metrics
    acc = float((preds == ytrue).mean())
    # AUC can fail if only one class present in ytrue; guard it.
    try:
        auc = float(roc_auc_score(ytrue, probs))
    except ValueError:
        auc = None

    report_dict = classification_report(
        ytrue, preds, target_names=classes, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(ytrue, preds)

    # Save metrics JSON
    with open(f"reports/metrics_{tag}.json", "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "roc_auc": auc,
                "report": report_dict,
                "classes": classes,
            },
            f,
            indent=2,
        )

    # Plot Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.set_title(f"Confusion Matrix ({tag})")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(classes)
    ax_cm.set_yticklabels(classes)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    fig_cm.tight_layout()
    fig_cm.savefig(f"reports/cm_{tag}.png", dpi=160)
    plt.close(fig_cm)

    # Plot ROC curve (if AUC available)
    if auc is not None:
        fpr, tpr, _ = roc_curve(ytrue, probs)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_title(f"ROC Curve ({tag})")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        fig_roc.tight_layout()
        fig_roc.savefig(f"reports/roc_{tag}.png", dpi=160)
        plt.close(fig_roc)

    print(
        f"[{tag}] accuracy={acc:.4f}",
        f"roc_auc={auc:.4f}" if auc is not None else "roc_auc=N/A",
    )


if __name__ == "__main__":
    # Basic CNN (rescale 1/255)
    evaluate("models/basic_cnn.h5", (128, 128), preproc=None, tag="basic")

    # # VGG16 (use VGG16 preprocessing)
    # evaluate("models/vgg16.h5", (224, 224), preproc=vgg_pre, tag="vgg16")

    # MobileNetV2 (use MobileNetV2 preprocessing)
    evaluate("models/mobilenetv2.h5", (128, 128), preproc=mv2_pre, tag="mobilenetv2")
