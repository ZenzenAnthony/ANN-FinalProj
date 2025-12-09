from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots" / "classification"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CSV = DATA_DIR / "goemotions_unrest_test_cls.csv"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "unrest_classifier.keras"
HISTORY_PATH = MODELS_DIR / "unrest_classifier_history.npy"
CONFIG_PATH = MODELS_DIR / "unrest_classifier_config.json"

for f in [TEST_CSV, VECTORIZER_PATH, MODEL_PATH]:
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")

# -----------------------------
# Load data
# -----------------------------
df_test = pd.read_csv(TEST_CSV)

TEXT_COL = "text"
LABEL_COL = "unrest_class"

if TEXT_COL not in df_test.columns or LABEL_COL not in df_test.columns:
    raise KeyError(f"Missing '{TEXT_COL}' or '{LABEL_COL}' in test CSV. "
                   f"Available: {df_test.columns}")

X_text = df_test[TEXT_COL]
y_true = df_test[LABEL_COL].astype(int).values

vectorizer = joblib.load(VECTORIZER_PATH)
X_tfidf = vectorizer.transform(X_text).toarray()

model = load_model(MODEL_PATH)

# -----------------------------
# Predictions & metrics
# -----------------------------
y_prob = model.predict(X_tfidf, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"Test accuracy: {acc:.4f}\n")

print("Classification report:")
print(classification_report(
    y_true, y_pred,
    target_names=["calm (0)", "mild (1)", "high (2)"]
))

# -----------------------------
# Confusion matrix plot
# -----------------------------
cm = confusion_matrix(y_true, y_pred)
classes = ["calm (0)", "mild (1)", "high (2)"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest")
ax.figure.colorbar(im, ax=ax)
ax.set(
    xticks=np.arange(len(classes)),
    yticks=np.arange(len(classes)),
    xticklabels=classes,
    yticklabels=classes,
    ylabel="True label",
    xlabel="Predicted label",
    title="Confusion Matrix (Unrest Classes)"
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate cells
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, format(cm[i, j], "d"),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

fig.tight_layout()
cm_path = PLOTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved confusion matrix plot to: {cm_path}")

# -----------------------------
# Loss / accuracy curve (if history exists)
# -----------------------------
if HISTORY_PATH.exists():
    history = np.load(HISTORY_PATH, allow_pickle=True).item()
    loss = history.get("loss")
    val_loss = history.get("val_loss")
    acc_hist = history.get("accuracy")
    val_acc_hist = history.get("val_accuracy")

    epochs = range(1, len(loss) + 1) if loss is not None else None

    if loss is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, loss, label="Train loss")
        if val_loss is not None:
            plt.plot(epochs, val_loss, label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Classifier Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        loss_path = PLOTS_DIR / "classifier_loss_curve.png"
        plt.savefig(loss_path, dpi=300)
        plt.close()
        print(f"Saved loss curve to: {loss_path}")

    if acc_hist is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, acc_hist, label="Train acc")
        if val_acc_hist is not None:
            plt.plot(epochs, val_acc_hist, label="Val acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Classifier Accuracy Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        acc_path = PLOTS_DIR / "classifier_accuracy_curve.png"
        plt.savefig(acc_path, dpi=300)
        plt.close()
        print(f"Saved accuracy curve to: {acc_path}")
else:
    print("No history file found; skipping loss/accuracy plots.")
