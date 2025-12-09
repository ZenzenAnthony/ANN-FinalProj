from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots" / "classification"

MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

TRAIN_CSV = DATA_DIR / "goemotions_unrest_train_cls.csv"
DEV_CSV   = DATA_DIR / "goemotions_unrest_dev_cls.csv"

BEST_MODEL_PATH   = MODELS_DIR / "unrest_classifier.keras"
BEST_HISTORY_PATH = MODELS_DIR / "unrest_classifier_history.npy"
BEST_CONFIG_PATH  = MODELS_DIR / "unrest_classifier_config.json"

# -----------------------------
# Load data & vectorizer
# -----------------------------
for f in [TRAIN_CSV, DEV_CSV, VECTORIZER_PATH]:
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")

df_train = pd.read_csv(TRAIN_CSV)
df_dev   = pd.read_csv(DEV_CSV)

TEXT_COL = "text"
LABEL_COL = "unrest_class"

for col in [TEXT_COL, LABEL_COL]:
    if col not in df_train.columns:
        raise KeyError(f"Column '{col}' not found in train CSV. "
                       f"Available: {df_train.columns}")
    if col not in df_dev.columns:
        raise KeyError(f"Column '{col}' not found in dev CSV. "
                       f"Available: {df_dev.columns}")

X_train_text = df_train[TEXT_COL]
y_train = df_train[LABEL_COL].astype(int).values

X_dev_text = df_dev[TEXT_COL]
y_dev = df_dev[LABEL_COL].astype(int).values

vectorizer = joblib.load(VECTORIZER_PATH)

X_train_tfidf = vectorizer.transform(X_train_text).toarray()
X_dev_tfidf   = vectorizer.transform(X_dev_text).toarray()

input_dim = X_train_tfidf.shape[1]
num_classes = len(np.unique(y_train))
print(f"Input dim: {input_dim}, num_classes: {num_classes}")

classes = np.array([0, 1, 2])
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
print("Class weights:", class_weight_dict)


# -----------------------------
# Model builder
# -----------------------------
def build_model(input_dim, hidden_units=128, dropout=0.3, learning_rate=1e-3):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(hidden_units // 2, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(num_classes, activation="softmax")
    ])
    opt = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# -----------------------------
# Hyperparameter grid
# -----------------------------
configs = [
    {"name": "h128_lr1e-3", "hidden": 128, "dropout": 0.3, "lr": 1e-3},
    {"name": "h128_lr3e-4", "hidden": 128, "dropout": 0.3, "lr": 3e-4},
    {"name": "h256_lr1e-3", "hidden": 256, "dropout": 0.4, "lr": 1e-3},
]

best_val_acc = -1.0
best_history = None
best_config = None

for cfg in configs:
    print("\n========================================")
    print(f"Training config: {cfg}")
    print("========================================")

    model = build_model(
        input_dim=input_dim,
        hidden_units=cfg["hidden"],
        dropout=cfg["dropout"],
        learning_rate=cfg["lr"],
    )

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train_tfidf,
        y_train,
        validation_data=(X_dev_tfidf, y_dev),
        epochs=20,
        batch_size=128,
        callbacks=[es],
        class_weight=class_weight_dict,
        verbose=0,
    )



    # Evaluate on dev
    y_dev_pred = np.argmax(model.predict(X_dev_tfidf, verbose=0), axis=1)
    dev_acc = accuracy_score(y_dev, y_dev_pred)
    print(f"Dev accuracy for {cfg['name']}: {dev_acc:.4f}")

    if dev_acc > best_val_acc:
        print(f"--> New best model with dev accuracy {dev_acc:.4f}")
        best_val_acc = dev_acc
        best_history = history.history
        best_config = cfg

        # Save best model & history
        model.save(BEST_MODEL_PATH)
        np.save(BEST_HISTORY_PATH, best_history)
        with open(BEST_CONFIG_PATH, "w") as f:
            json.dump({
                "config": best_config,
                "best_dev_accuracy": best_val_acc
            }, f, indent=2)

print("\n========================================")
print("Hyperparameter search complete.")
print(f"Best dev accuracy: {best_val_acc:.4f}")
print(f"Best config: {best_config}")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"History saved to: {BEST_HISTORY_PATH}")
print(f"Config saved to: {BEST_CONFIG_PATH}")
