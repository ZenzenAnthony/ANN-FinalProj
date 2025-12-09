import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

# -----------------------------
# Paths based on your project structure
# -----------------------------
# This file: PublicUnrest/src/evaluate_regression.py
# BASE_DIR : PublicUnrest/
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
PLOTS_DIR = BASE_DIR / "plots" / "regression"

MODEL_PATH = MODELS_DIR / "unrest_model.keras"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
TEST_CSV = DATA_DIR / "goemotions_unrest_test.csv"
HISTORY_PATH = MODELS_DIR / "unrest_history.npy"  # optional, if you saved history

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("Base directory:", BASE_DIR)
print("Model path:", MODEL_PATH)
print("Vectorizer path:", VECTORIZER_PATH)
print("Test CSV:", TEST_CSV)
print("Plots dir:", PLOTS_DIR)

# -----------------------------
# Sanity checks on files
# -----------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(f"Vectorizer file not found at: {VECTORIZER_PATH}")
if not TEST_CSV.exists():
    raise FileNotFoundError(f"Test CSV not found at: {TEST_CSV}")

# -----------------------------
# 1. Load model, vectorizer, and data
# -----------------------------
print("Loading model and vectorizer...")
model = load_model(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Loading test data...")
df_test = pd.read_csv(TEST_CSV)

# Adjust column names here if yours are different
TEXT_COL = "text"
TARGET_COL = "unrest_percent"

if TEXT_COL not in df_test.columns:
    raise KeyError(f"Column '{TEXT_COL}' not found in test CSV. Available: {df_test.columns}")
if TARGET_COL not in df_test.columns:
    raise KeyError(f"Column '{TARGET_COL}' not found in test CSV. Available: {df_test.columns}")

X_text = df_test[TEXT_COL]
y_true = df_test[TARGET_COL].astype(float)

print(f"Test samples: {len(df_test)}")

# Transform text with TF-IDF
X_tfidf = vectorizer.transform(X_text)

# Keras expects dense numpy array (not scipy sparse)
X_input = X_tfidf.toarray()

# Predict
y_pred = model.predict(X_input).reshape(-1)
# If your model outputs shape (n,1), reshape flattens it.

# Basic metrics
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"Test MAE: {mae:.4f}")
print(f"Test R²:  {r2:.4f}")

# -----------------------------
# 2. Loss Curve (if history was saved)
# -----------------------------
if HISTORY_PATH.exists():
    print(f"Loading training history from: {HISTORY_PATH}")
    history = np.load(HISTORY_PATH, allow_pickle=True).item()
    loss = history.get("loss")
    val_loss = history.get("val_loss")

    if loss is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(loss, label="Training Loss", marker="o")
        if val_loss is not None:
            plt.plot(val_loss, label="Validation Loss", marker="x")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_curve_path = PLOTS_DIR / "loss_curve.png"
        plt.savefig(loss_curve_path, dpi=300)
        plt.close()
        print(f"Saved loss curve to: {loss_curve_path}")
    else:
        print("History file found, but no 'loss' key inside.")
else:
    print("No training history file found; skipping loss curve. "
          "You can save history during training to enable this plot.")

# -----------------------------
# 3. Predicted vs Actual Scatter Plot
# -----------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.4, edgecolor="k", linewidth=0.3)
min_val = float(min(y_true.min(), y_pred.min()))
max_val = float(max(y_true.max(), y_pred.max()))
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)  # y = x line

plt.title(f"Predicted vs Actual Unrest (%)\nMAE={mae:.2f}, R²={r2:.2f}")
plt.xlabel("Actual Unrest Percent")
plt.ylabel("Predicted Unrest Percent")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.grid(True, alpha=0.3)
plt.tight_layout()

scatter_path = PLOTS_DIR / "pred_vs_actual_scatter.png"
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Saved predicted vs actual scatter to: {scatter_path}")

# -----------------------------
# 4. Distribution of Target (unrest_percent)
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(y_true, bins=20, alpha=0.8, edgecolor="black")
plt.title("Distribution of Unrest Percent (Test Set)")
plt.xlabel("Unrest Percent")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()

dist_target_path = PLOTS_DIR / "unrest_distribution.png"
plt.savefig(dist_target_path, dpi=300)
plt.close()
print(f"Saved unrest distribution histogram to: {dist_target_path}")

# -----------------------------
# 5. Error Distribution (prediction - true)
# -----------------------------
errors = y_pred - y_true
mean_error = float(np.mean(errors))
std_error = float(np.std(errors))

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, alpha=0.8, edgecolor="black")
plt.axvline(0, color="r", linestyle="--", linewidth=1, label="Zero Error")
plt.axvline(mean_error, color="g", linestyle="-", linewidth=1, label=f"Mean Error = {mean_error:.2f}")

plt.title("Error Distribution (Prediction - Actual)")
plt.xlabel("Error (Predicted - Actual)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

err_hist_path = PLOTS_DIR / "error_distribution.png"
plt.savefig(err_hist_path, dpi=300)
plt.close()
print(f"Saved error distribution histogram to: {err_hist_path}")

# -----------------------------
# 6. Error Boxplot
# -----------------------------
plt.figure(figsize=(4, 6))
plt.boxplot(errors, vert=True, showmeans=True)
plt.title("Error Boxplot")
plt.ylabel("Error (Predicted - Actual)")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

err_boxplot_path = PLOTS_DIR / "error_boxplot.png"
plt.savefig(err_boxplot_path, dpi=300)
plt.close()
print(f"Saved error boxplot to: {err_boxplot_path}")

print("All regression evaluation plots generated.")