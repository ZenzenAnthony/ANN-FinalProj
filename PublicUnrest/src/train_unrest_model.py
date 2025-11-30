"""
Train a neural network to predict unrest percentage from GoEmotions text.

- Loads processed CSVs from data/processed/
- Uses TF-IDF features on text_clean
- Trains an MLP regression model (output in [0, 1], scaled to 0-100%)
- Evaluates on test set
- Saves model and vectorizer into models/
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers


# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Helper: load data ----------

def load_splits():
    train_path = DATA_PROCESSED_DIR / "goemotions_unrest_train.csv"
    dev_path = DATA_PROCESSED_DIR / "goemotions_unrest_dev.csv"
    test_path = DATA_PROCESSED_DIR / "goemotions_unrest_test.csv"

    if not train_path.exists() or not dev_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"One or more processed CSVs not found in {DATA_PROCESSED_DIR}. "
            "Make sure prepare_goemotions_unrest.py has been run."
        )

    df_train = pd.read_csv(train_path)
    df_dev = pd.read_csv(dev_path)
    df_test = pd.read_csv(test_path)

    # Prefer cleaned text if available
    text_col = "text_clean" if "text_clean" in df_train.columns else "text"

    X_train_text = df_train[text_col].fillna("")
    X_dev_text = df_dev[text_col].fillna("")
    X_test_text = df_test[text_col].fillna("")

    # Targets: unrest_percent scaled to [0,1]
    y_train = (df_train["unrest_percent"].astype(float) / 100.0).values
    y_dev = (df_dev["unrest_percent"].astype(float) / 100.0).values
    y_test = (df_test["unrest_percent"].astype(float) / 100.0).values

    return (X_train_text, y_train), (X_dev_text, y_dev), (X_test_text, y_test)


# ---------- Helper: build TF-IDF + model ----------

def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=2
    )
    return vectorizer


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),  # output in [0,1]
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",          # regression
        metrics=["mae"],     # mean absolute error in [0,1]
    )
    return model


# ---------- Training pipeline ----------

def main():
    print("Project root:", PROJECT_ROOT)
    print("Processed data dir:", DATA_PROCESSED_DIR)
    print("Models dir:", MODELS_DIR)

    # 1. Load data
    (X_train_text, y_train), (X_dev_text, y_dev), (X_test_text, y_test) = load_splits()
    print(f"Train size: {len(X_train_text)}, Dev size: {len(X_dev_text)}, Test size: {len(X_test_text)}")

    # 2. TF-IDF vectorization
    print("Fitting TF-IDF vectorizer...")
    vectorizer = build_vectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_dev = vectorizer.transform(X_dev_text)
    X_test = vectorizer.transform(X_test_text)

    input_dim = X_train.shape[1]
    print("TF-IDF feature dimension:", input_dim)

    # 3. Build model
    model = build_model(input_dim=input_dim)
    model.summary()

    # 4. Train model
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Training model...")
    history = model.fit(
        X_train.toarray(),
        y_train,
        validation_data=(X_dev.toarray(), y_dev),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )

    # 5. Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test.toarray()).reshape(-1)

    # Convert back to percentages for human-friendly metrics
    y_test_pct = y_test * 100.0
    y_pred_pct = y_pred * 100.0

    mae = mean_absolute_error(y_test_pct, y_pred_pct)
    mse = mean_squared_error(y_test_pct, y_pred_pct)
    r2 = r2_score(y_test_pct, y_pred_pct)

    print("\nTest set performance (on unrest_percent):")
    print(f"MAE: {mae:.2f} percentage points")
    print(f"MSE: {mse:.2f}")
    print(f"R^2: {r2:.4f}")

    # Show a few sample predictions
    results_sample = pd.DataFrame(
        {
            "unrest_true_pct": y_test_pct[:10],
            "unrest_pred_pct": y_pred_pct[:10],
        }
    )
    print("\nSample predictions (first 10):")
    print(results_sample)

    # 6. Save model + vectorizer
    model_path = MODELS_DIR / "unrest_model.keras"
    vec_path = MODELS_DIR / "tfidf_vectorizer.joblib"

    model.save(model_path)  # saves in new Keras format
    joblib.dump(vectorizer, vec_path)

    print("\nSaved model to:", model_path)
    print("Saved TF-IDF vectorizer to:", vec_path)


if __name__ == "__main__":
    main()
