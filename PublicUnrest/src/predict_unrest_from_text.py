"""
Simple CLI interface to classify new text into unrest levels.

Usage (from PublicUnrest root, with venv active):
    python src/predict_unrest_from_text.py
"""

from pathlib import Path

import joblib
import numpy as np
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

CLASSIFIER_PATH = MODELS_DIR / "unrest_classifier.keras"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"

CLASS_LABELS = {
    0: "calm",
    1: "mild unrest",
    2: "high unrest",
}


def load_model_and_vectorizer():
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Classifier model not found at {CLASSIFIER_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {VECTORIZER_PATH}")

    print("Loading classifier from:", CLASSIFIER_PATH)
    model = keras.models.load_model(CLASSIFIER_PATH)

    print("Loading TF-IDF vectorizer from:", VECTORIZER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


def predict_unrest(text: str, model, vectorizer):
    # Basic cleaning (very light; your training pipeline did more)
    text = text.strip()
    if not text:
        return None

    X = vectorizer.transform([text])
    probs = model.predict(X.toarray(), verbose=0)[0]
    pred_class = int(np.argmax(probs))

    return pred_class, probs


def main():
    model, vectorizer = load_model_and_vectorizer()

    print("\nPublic Unrest Classifier")
    print("Type a sentence and press Enter to classify it.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        user_input = input("Text> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        result = predict_unrest(user_input, model, vectorizer)
        if result is None:
            print("Please enter non-empty text.")
            continue

        pred_class, probs = result
        label = CLASS_LABELS.get(pred_class, str(pred_class))

        print(f"Predicted class: {pred_class} ({label})")
        print("Probabilities:")
        for idx, p in enumerate(probs):
            print(f"  {idx} ({CLASS_LABELS.get(idx)}): {p:.3f}")
        print()


if __name__ == "__main__":
    main()
