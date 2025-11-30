"""
Prepare GoEmotions data for public unrest prediction.

- Reads original train/dev/test .tsv files from ../Goemotion/data/
- Uses emotions.txt to map emotion ids -> names
- Uses PublicUnrest/data/emotion_weights.json to compute unrest_percent (0–100)
- Cleans text
- Saves processed CSVs into PublicUnrest/data/processed/
"""

import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ---------- Paths ----------

# This file is in PublicUnrest/src/, so parent().parent() = PublicUnrest/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOEMOTION_DATA_DIR = PROJECT_ROOT.parent / "Goemotion" / "data"
OUR_DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = OUR_DATA_DIR / "processed"

EMOTIONS_FILE = GOEMOTION_DATA_DIR / "emotions.txt"
WEIGHTS_FILE = OUR_DATA_DIR / "emotion_weights.json"

TRAIN_TSV = GOEMOTION_DATA_DIR / "train.tsv"
DEV_TSV = GOEMOTION_DATA_DIR / "dev.tsv"
TEST_TSV = GOEMOTION_DATA_DIR / "test.tsv"


# ---------- Load config: emotions + weights ----------

def load_emotions() -> List[str]:
    if not EMOTIONS_FILE.exists():
        raise FileNotFoundError(f"emotions.txt not found at {EMOTIONS_FILE}")
    lines = EMOTIONS_FILE.read_text(encoding="utf-8").splitlines()
    emotions = [l.strip() for l in lines if l.strip()]
    return emotions


def load_weights() -> dict:
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(f"emotion_weights.json not found at {WEIGHTS_FILE}")
    return json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))


# ---------- Label -> unrest_percent ----------

def labels_to_unrest_percent(
    labels: Optional[str],
    emotions: List[str],
    emo_weights: dict
) -> float:
    """
    Convert comma-separated emotion ids (as used in train/dev/test.tsv)
    into a single unrest percentage in [0, 100].
    """
    if not isinstance(labels, str) or not labels.strip():
        return 0.0

    try:
        ids = [int(x) for x in labels.split(",") if x != ""]
    except ValueError:
        # In case of odd formatting, fall back to 0
        return 0.0

    if not ids:
        return 0.0

    weights = []
    for i in ids:
        if 0 <= i < len(emotions):
            emo = emotions[i]
            w = emo_weights.get(emo, 0.0)
            weights.append(float(w))
    if not weights:
        return 0.0

    unrest_0_1 = sum(weights) / len(weights)
    return unrest_0_1 * 100.0


# ---------- Text cleaning ----------

URL_RE = re.compile(r"http\S*|\S*\.com\S*|\S*www\S*", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"\s@\S+")
PUNCT_CHARS = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
PUNCT_RE = re.compile(f"[{re.escape(PUNCT_CHARS)}]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Roughly follow GoEmotions cleaning: lowercase, strip URLs, mentions, punctuation."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = PUNCT_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t)
    t = t.strip()
    return t


# ---------- Main pipeline ----------

def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} file not found at {path}")

    # GoEmotions train/dev/test.tsv have no header and 3 columns:
    # 1) text  2) comma-separated emotion ids  3) id  :contentReference[oaicite:3]{index=3}
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "labels", "id"],
        encoding="utf-8"
    )
    df["split"] = split_name
    return df


def main() -> None:
    print("Project root:", PROJECT_ROOT)
    print("GoEmotions data dir:", GOEMOTION_DATA_DIR)
    print("Our data dir:", OUR_DATA_DIR)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load configs
    emotions = load_emotions()
    emo_weights = load_weights()
    print("Loaded", len(emotions), "emotions.")
    print("Loaded", len(emo_weights), "emotion weights.")

    # Load splits
    print("Loading train/dev/test tsv files...")
    df_train = load_split(TRAIN_TSV, "train")
    df_dev = load_split(DEV_TSV, "dev")
    df_test = load_split(TEST_TSV, "test")

    df_all = pd.concat([df_train, df_dev, df_test], ignore_index=True)
    print(f"Total rows (all splits): {len(df_all)}")

    # Compute unrest_percent
    print("Computing unrest_percent...")
    df_all["unrest_percent"] = df_all["labels"].apply(
        lambda lbl: labels_to_unrest_percent(lbl, emotions, emo_weights)
    )

    # Clean text
    print("Cleaning text...")
    df_all["text_clean"] = df_all["text"].apply(clean_text)

    # Quick sanity stats
    print("Unrest percent stats:")
    print(df_all["unrest_percent"].describe())

    # Save ALL combined
    all_out = PROCESSED_DIR / "goemotions_unrest_all.csv"
    df_all.to_csv(all_out, index=False, encoding="utf-8")
    print("Saved combined dataset to:", all_out)

    # Save per split for convenience
    for split_name, df_split in df_all.groupby("split"):
        out_path = PROCESSED_DIR / f"goemotions_unrest_{split_name}.csv"
        df_split.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Saved {split_name} split to:", out_path)


if __name__ == "__main__":
    main()
