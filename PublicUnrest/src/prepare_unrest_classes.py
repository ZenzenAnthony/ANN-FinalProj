from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_FILE = DATA_DIR / "goemotions_unrest_train.csv"
DEV_FILE   = DATA_DIR / "goemotions_unrest_dev.csv"
TEST_FILE  = DATA_DIR / "goemotions_unrest_test.csv"

OUT_TRAIN = DATA_DIR / "goemotions_unrest_train_cls.csv"
OUT_DEV   = DATA_DIR / "goemotions_unrest_dev_cls.csv"
OUT_TEST  = DATA_DIR / "goemotions_unrest_test_cls.csv"

for f in [TRAIN_FILE, DEV_FILE, TEST_FILE]:
    if not f.exists():
        raise FileNotFoundError(f"Missing file: {f}")

# -----------------------------
# 1. Compute quantile thresholds on TRAIN ONLY
# -----------------------------
df_train = pd.read_csv(TRAIN_FILE)

if "unrest_percent" not in df_train.columns:
    raise KeyError(f"'unrest_percent' column not found in {TRAIN_FILE}. "
                   f"Available: {df_train.columns}")

unrest_vals = df_train["unrest_percent"].astype(float)

LOW_Q = 0.30
HIGH_Q = 0.70

q1 = unrest_vals.quantile(LOW_Q)
q2 = unrest_vals.quantile(HIGH_Q)

print("Quantile-based thresholds:")
print(f"  q1 (~{int(LOW_Q*100)}rd percentile) = {q1:.2f}")
print(f"  q2 (~{int(HIGH_Q*100)}th percentile) = {q2:.2f}")
print("Class rules:")
print(f"  0 = calm   : unrest_percent <= {q1:.2f}")
print(f"  1 = mild   : {q1:.2f} < unrest_percent <= {q2:.2f}")
print(f"  2 = high   : unrest_percent > {q2:.2f}")

def map_to_class(v: float) -> int:
    if v <= q1:
        return 0  # calm
    elif v <= q2:
        return 1  # mild
    else:
        return 2  # high

# -----------------------------
# 2. Apply to train/dev/test
# -----------------------------
def add_class_column(in_path: Path, out_path: Path):
    df = pd.read_csv(in_path)
    if "unrest_percent" not in df.columns:
        raise KeyError(f"'unrest_percent' column not found in {in_path}. "
                       f"Available: {df.columns}")
    df["unrest_class"] = df["unrest_percent"].astype(float).apply(map_to_class)
    df.to_csv(out_path, index=False)
    print(f"Saved with 'unrest_class' to: {out_path}")
    print(df["unrest_class"].value_counts(normalize=True).sort_index()
          .rename("class_proportion"))

add_class_column(TRAIN_FILE, OUT_TRAIN)
add_class_column(DEV_FILE, OUT_DEV)
add_class_column(TEST_FILE, OUT_TEST)

print("Done. Use *_cls.csv files for classification.")
