"""
Split the labelled golden set into train/val/test, grouped by conversation_id
so no conversation appears in two splits.

Excludes utility labels (not_support_related, insufficient_context) from the
classifier splits — they are not intents. They go to a separate file used
later for escalation analysis.

Writes:
    golden_set/train.csv
    golden_set/val.csv
    golden_set/test.csv
    golden_set/utility.csv
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

SEED = 42
CSV = repo_path("golden_set", "golden_set_to_label.csv")
OUT_DIR = repo_path("golden_set")

UTILITY = {"not_support_related", "insufficient_context"}


def main():
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["intent"])
    df["intent"] = df["intent"].astype(str).str.strip()

    utility = df[df["intent"].isin(UTILITY)].copy()
    real = df[~df["intent"].isin(UTILITY)].copy()
    print(f"Total: {len(df)} | real intents: {len(real)} | utility: {len(utility)}")

    groups = real["conversation_id"].values

    # 70/15/15 via two GroupShuffleSplits
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, rest_idx = next(gss1.split(real, groups=groups))

    rest = real.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(gss2.split(rest, groups=rest["conversation_id"].values))

    train_df = real.iloc[train_idx].copy()
    val_df   = rest.iloc[val_idx].copy()
    test_df  = rest.iloc[test_idx].copy()

    # Leakage assertion — this is a real test, not a comment
    def ids(d): return set(d["conversation_id"])
    assert ids(train_df).isdisjoint(ids(val_df)), "train/val leakage"
    assert ids(train_df).isdisjoint(ids(test_df)), "train/test leakage"
    assert ids(val_df).isdisjoint(ids(test_df)), "val/test leakage"
    print("Leakage check: PASS (no conversation_id overlap)")

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("\nClass distribution:")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name}:")
        print(d["intent"].value_counts().to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUT_DIR / "val.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    utility.to_csv(OUT_DIR / "utility.csv", index=False)
    print(f"\nWrote {OUT_DIR}/train.csv, val.csv, test.csv, utility.csv")


if __name__ == "__main__":
    main()