"""
Recompute CV summary metrics from the cached audit predictions.
No API calls. Reads reports/audit_predictions.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

CACHE = repo_path("reports", "audit_predictions.csv")


def main():
    df = pd.read_csv(CACHE)
    if "status" in df.columns:
        err_rate = (df["status"] != "success").mean()
        print(f"Error rate in cached predictions: {err_rate:.1%}")
        if err_rate > 0.05:
            print("EVALUATION INCOMPLETE — do not report these metrics.")
            return
    print(f"Rows: {len(df)}  Folds: {df['fold'].nunique()}\n")

    # Sanity check: is any fold dominated by the fallback?
    fallback_frac = ((df["pred"] == "general_complaint") &
                     (df["confidence"] == 0.0)).mean()
    print(f"Fallback rate (pred=general_complaint & conf=0.0): {fallback_frac:.1%}")
    if fallback_frac > 0.05:
        print("  WARNING: >5% fallbacks — cached predictions may be poisoned.")
        return

    # Per-fold metrics
    accs, f1s = [], []
    for fold, g in df.groupby("fold"):
        acc = accuracy_score(g["true"], g["pred"])
        f1 = f1_score(g["true"], g["pred"], average="macro", zero_division=0)
        accs.append(acc); f1s.append(f1)
        print(f"  fold {fold}: acc={acc:.3f}  macro_F1={f1:.3f}  (n={len(g)})")

    print(f"\nLLM few-shot (from cached predictions)")
    print(f"  Accuracy  mean={np.mean(accs):.3f}  std={np.std(accs):.3f}")
    print(f"  Macro F1  mean={np.mean(f1s):.3f}  std={np.std(f1s):.3f}")
    print(f"\n  Aggregated per-class (all folds):")
    print(classification_report(df["true"], df["pred"], zero_division=0))

    # Update the summary CSV
    summary_path = repo_path("reports", "baseline_cv_results.csv")
    try:
        summary = pd.read_csv(summary_path)
    except FileNotFoundError:
        summary = pd.DataFrame(columns=["model","acc_mean","acc_std","f1_mean","f1_std"])
    summary = summary[summary["model"] != "LLM few-shot"]  # drop stale row if present
    new = pd.DataFrame([{
        "model": "LLM few-shot",
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
    }])
    summary = pd.concat([summary, new], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nUpdated {summary_path}")


if __name__ == "__main__":
    main()