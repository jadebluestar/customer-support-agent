"""
Cross-validated evaluation for the intent classifier.
Runs 5-fold GroupKFold on the 160 real-intent examples.
Compares: majority baseline, TF-IDF+LogReg, LLM few-shot (OpenRouter).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_classifier import classify
from src.project_paths import repo_path

SEED = 42
N_SPLITS = 5
DATA = repo_path("golden_set", "golden_set_to_label.csv")
UTILITY = {"not_support_related", "insufficient_context"}


def load_real():
    df = pd.read_csv(DATA).dropna(subset=["intent"])
    df["intent"] = df["intent"].astype(str).str.strip()
    return df[~df["intent"].isin(UTILITY)].reset_index(drop=True)


def cv_score(name, df, fit_predict_fn):
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = df["conversation_id"].values
    accs, f1s = [], []
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(gkf.split(df, groups=groups), 1):
        train_df, test_df = df.iloc[tr], df.iloc[te]
        y_true = test_df["intent"].values
        y_pred = fit_predict_fn(train_df, test_df)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        accs.append(acc); f1s.append(f1)
        all_true.extend(y_true); all_pred.extend(y_pred)
        print(f"  fold {fold}: acc={acc:.3f}  macro_F1={f1:.3f}  (n={len(test_df)})")

    print(f"\n{name}")
    print(f"  Accuracy  mean={np.mean(accs):.3f}  std={np.std(accs):.3f}")
    print(f"  Macro F1  mean={np.mean(f1s):.3f}  std={np.std(f1s):.3f}")
    print(f"\n  Aggregated per-class (across all folds):")
    print(classification_report(all_true, all_pred, zero_division=0))
    return {"model": name,
            "acc_mean": np.mean(accs), "acc_std": np.std(accs),
            "f1_mean": np.mean(f1s), "f1_std": np.std(f1s)}


def majority_fn(train_df, test_df):
    majority = train_df["intent"].mode()[0]
    return [majority] * len(test_df)


def tfidf_fn(train_df, test_df):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(train_df["text"], train_df["intent"])
    return pipe.predict(test_df["text"])


def llm_fn(train_df, test_df):
    # LLM is zero/few-shot — train_df is ignored
    out = []
    statuses = []
    for i, t in enumerate(test_df["text"], 1):
        result, resp = classify(t)
        statuses.append(resp.status)
        if resp.ok:
            out.append(result["intent"])
        else:
            out.append("__ERROR__")  # will be counted as wrong, but we track it
        print(f"    [llm] {i}/{len(test_df)} -> {out[-1]}  ({resp.status})")

    n_err = sum(1 for s in statuses if s != "success")
    err_rate = n_err / len(statuses)
    if err_rate > 0.05:
        raise RuntimeError(
            f"EVALUATION INCOMPLETE — LLM error rate {err_rate:.1%} "
            f"({n_err}/{len(statuses)}). Do not report these metrics."
        )
    return out


def main():
    os.makedirs("reports", exist_ok=True)

    df = load_real()
    print(f"Real-intent examples: {len(df)}  classes: {df['intent'].nunique()}")
    print(f"Class counts:\n{df['intent'].value_counts().to_string()}\n")

    results = []

    print("=" * 60)
    print("Baseline 1: majority class")
    print("=" * 60)
    results.append(cv_score("Baseline 1 (majority)", df, majority_fn))

    print("\n" + "=" * 60)
    print("Baseline 2: TF-IDF + Logistic Regression")
    print("=" * 60)
    results.append(cv_score("Baseline 2 (TF-IDF+LR)", df, tfidf_fn))

    print("\n" + "=" * 60)
    print("LLM few-shot (OpenRouter)")
    print("=" * 60)
    results.append(cv_score("LLM few-shot", df, llm_fn))

    out_path = repo_path("reports", "baseline_cv_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"{r['model']:25s}  acc={r['acc_mean']:.3f}±{r['acc_std']:.3f}   "
              f"macroF1={r['f1_mean']:.3f}±{r['f1_std']:.3f}")


if __name__ == "__main__":
    main()