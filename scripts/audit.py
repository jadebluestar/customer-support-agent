"""
Experiment audit. Answers:
  #9  confusion matrix (aggregated across 5 folds)
  #10 15 misclassified examples with true/pred + text
  #11 definitions and training examples for the 3 package intents
  #12 whether the 3 package intents are genuinely distinct
  #13 per-intent support, flagging too-thin classes
  #5  cross-conversation near-duplicate check
"""
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_classifier import classify, ALLOWED, DEFS
from src.project_paths import repo_path
from sklearn.metrics import confusion_matrix

DATA = repo_path("golden_set", "golden_set_to_label.csv")
UTILITY = {"not_support_related", "insufficient_context"}
SEED = 42
N_SPLITS = 5


def load_real():
    df = pd.read_csv(DATA).dropna(subset=["intent"])
    df["intent"] = df["intent"].astype(str).str.strip()
    return df[~df["intent"].isin(UTILITY)].reset_index(drop=True)


def collect_predictions(df):
    """Run LLM on every fold's held-out set, collect (true, pred, text, conv_id)."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    groups = df["conversation_id"].values
    rows = []
    errors = 0

    for fold, (_, te) in enumerate(gkf.split(df, groups=groups), 1):
        for _, r in df.iloc[te].iterrows():
            result, resp = classify(r["text"])
            if not resp.ok:
                errors += 1
            rows.append({
                "fold": fold,
                "conversation_id": r["conversation_id"],
                "text": r["text"],
                "true": r["intent"],
                "pred": result["intent"] if resp.ok else "__ERROR__",
                "confidence": result["confidence"] if resp.ok else None,
                "reasoning": result.get("reasoning") if resp.ok else resp.error,
                "ambiguous": r.get("is_ambiguous"),
                "status": resp.status,
            })

    err_rate = errors / len(rows)
    print(f"\nLLM error rate: {err_rate:.1%} ({errors}/{len(rows)})")
    if err_rate > 0.05:
        raise RuntimeError("Audit incomplete — error rate too high.")
    return pd.DataFrame(rows)


def q9_confusion(pred_df):
    labels = sorted(set(pred_df["true"].unique()) | set(pred_df["pred"].unique()))
    cm = confusion_matrix(pred_df["true"], pred_df["pred"], labels=labels)
    print("\n#9 CONFUSION MATRIX (rows=true, cols=pred)")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string())
    print("\nPer-intent accuracy (diag / row sum):")
    for l in labels:
        row = cm_df.loc[l]
        total = row.sum()
        correct = row[l]
        print(f"  {l:35s}  {correct}/{total}")


def q10_misclassified(pred_df, n_per_class=5):
    print("\n#10 MISCLASSIFIED EXAMPLES")
    wrong = pred_df[pred_df["true"] != pred_df["pred"]]
    print(f"Total misclassified: {len(wrong)}/{len(pred_df)} "
          f"({len(wrong)/len(pred_df):.1%})\n")

    focus = ["how_to_or_feature_question", "lost_or_stolen_package",
             "delivery_delay_or_missing", "order_status_or_tracking"]
    for cls in focus:
        sub = wrong[wrong["true"] == cls].head(n_per_class)
        print(f"--- true={cls} (n_wrong={len(wrong[wrong['true']==cls])}) ---")
        for _, r in sub.iterrows():
            print(f"  pred={r['pred']:32s} conf={r['confidence']}")
            print(f"    text: {str(r['text'])[:180]}")
        print()


def q11_definitions_and_examples(df):
    print("\n#11 DEFINITIONS + TRAINING EXAMPLES FOR 3 PACKAGE INTENTS")
    # Print the block of DEFS relevant to the three intents
    for line in DEFS.strip().splitlines():
        if any(k in line for k in ["delivery_delay", "lost_or_stolen",
                                    "order_status", "wrong_or_damaged"]):
            print(f"  DEF: {line}")
    print()
    for cls in ["delivery_delay_or_missing", "lost_or_stolen_package",
                "order_status_or_tracking"]:
        print(f"--- {cls} ---")
        for _, r in df[df["intent"] == cls].head(5).iterrows():
            print(f"  {str(r['text'])[:180]}")
        print()


def q12_are_package_intents_distinct(df):
    print("\n#12 OVERLAP BETWEEN PACKAGE INTENTS")
    # Compare vocabulary / keyword overlap in the example text
    import re
    buckets = {
        "delivery_delay_or_missing": df[df["intent"] == "delivery_delay_or_missing"]["text"],
        "lost_or_stolen_package":    df[df["intent"] == "lost_or_stolen_package"]["text"],
        "order_status_or_tracking":  df[df["intent"] == "order_status_or_tracking"]["text"],
    }
    keyword_sets = {}
    for name, texts in buckets.items():
        tokens = set()
        for t in texts:
            tokens.update(re.findall(r"[a-z]+", str(t).lower()))
        keyword_sets[name] = tokens
        print(f"  {name}: {len(texts)} examples, {len(tokens)} unique tokens")

    print("\n  Pairwise Jaccard similarity of token sets:")
    names = list(buckets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = keyword_sets[names[i]], keyword_sets[names[j]]
            jac = len(a & b) / len(a | b)
            print(f"    {names[i]} vs {names[j]}: {jac:.3f}")
    print("\n  (Jaccard > 0.4 suggests heavy overlap; < 0.25 suggests distinct)")


def q13_thin_classes(df):
    print("\n#13 CLASS SUPPORT")
    counts = df["intent"].value_counts()
    too_thin = counts[counts < 5]
    borderline = counts[(counts >= 5) & (counts < 10)]
    print(f"  <5 examples (per-class metrics meaningless):")
    for k, v in too_thin.items():
        print(f"    {k}: {v}")
    print(f"  5-9 examples (unreliable):")
    for k, v in borderline.items():
        print(f"    {k}: {v}")


def q5_near_duplicates(df):
    print("\n#5 CROSS-CONVERSATION NEAR-DUPLICATE CHECK")
    # Very cheap: shared text prefix (first 40 chars) appearing in >1 conversation
    df = df.copy()
    df["prefix"] = df["text"].astype(str).str.lower().str[:40]
    dup = df.groupby("prefix")["conversation_id"].nunique()
    dup = dup[dup > 1]
    print(f"  prefixes with >1 conversation_id: {len(dup)}")
    for p in dup.head(10).index:
        print(f"    '{p}'")
    if len(dup) == 0:
        print("  No duplicate prefixes found. Contamination risk is low.")


def main():
    df = load_real()
    print(f"Real-intent examples: {len(df)}")
    q13_thin_classes(df)

    # Cache predictions once — do NOT re-run LLM per question
    cache_path = repo_path("reports", "audit_predictions.csv")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(cache_path):
        print(f"\nUsing cached predictions: {cache_path}")
        pred_df = pd.read_csv(cache_path)
    else:
        print("\nRunning LLM on all 5 folds (this is the expensive step)...")
        pred_df = collect_predictions(df)
        pred_df.to_csv(cache_path, index=False)
        print(f"Wrote {cache_path}")

    q9_confusion(pred_df)
    q10_misclassified(pred_df)
    q11_definitions_and_examples(df)
    q12_are_package_intents_distinct(df)
    q5_near_duplicates(df)


if __name__ == "__main__":
    main()