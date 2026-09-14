"""
Compute Recall@K, MRR, Precision@K from hand-labelled retrieval labels.
Refuses to report if fewer than 20 queries have been fully labelled.
"""
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

LABELS = repo_path("data", "retrieval_labels_TO_FILL.csv")
OUT    = repo_path("reports", "retrieval_eval.md")
MIN_LABELLED_QUERIES = 20


def main():
    df = pd.read_csv(LABELS)
    df["relevant"] = pd.to_numeric(df["relevant"], errors="coerce")

    fully = df.dropna(subset=["relevant"]).groupby("query_id").size()
    n_queries = len(fully)
    if n_queries < MIN_LABELLED_QUERIES:
        print(f"INCOMPLETE: {n_queries} queries labelled, need >= {MIN_LABELLED_QUERIES}")
        return

    df = df.dropna(subset=["relevant"])
    per_query = df.groupby("query_id")

    r1 = r3 = r5 = 0
    mrr_sum = 0.0
    precisions = []

    for qid, g in per_query:
        g = g.sort_values("rank")
        rels = g["relevant"].tolist()
        if rels[0] == 1: r1 += 1
        if any(r == 1 for r in rels[:3]): r3 += 1
        if any(r == 1 for r in rels[:5]): r5 += 1
        first = next((i + 1 for i, r in enumerate(rels) if r == 1), None)
        mrr_sum += (1.0 / first) if first else 0.0
        precisions.append(sum(rels) / len(rels))

    N = n_queries
    metrics = {
        "n_queries": N,
        "recall@1": r1 / N,
        "recall@3": r3 / N,
        "recall@5": r5 / N,
        "mrr": mrr_sum / N,
        "precision@5": sum(precisions) / N,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# Retrieval evaluation\n\n")
        f.write(f"Queries evaluated: {metrics['n_queries']}\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v:.3f} |\n")
        f.write(
            "\n**Relevance definition:** a retrieved case is relevant if (a) the "
            "retrieved customer message describes the same category of problem "
            "as the query, AND (b) the retrieved brand reply shows a resolution "
            "approach appropriate for the query.\n"
        )

    print(f"Wrote {OUT}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()