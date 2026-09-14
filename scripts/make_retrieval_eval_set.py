"""
Sample 25 queries from the golden set, retrieve top-5 for each, and write
them out for hand-labelling. One row per (query, retrieved) pair.

Output: data/retrieval_labels_TO_FILL.csv
  columns: query_id, query_text, rank, retrieved_conversation_id,
           retrieved_customer, retrieved_reply, relevant
  `relevant` is left blank for you to fill with 1 or 0.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path
from src.retriever import Retriever

GOLDEN = repo_path("golden_set", "golden_set_to_label.csv")
OUT    = repo_path("data", "retrieval_labels_TO_FILL.csv")
N_QUERIES = 25
K = 5
SEED = 42

UTILITY = {"not_support_related", "insufficient_context"}


def main():
    df = pd.read_csv(GOLDEN).dropna(subset=["intent"])
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[~df["intent"].isin(UTILITY)]

    # Stratified: up to 2 per intent, then top up randomly to N_QUERIES
    sampled = []
    for intent, g in df.groupby("intent"):
        sampled.append(g.sample(min(len(g), 2), random_state=SEED))
    sampled = pd.concat(sampled)
    if len(sampled) < N_QUERIES:
        rest = df.drop(sampled.index).sample(
            N_QUERIES - len(sampled), random_state=SEED
        )
        sampled = pd.concat([sampled, rest])

    sampled = sampled.head(N_QUERIES).reset_index(drop=True)
    print(f"Sampled {len(sampled)} queries across {sampled['intent'].nunique()} intents")

    r = Retriever()
    rows = []
    for qi, q in sampled.iterrows():
        hits = r.retrieve(q["text"], k=K)
        for rank, h in enumerate(hits, 1):
            rows.append({
                "query_id": qi,
                "query_text": q["text"],
                "query_intent": q["intent"],
                "rank": rank,
                "retrieved_conversation_id": h.conversation_id,
                "retrieved_customer": h.customer_message,
                "retrieved_reply": h.brand_reply,
                "score": round(h.score, 3),
                "relevant": "",   # <-- fill with 1 or 0
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} — {len(out)} rows to label (25 queries × 5 hits)")


if __name__ == "__main__":
    main()