"""
Build the retrieval corpus: (customer_message, brand_reply) pairs from
historical AmazonHelp conversations, excluding any conversation in the
golden set. Embeds each customer_message with all-MiniLM-L6-v2.

Outputs:
    data/retrieval_corpus.csv       — conversation_id, customer_message, brand_reply
    data/retrieval_embeddings.npy   — float32, (N, 384), row-aligned with corpus
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

CONV_PATH    = repo_path("data", "amazonhelp_conversations.csv")
GOLDEN_PATH  = repo_path("golden_set", "golden_set_to_label.csv")
OUT_CSV      = repo_path("data", "retrieval_corpus.csv")
OUT_EMB      = repo_path("data", "retrieval_embeddings.npy")
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
MAX_PAIRS    = 25_000


def build_pairs(conv_path, exclude_ids):
    conv = pd.read_csv(conv_path)
    conv = conv[conv["opening_lang"] == "en"]
    conv = conv.sort_values(["conversation_id", "turn_index"])

    pairs = []
    for cid, g in conv.groupby("conversation_id", sort=False):
        if cid in exclude_ids:
            continue
        rows = g.reset_index(drop=True)
        for i in range(len(rows) - 1):
            if rows.at[i, "speaker"] == "customer" and rows.at[i + 1, "speaker"] == "support":
                cm = str(rows.at[i, "clean_text"])
                br = str(rows.at[i + 1, "clean_text"])
                if len(cm) >= 10 and len(br) >= 5:
                    pairs.append({
                        "conversation_id": cid,
                        "customer_message": cm,
                        "brand_reply": br,
                    })
                    if len(pairs) >= MAX_PAIRS:
                        return pd.DataFrame(pairs)
    return pd.DataFrame(pairs)


def main():
    golden = pd.read_csv(GOLDEN_PATH)
    exclude = set(golden["conversation_id"].unique())
    print(f"Golden conversations to exclude: {len(exclude)}")

    print("Building pairs...")
    pairs = build_pairs(CONV_PATH, exclude)
    print(f"  {len(pairs):,} pairs")

    # Leakage assertion — this must pass or we stop.
    overlap = set(pairs["conversation_id"]).intersection(exclude)
    assert not overlap, f"LEAKAGE: {len(overlap)} golden conversations in corpus"
    print("Leakage check: PASS")

    print(f"Embedding {len(pairs):,} customer messages with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        pairs["customer_message"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so cosine = dot product
    ).astype("float32")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUT_CSV, index=False)
    np.save(OUT_EMB, embeddings)
    print(f"Wrote {OUT_CSV} and {OUT_EMB}  ({embeddings.shape})")


if __name__ == "__main__":
    main()