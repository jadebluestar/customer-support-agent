"""
Clusters English-only AmazonHelp opening customer messages to surface
real intent candidates from actual data (not invented ones).

Reads data/amazonhelp_conversations.csv (produced by extract_conversations.py).
Writes data/opening_messages_clustered.csv so you can skim cluster
assignments yourself, not just trust printed samples.

Usage:
    python explore_intents.py
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

IN_PATH = repo_path("data", "amazonhelp_conversations.csv")
OUT_PATH = repo_path("data", "opening_messages_clustered.csv")
N_CLUSTERS = 25
RANDOM_STATE = 42

# Generic brand/boilerplate terms that don't distinguish one intent from
# another -- without these, TF-IDF gets dominated by "amazon"/"amazonhelp"
# the same way it did before (see prior cluster 3, 59% of the data).
EXTRA_STOPWORDS = [
    "amazon", "amazonhelp", "amp", "http", "https", "prime", "com",
]

ORDER_NUM_RE = re.compile(r"\b\d{4,}[\d-]*\b")


def normalize(text: str) -> str:
    text = ORDER_NUM_RE.sub(" ORDERNUM ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    conv = pd.read_csv(IN_PATH)

    opening = conv[(conv["turn_index"] == 0) & (conv["opening_lang"] == "en")].copy()
    opening = opening[opening["clean_text"].str.len() >= 5]  # drop near-empty texts
    print(f"{len(opening):,} English opening messages to cluster")

    opening["norm_text"] = opening["clean_text"].apply(normalize)

    base_stopwords = list(TfidfVectorizer(stop_words="english").get_stop_words())
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words=base_stopwords + EXTRA_STOPWORDS,
        min_df=5,
    )
    X = vectorizer.fit_transform(opening["norm_text"])
    terms = vectorizer.get_feature_names_out()

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    opening["cluster"] = km.fit_predict(X)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    opening[["conversation_id", "clean_text", "cluster"]].to_csv(OUT_PATH, index=False)
    print(f"Wrote cluster assignments to {OUT_PATH}\n")

    print("--- Cluster summaries (sorted by size, largest first) ---")
    order = opening["cluster"].value_counts().index
    for c in order:
        sub = opening[opening["cluster"] == c]
        center = km.cluster_centers_[c]
        top_term_idx = center.argsort()[::-1][:8]
        top_terms = ", ".join(terms[i] for i in top_term_idx)
        print(f"\nCluster {c} (n={len(sub)}) -- top terms: {top_terms}")
        for txt in sub["clean_text"].head(3):
            print(f"    e.g. {txt[:140]}")


if __name__ == "__main__":
    main()