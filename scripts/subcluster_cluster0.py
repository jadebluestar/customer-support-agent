"""
Re-clusters Cluster 0 from explore_intents.py at higher resolution, with the
apostrophe tokenization bug fixed (sklearn's default token pattern splits
"don't" into "don" + "t", which is why last time's Cluster 11 was junk).

Reads data/opening_messages_clustered.csv (produced by explore_intents.py).
Writes data/cluster0_subclustered.csv.

Usage:
    python subcluster_cluster0.py
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

IN_PATH = repo_path("data", "opening_messages_clustered.csv")
OUT_PATH = repo_path("data", "cluster0_subclustered.csv")
TARGET_CLUSTER = 0
N_SUBCLUSTERS = 15
RANDOM_STATE = 42

EXTRA_STOPWORDS = [
    "amazon", "amazonhelp", "amp", "http", "https", "prime", "com",
]

ORDER_NUM_RE = re.compile(r"\b\d{4,}[\d-]*\b")

# FIX: default sklearn token pattern (\b\w\w+\b) treats apostrophes as
# separators, so "don't" -> "don" + "t" -> "t" gets dropped as too short,
# leaving orphaned "don". This pattern keeps a single internal apostrophe
# as part of the word: "don't", "can't", "haven't" survive intact.
TOKEN_PATTERN = r"(?u)\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b"


def normalize(text: str) -> str:
    text = text.replace("\u2019", "'") 
    text = ORDER_NUM_RE.sub(" ORDERNUM ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    df = pd.read_csv(IN_PATH)
    sub = df[df["cluster"] == TARGET_CLUSTER].copy()
    sub = sub[sub["clean_text"].str.len() >= 5]
    print(f"{len(sub):,} messages in cluster {TARGET_CLUSTER} to re-cluster")

    sub["norm_text"] = sub["clean_text"].apply(normalize)

    base_stopwords = list(TfidfVectorizer(stop_words="english").get_stop_words())
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words=base_stopwords + EXTRA_STOPWORDS,
        token_pattern=TOKEN_PATTERN,
        min_df=5,
    )
    X = vectorizer.fit_transform(sub["norm_text"])
    terms = vectorizer.get_feature_names_out()

    km = KMeans(n_clusters=N_SUBCLUSTERS, random_state=RANDOM_STATE, n_init=10)
    sub["subcluster"] = km.fit_predict(X)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub[["conversation_id", "clean_text", "subcluster"]].to_csv(OUT_PATH, index=False)
    print(f"Wrote subcluster assignments to {OUT_PATH}\n")

    print("--- Subcluster summaries (sorted by size, largest first) ---")
    order = sub["subcluster"].value_counts().index
    for c in order:
        rows = sub[sub["subcluster"] == c]
        center = km.cluster_centers_[c]
        top_term_idx = center.argsort()[::-1][:8]
        top_terms = ", ".join(terms[i] for i in top_term_idx)
        print(f"\nSubcluster {c} (n={len(rows)}) -- top terms: {top_terms}")
        for txt in rows["clean_text"].head(3):
            print(f"    e.g. {txt[:140]}")


if __name__ == "__main__":
    main()