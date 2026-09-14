"""
Extracts AmazonHelp conversations from twcs.csv into a clean, flat table:
one row per tweet, grouped by conversation_id, ordered by time.

Does NOT touch the raw dataset in place. Writes a small derived file
(data/amazonhelp_conversations.csv) that we'll actually use going forward,
so we never reload the full 2.8M-row CSV again.

FIX from v1: v1 defined a conversation's root as "walk in_response_to_tweet_id
to the topmost ancestor, whoever that is." That's wrong when the topmost
ancestor is an outbound/brand tweet (e.g. a broadcast or promo) that many
unrelated customers reply to independently -- v1 merged all of them into one
fake "conversation." A valid conversation root must be an INBOUND tweet with
no parent (a customer starting a thread). Any tweet whose walk-up doesn't
land on one of those is dropped, not merged into something else.

Also adds:
- language detection on each conversation's opening message, so we can
  scope the taxonomy work to English only (the raw AmazonHelp data mixes
  English, Italian, Spanish, French, Portuguese, Japanese -- confirmed by
  clustering the unfiltered set).
- a `clean_text` column with URLs and @mentions/anonymized handles stripped,
  so downstream clustering isn't dominated by boilerplate tokens.

Usage:
    python extract_conversations.py
    (first: pip install langdetect)
"""
import os
import re
import sys
from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

CSV_PATH = "/home/jadebluestar/.cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/versions/10/twcs/twcs.csv"
BRAND = "AmazonHelp"
OUT_DIR = repo_path("data")
OUT_PATH = repo_path("data", "amazonhelp_conversations.csv")
MAX_WALK = 30  # cap on how far up the reply chain we walk per tweet; longer than any real thread here

URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")


def clean_text(text: str) -> str:
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_detect_lang(text: str) -> str:
    try:
        return detect(text) if len(text.strip()) >= 3 else "unknown"
    except LangDetectException:
        return "unknown"


def find_roots(df: pd.DataFrame) -> pd.Series:
    """For every tweet, find the ultimate ancestor (root) by walking
    in_response_to_tweet_id backward. Returns a Series indexed by tweet_id.
    Note: this is the RAW topmost ancestor -- validity (inbound, no parent)
    is checked separately in main()."""
    parent = df["in_response_to_tweet_id"]  # Series: tweet_id -> parent tweet_id (or NA)
    valid_ids = set(df.index)

    root = pd.Series(df.index, index=df.index, dtype="Int64")

    for _ in range(MAX_WALK):
        candidate_parent = root.map(parent)  # parent of current pointer
        movable = candidate_parent.notna() & candidate_parent.isin(valid_ids)
        if not movable.any():
            break
        root.loc[movable] = candidate_parent.loc[movable]

    return root


def main():
    print(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(
        CSV_PATH,
        dtype={
            "tweet_id": "int64",
            "author_id": "str",
            "inbound": "bool",
            "text": "str",
            "response_tweet_id": "str",
        },
    )
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype("Int64")
    df["created_at"] = pd.to_datetime(df["created_at"], format="mixed", utc=True, errors="coerce")
    df = df.set_index("tweet_id", drop=False)

    print("Finding conversation roots (this walks the reply graph)...")
    df["raw_root_id"] = find_roots(df)

    # A valid conversation root must itself be an inbound tweet with no parent.
    # Anything whose walk-up lands elsewhere (e.g. a brand broadcast that many
    # unrelated customers replied to) is not a real single conversation -- drop it.
    valid_roots = set(
        df.index[df["inbound"] & df["in_response_to_tweet_id"].isna()]
    )
    df["root_id"] = df["raw_root_id"].where(df["raw_root_id"].isin(valid_roots))
    dropped = df["root_id"].isna().sum()
    print(f"Dropped {dropped:,} tweets with no valid (inbound, parentless) conversation root")
    df = df.dropna(subset=["root_id"])

    print(f"Filtering to conversations that include {BRAND} ...")
    brand_roots = set(df.loc[df["author_id"] == BRAND, "root_id"])
    conv = df[df["root_id"].isin(brand_roots)].copy()

    conv = conv.sort_values(["root_id", "created_at"])
    conv["turn_index"] = conv.groupby("root_id").cumcount()
    conv["speaker"] = conv["inbound"].map({True: "customer", False: "support"})
    conv["clean_text"] = conv["text"].apply(clean_text)

    print("Detecting language of each conversation's opening message...")
    opening = conv[conv["turn_index"] == 0].set_index("root_id")["clean_text"]
    opening_lang = opening.apply(safe_detect_lang)
    conv["opening_lang"] = conv["root_id"].map(opening_lang)

    n_convs = conv["root_id"].nunique()
    n_english = opening_lang.eq("en").sum()
    print(f"{n_english:,} / {n_convs:,} conversations ({n_english / n_convs:.1%}) have an English opening message")

    out_cols = [
        "root_id", "turn_index", "tweet_id", "speaker", "author_id",
        "created_at", "text", "clean_text", "opening_lang",
    ]
    conv = conv[out_cols].rename(columns={"root_id": "conversation_id"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conv.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(conv):,} rows across {conv['conversation_id'].nunique():,} conversations to {OUT_PATH}")

    # --- summary stats ---
    lengths = conv.groupby("conversation_id").size()
    print("\n--- Conversation length distribution (turns) ---")
    print(lengths.describe())
    print("\nPercentiles:")
    print(lengths.quantile([0.5, 0.75, 0.9, 0.95, 0.99]))

    print("\n--- Text length by speaker ---")
    print(conv.groupby("speaker")["text"].apply(lambda s: s.str.len().describe()))

    print("\n--- Timestamp span ---")
    print(f"Earliest: {conv['created_at'].min()}   Latest: {conv['created_at'].max()}")

    print("\n--- 2 example conversations ---")
    sample_ids = lengths[lengths.between(3, 5)].sample(min(2, len(lengths)), random_state=42).index
    for cid in sample_ids:
        print(f"\nConversation {cid}:")
        for _, row in conv[conv["conversation_id"] == cid].iterrows():
            print(f"  [{row['speaker']:8s}] {row['text'][:140]}")


if __name__ == "__main__":
    main()