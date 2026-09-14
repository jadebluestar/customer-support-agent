"""
Build a stratified ~200-example golden set for hand-labelling.

Inputs:
    data/amazonhelp_conversations.csv       (from extract_conversations.py)
    data/opening_messages_clustered.csv     (from explore_intents.py)

Outputs:
    golden_set/golden_set_to_label.csv      (200 rows; intent/is_ambiguous/is_multi_intent/notes blank)
    golden_set/sampling_manifest.json       (records every sampling decision, seed, counts)

Design decisions (add to your decision log):
    D-16. Golden set = 200 examples, sampled from conversation openings only
          (turn_index==0), because that is what a deployed system sees first.
    D-17. Stratified, not random: normal / short / noisy / ambiguous /
          multi_intent / low_context.
    D-18. AmazonHelp, English-only.
    D-19. Every example carries conversation_id, so leakage-safe splitting
          downstream is mechanical.
    D-20. `cluster_hint` is a hint, not ground truth — the labeler decides.
    D-21. Seed = 42 for reproducibility.
"""
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

SEED = 42
STRATA = {
    "normal":        80,
    "short":         30,
    "noisy":         30,
    "ambiguous":     20,
    "multi_intent":  20,
    "low_context":   20,
}

CONV_PATH    = repo_path("data", "amazonhelp_conversations.csv")
CLUSTER_PATH = repo_path("data", "opening_messages_clustered.csv")
OUT_DIR      = repo_path("golden_set")
OUT_CSV      = repo_path("golden_set", "golden_set_to_label.csv")
OUT_MANIFEST = repo_path("golden_set", "sampling_manifest.json")

URL_RE   = re.compile(r"https?://\S+")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF]+",
    flags=re.UNICODE,
)

MULTI_INTENT_HINTS  = [" and ", " also ", " but ", " plus ", "&"]
LOW_CONTEXT_CUES    = [
    "this is broken", "it's broken", "its broken", "again",
    "still not", "same problem", "not working", "doesn't work", "doesnt work",
]
ISSUE_KEYWORDS = [
    "refund", "charge", "delivery", "order", "account", "login",
    "app", "return", "package", "payment", "cancel", "damaged",
    "locked", "tracking",
]


def is_short(text, max_len=30):
    return len(text.strip()) < max_len


def is_noisy(text):
    if URL_RE.search(text):
        return True
    if EMOJI_RE.search(text):
        return True
    letters = sum(c.isalpha() for c in text)
    return letters > 0 and (letters / max(len(text), 1)) < 0.6


def looks_multi_intent(text):
    t = text.lower()
    if not any(h in t for h in MULTI_INTENT_HINTS):
        return False
    return sum(k in t for k in ISSUE_KEYWORDS) >= 2


def looks_low_context(text):
    t = text.lower().strip()
    if len(t) > 60:
        return False
    return any(c in t for c in LOW_CONTEXT_CUES)


def classify_stratum(text):
    # Priority order matters so each example lands in exactly one bucket.
    if is_short(text):            return "short"
    if looks_low_context(text):   return "low_context"
    if looks_multi_intent(text):  return "multi_intent"
    if is_noisy(text):            return "noisy"
    return "normal"


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading conversations ...")
    conv = pd.read_csv(CONV_PATH)
    print(f"  {len(conv):,} rows, {conv['conversation_id'].nunique():,} conversations")

    openings = conv[(conv["turn_index"] == 0) & (conv["opening_lang"] == "en")].copy()
    openings = openings[openings["clean_text"].fillna("").str.len() >= 5]
    print(f"  {len(openings):,} eligible English openings")

    if os.path.exists(CLUSTER_PATH):
        cl = pd.read_csv(CLUSTER_PATH).set_index("conversation_id")["cluster"]
        openings["cluster_hint"] = openings["conversation_id"].map(cl)
        print(f"  attached cluster_hint for {openings['cluster_hint'].notna().sum():,} openings")
    else:
        openings["cluster_hint"] = pd.NA

    openings["stratum"] = openings["clean_text"].apply(classify_stratum)
    print("\n  Stratum sizes available:")
    for name, count in openings["stratum"].value_counts().items():
        print(f"    {name}: {count:,}")

    # The ambiguous bucket is not heuristic-detected; we deliberately pull
    # from cluster boundaries: openings whose cluster_hint sits in the
    # smallest clusters (where centroid confidence is lowest).
    sampled = []
    for stratum, n in STRATA.items():
        pool = openings[openings["stratum"] == stratum].copy()
        if pool.empty:
            print(f"  [warn] empty stratum '{stratum}', skipping")
            continue
        # Shuffle then stable-sort by cluster so we spread across clusters.
        pool = pool.sample(frac=1, random_state=SEED)
        if pool["cluster_hint"].notna().any():
            pool = pool.sort_values("cluster_hint", kind="stable", na_position="last")
        sampled.append(pool.head(n))
        print(f"  sampled {min(n, len(pool)):,} for '{stratum}'")

    # For the "ambiguous" stratum specifically, pull from the smallest clusters
    # (lowest confidence boundaries) instead of general pool.
    amb_pool = openings[openings["stratum"] == "normal"].copy()
    if amb_pool["cluster_hint"].notna().any():
        size_by_cluster = amb_pool["cluster_hint"].value_counts()
        smallest_clusters = size_by_cluster.tail(5).index.tolist()
        amb_pool = amb_pool[amb_pool["cluster_hint"].isin(smallest_clusters)]
        amb_pool = amb_pool.sample(frac=1, random_state=SEED)
        # Remove any already sampled conv ids
        chosen = set(pd.concat(sampled, ignore_index=True)["conversation_id"])
        amb_pool = amb_pool[~amb_pool["conversation_id"].isin(chosen)]
        sampled.append(amb_pool.head(STRATA["ambiguous"]))
        print(f"  sampled {min(STRATA['ambiguous'], len(amb_pool)):,} for 'ambiguous' (from boundary clusters)")

    golden = pd.concat(sampled, ignore_index=True)
    golden = golden.drop_duplicates(subset=["conversation_id"]).reset_index(drop=True)

    out = golden[[
        "conversation_id", "tweet_id", "clean_text", "cluster_hint", "stratum",
    ]].rename(columns={"clean_text": "text"}).copy()
    out["intent"]          = ""
    out["is_ambiguous"]    = ""
    out["is_multi_intent"] = ""
    out["notes"]           = ""

    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_CSV}")

    manifest = {
        "seed": SEED,
        "target_total": sum(STRATA.values()),
        "actual_total": len(out),
        "strata": STRATA,
        "stratum_counts": out["stratum"].value_counts().to_dict(),
        "cluster_hint_coverage": int(out["cluster_hint"].notna().sum()),
        "source": CONV_PATH,
        "notes": [
            "Sampled from conversation openings (turn_index=0), English-only.",
            "Each example carries conversation_id for leakage-safe splitting.",
            "cluster_hint is a hint, not ground truth — labeler decides intent.",
            "'ambiguous' stratum drawn from the 5 smallest clusters as a",
            "  proxy for centroid-boundary uncertainty, not from the general pool.",
        ],
    }
    with open(OUT_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {OUT_MANIFEST}")

    print("\n--- Sample preview ---")
    for _, r in out.head(10).iterrows():
        print(f"  [{r['stratum']:12s}] {r['text'][:120]}")


if __name__ == "__main__":
    main()