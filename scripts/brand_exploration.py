"""
Run this locally against your twcs.csv. It does NOT modify or upload the dataset —
it only prints aggregate stats per candidate brand so we can pick one with real
conversation quality, not just tweet volume.

Usage:
    python brand_exploration.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

CSV_PATH = "/home/jadebluestar/.cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/versions/10/twcs/twcs.csv"

# Candidates spanning different domains (retail/tech/travel/telecom) so we have
# a real choice to justify, not just "biggest number wins"
CANDIDATES = [
    "AmazonHelp", "AppleSupport", "Uber_Support", "SpotifyCares",
    "Delta", "AmericanAir", "TMobileHelp", "comcastcares",
    "AskPlayStation", "ChipotleTweets", "AskPayPal",
]

def main():
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
    # in_response_to_tweet_id -> nullable Int64 to avoid float weirdness
    df["in_response_to_tweet_id"] = df["in_response_to_tweet_id"].astype("Int64")

    df = df.set_index("tweet_id", drop=False)
    parent_map = df["in_response_to_tweet_id"].to_dict()  # tweet_id -> parent_id (or NA)

    def find_root(tweet_id, cache={}):
        path = []
        cur = tweet_id
        while True:
            if cur in cache:
                root = cache[cur]
                break
            parent = parent_map.get(cur, pd.NA)
            if pd.isna(parent) or parent not in parent_map:
                root = cur
                break
            path.append(cur)
            cur = parent
        for t in path:
            cache[t] = root
        return root

    results = []
    for brand in CANDIDATES:
        brand_outbound_ids = set(df.index[(df["author_id"] == brand) & (~df["inbound"])])
        if not brand_outbound_ids:
            continue

        # Sample to keep this fast on 2.8M rows; increase if you want exact numbers later
        sample_ids = list(brand_outbound_ids)[:20000]

        roots = set()
        for tid in sample_ids:
            roots.add(find_root(tid))

        conv_lengths = []
        resolved = 0
        for root in roots:
            chain = [root]
            cur = root
            # walk forward via response_tweet_id (may be comma-separated; take first)
            visited = 0
            while visited < 50:
                resp = df.at[cur, "response_tweet_id"] if cur in df.index else None
                if pd.isna(resp) or resp == "":
                    break
                next_id = int(str(resp).split(",")[0])
                if next_id not in df.index:
                    break
                chain.append(next_id)
                cur = next_id
                visited += 1
            conv_lengths.append(len(chain))
            if len(chain) >= 2:
                resolved += 1

        avg_len = sum(conv_lengths) / len(conv_lengths) if conv_lengths else 0
        multi_turn = sum(1 for l in conv_lengths if l >= 4) / len(conv_lengths) if conv_lengths else 0
        avg_text_len = df.loc[list(sample_ids), "text"].str.len().mean()

        results.append({
            "brand": brand,
            "total_outbound_tweets": len(brand_outbound_ids),
            "sampled_conversations": len(roots),
            "avg_conversation_length": round(avg_len, 2),
            "pct_conversations_multi_turn(>=4)": round(multi_turn * 100, 1),
            "pct_conversations_with_any_reply": round(resolved / len(conv_lengths) * 100, 1) if conv_lengths else 0,
            "avg_text_length_chars": round(avg_text_len, 1),
        })

    out = pd.DataFrame(results).sort_values("avg_conversation_length", ascending=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()