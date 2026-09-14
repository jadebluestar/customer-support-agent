"""
One-shot dataset inspection for the Kaggle Customer Support on Twitter dataset.
Run:  python inspect_data.py
"""
import os
import glob
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import kagglehub

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.project_paths import repo_path

# ----------------------------------------------------------------------
# 1. Locate the dataset (uses cache if already downloaded)
# ----------------------------------------------------------------------
dataset_path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
print(f"[info] dataset_path = {dataset_path}")

# Find the CSV file inside (usually twcs.csv)
csv_candidates = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
print(f"[info] csv candidates = {csv_candidates}")

if not csv_candidates:
    raise FileNotFoundError("No CSV found in dataset directory.")

# Prefer twcs.csv if present
csv_path = next((p for p in csv_candidates if os.path.basename(p) == "twcs.csv"), csv_candidates[0])
print(f"[info] using csv_path = {csv_path}")
print(f"[info] size (MB) = {os.path.getsize(csv_path) / 1e6:.1f}")

# ----------------------------------------------------------------------
# 2. Small sample inspection
# ----------------------------------------------------------------------
print("\n================ SAMPLE (first 10,000 rows) ================")
df = pd.read_csv(csv_path, nrows=10_000)

print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DTYPES ---")
print(df.dtypes)

print("\n--- HEAD 20 ---")
print(df.head(20).to_string())

print("\n--- NULLS ---")
print(df.isna().sum())

print("\n--- UNIQUE COUNTS ---")
for col in df.columns:
    print(f"  {col}: {df[col].nunique(dropna=False)}")

for col in ["inbound", "author_id", "tweet_id", "response_tweet_id", "in_response_to_tweet_id"]:
    if col in df.columns:
        print(f"\n--- {col} value counts (head 20) ---")
        print(df[col].value_counts(dropna=False).head(20))

# Show a few thread-link rows explicitly
link_cols = [c for c in ["tweet_id", "in_response_to_tweet_id", "response_tweet_id"] if c in df.columns]
if link_cols:
    print("\n--- THREAD LINK SAMPLE ---")
    print(df[link_cols].head(20).to_string())

# ----------------------------------------------------------------------
# 3. Chunked full-file scan (memory-safe)
# ----------------------------------------------------------------------
print("\n================ FULL FILE SCAN (chunked) ================")
chunksize = 200_000
brand_counts = Counter()
inbound_counts = Counter()
null_counts = Counter()
total = 0

for chunk in pd.read_csv(csv_path, chunksize=chunksize):
    total += len(chunk)

    if "author_id" in chunk.columns:
        brand_counts.update(chunk["author_id"].dropna().astype(str))

    if "inbound" in chunk.columns:
        inbound_counts.update(chunk["inbound"].dropna().astype(str))

    for col in chunk.columns:
        null_counts[col] += int(chunk[col].isna().sum())

print(f"\nTotal rows: {total}")
print(f"\nInbound counts: {dict(inbound_counts)}")
print(f"\nNull counts: {dict(null_counts)}")

print("\nTop 30 author_ids (potential brands):")
for author, cnt in brand_counts.most_common(30):
    print(f"  {author}: {cnt}")

# Save summary to disk so we can refer back to it
summary = {
    "csv_path": csv_path,
    "total_rows": total,
    "inbound_counts": dict(inbound_counts),
    "null_counts": dict(null_counts),
    "top_authors": brand_counts.most_common(50),
    "columns": df.columns.tolist(),
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
}
summary_path = repo_path("inspect_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[info] wrote {summary_path}")