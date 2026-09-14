"""
Run the agent on 20 golden examples and save the full output for
judging and hand-rating.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import SupportAgent
from src.project_paths import repo_path

GOLDEN = repo_path("golden_set", "golden_set_to_label.csv")
OUT    = repo_path("data", "reply_eval_agent_output.jsonl")
N      = 20
SEED   = 42

UTILITY = {"not_support_related", "insufficient_context"}


def main():
    df = pd.read_csv(GOLDEN).dropna(subset=["intent"])
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[~df["intent"].isin(UTILITY)]

    # Stratified sample: up to 2 per intent, then top up randomly
    sampled = []
    for intent, g in df.groupby("intent"):
        sampled.append(g.sample(min(len(g), 2), random_state=SEED))
    sampled = pd.concat(sampled)
    if len(sampled) > N:
        sampled = sampled.sample(N, random_state=SEED)
    sampled = sampled.head(N).reset_index(drop=True)

    print(f"Sampled {len(sampled)} examples across "
          f"{sampled['intent'].nunique()} intents")

    agent = SupportAgent()
    out = []
    for i, row in sampled.iterrows():
        print(f"[{i+1}/{len(sampled)}] {str(row['text'])[:70]}")
        result = agent.handle(row["text"])
        out.append({
            "example_id": i,
            "message": row["text"],
            "true_intent": row["intent"],
            **result,
        })

    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"Wrote {OUT} — {len(out)} examples")


if __name__ == "__main__":
    main()