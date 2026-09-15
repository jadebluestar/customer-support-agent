"""
Retry only the still-failing reply generation examples, with a pause between
calls to avoid Groq's per-minute rate limit.
"""
import json
import time
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent import SupportAgent


def main():
    path = "data/reply_eval_agent_output.jsonl"
    with open(path) as f:
        rows = [json.loads(line) for line in f]

    failed = [r for r in rows if r.get("reply_status") == "error"]
    print(f"Retrying {len(failed)} examples (10s pause between each)...")

    agent = SupportAgent()
    for r in failed:
        eid = r["example_id"]
        print(f"  [{eid}] {r['message'][:70]}")
        time.sleep(10)
        new = agent.handle(r["message"])
        for k, v in new.items():
            r[k] = v
        print(f"      -> {r['reply_status']}  decision={r['decision']}")

    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    remaining = sum(1 for r in rows if r.get("reply_status") == "error")
    print(f"\nDone. Remaining errors: {remaining}/{len(rows)}")


if __name__ == "__main__":
    main()