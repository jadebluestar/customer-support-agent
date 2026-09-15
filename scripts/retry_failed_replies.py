"""
Retry reply generation for examples that failed with reply_status == "error".
Updates data/reply_eval_agent_output.jsonl in place.
"""
import json
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent import SupportAgent


def main():
    path = "data/reply_eval_agent_output.jsonl"
    with open(path) as f:
        rows = [json.loads(line) for line in f]

    failed = [r for r in rows if r.get("reply_status") == "error"]
    print(f"Retrying {len(failed)} failed examples...")

    agent = SupportAgent()
    for r in failed:
        eid = r["example_id"]
        print(f"  [{eid}] {r['message'][:70]}")
        new = agent.handle(r["message"])
        # replace all pipeline fields
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