"""
LLM judge for reply quality. Uses the same Groq provider as the classifier.

Output: data/reply_ratings_judge.csv
Columns: example_id, correctness, relevance, helpfulness, groundedness,
         hallucination (Y/N), escalation_correct (Y/N), notes
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_classifier import _call_raw_with_backoff
from src.project_paths import repo_path


SYSTEM = (
    "You are a strict evaluator of customer-support replies. "
    "Respond with JSON only."
)

RUBRIC = """Rate the following support reply on six dimensions.

Customer message: "{message}"
True intent: {intent}
Retrieved evidence (historical resolutions):
{evidence}

System decision: {decision}
System reply: "{reply}"

Rubric:
- correctness (1-5): does the reply accurately reflect what the evidence shows?
- relevance (1-5): does it address the customer's actual issue?
- helpfulness (1-5): does it move the issue forward?
- groundedness (1-5): is every claim in the reply supported by the evidence?
- hallucination (Y/N): does it invent an order number, refund amount,
  delivery date, policy, or account-specific fact not present in the evidence?
- escalation_correct (Y/N): was the handle/escalate decision right for this
  case? Escalating an obvious automated case or auto-handling a sensitive one
  are both wrong.

Respond with JSON only:
{{
  "correctness": N, "relevance": N, "helpfulness": N, "groundedness": N,
  "hallucination": "Y" | "N", "escalation_correct": "Y" | "N",
  "notes": "<short>"
}}
"""


def main():
    input_path = repo_path("data", "reply_eval_agent_output.jsonl")
    output_path = repo_path("data", "reply_ratings_judge.csv")
    with open(input_path) as f:
        examples = [json.loads(line) for line in f]

    rows = []
    for ex in examples:
        evidence_lines = []
        for i, h in enumerate(ex.get("evidence", []), 1):
            evidence_lines.append(f"{i}. C: {h['customer_message'][:200]}")
            evidence_lines.append(f"   R: {h['brand_reply'][:200]}")
        evidence_text = "\n".join(evidence_lines) or "(no evidence)"

        prompt = RUBRIC.format(
            message=ex["message"],
            intent=ex["intent"],
            evidence=evidence_text,
            decision=ex["decision"],
            reply=ex.get("reply") or "(escalated, no reply)",
        )
        payload, resp = _call_raw_with_backoff(prompt, max_tokens=512)
        if not resp.ok or payload is None:
            print(f"[{ex['example_id']}] judge failed: {resp.status}")
            continue

        rows.append({
            "example_id": ex["example_id"],
            "correctness": payload.get("correctness"),
            "relevance": payload.get("relevance"),
            "helpfulness": payload.get("helpfulness"),
            "groundedness": payload.get("groundedness"),
            "hallucination": payload.get("hallucination"),
            "escalation_correct": payload.get("escalation_correct"),
            "notes": payload.get("notes", ""),
        })
        print(f"[{ex['example_id']}] judged")

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {output_path} ({len(rows)} examples)")


if __name__ == "__main__":
    main()