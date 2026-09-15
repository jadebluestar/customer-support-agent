"""
The support pipeline. Deterministic five-stage funnel:
classify -> retrieve -> provisional escalation -> generate -> final escalation.

Not an autonomous agent. No loops, no tool calls, no hidden state.
"""
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.llm_classifier import classify
from src.retriever import Retriever
from src.reply_generator import generate
from src.escalation import should_escalate


class SupportAgent:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()

    def handle(self, message: str) -> dict:
        # 1. Classify
        result, llm_resp = classify(message)
        intent_result = {
            "intent":    result["intent"] if result else None,
            "confidence": result["confidence"] if result else None,
            "status":    llm_resp.status,
        }

        # 2. Retrieve
        hits = self.retriever.retrieve(message, k=3)

        # 3. Provisional escalation before spending a generation call
        escalate, reason = should_escalate(intent_result, hits, None)
        if escalate and reason.split(":")[0] in {
            "classifier_error", "out_of_scope", "sensitive_intent",
            "low_confidence", "no_evidence", "weak_evidence",
        }:
            return {
                "intent": intent_result["intent"],
                "intent_confidence": intent_result["confidence"],
                "intent_status": intent_result["status"],
                "evidence": [asdict(h) for h in hits],
                "retrieval_status": "ok" if hits else "empty",
                "decision": "escalate",
                "reason": reason,
                "reply": None,
                "reply_status": "skipped",
            }

        # 4. Generate grounded reply
        gen, gen_resp = generate(message, intent_result["intent"], hits)

        # 5. Final escalation (generator may override the provisional decision)
        escalate, reason = should_escalate(intent_result, hits, gen)
        return {
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "intent_status": intent_result["status"],
            "evidence": [asdict(h) for h in hits],
            "retrieval_status": "ok" if hits else "empty",
            "decision": "escalate" if escalate else "handle",
            "reason": reason,
            "reply": None if escalate or gen is None else gen.reply,
            "reply_status": (
                "escalated" if escalate else
                "error"     if gen is None else
                "success"
            ),
        }


if __name__ == "__main__":
    agent = SupportAgent()
    for msg in [
        "my package was supposed to arrive 3 days ago and it's still not here",
        "why was I charged $9.99 twice this month",
        "someone hacked my account, orders are appearing",
        "your service is terrible, I'm done",
    ]:
        print("=" * 70)
        print(f"MESSAGE: {msg}\n")
        out = agent.handle(msg)
        for k in ["intent", "intent_confidence", "decision", "reason", "reply"]:
            print(f"  {k}: {out[k]}")
        print()