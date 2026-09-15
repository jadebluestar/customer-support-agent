"""
Grounded reply generation. Reuses the provider plumbing from llm_classifier.py
by importing its `_client`, `_call_raw`, and `LLMResponse`.
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.llm_classifier import _call_raw_with_backoff, LLMResponse


SYSTEM = (
    "You are an AmazonHelp customer-support agent. "
    "You reply grounded ONLY in the evidence provided. "
    "Respond with JSON only."
)

PROMPT = """Draft a reply to the customer.

Intent: {intent}
Customer message: "{message}"

Evidence (historical AmazonHelp resolutions for similar issues):
{evidence}

Rules:
- Under 280 characters.
- Do NOT invent order numbers, refund amounts, delivery dates, policies,
  or account-specific facts. Only use information present in the evidence.
- Use AmazonHelp's voice: brief, apologetic, action-oriented.
- If the evidence does not cover this case, return decision="escalate".

Respond with JSON only:
{{
  "decision": "handle" | "escalate",
  "reply": "<reply text, empty if escalate>",
  "reason": "<short>",
  "unsupported_claims": ["..."]
}}
"""


@dataclass
class GeneratedReply:
    decision: str
    reply: str
    reason: str
    unsupported_claims: list = field(default_factory=list)


def _format_evidence(hits):
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. Customer: {h.customer_message}")
        lines.append(f"   AmazonHelp: {h.brand_reply}")
    return "\n".join(lines)


def generate(message: str, intent: str, hits) -> tuple[GeneratedReply | None, LLMResponse]:
    """
    Returns (GeneratedReply | None, LLMResponse). On failure, GeneratedReply
    is None and the LLMResponse carries the status. Callers must check.
    """
    evidence = _format_evidence(hits) if hits else "(no evidence retrieved)"
    prompt = PROMPT.format(intent=intent, message=message, evidence=evidence)

    payload, resp = _call_raw_with_backoff(prompt, max_tokens=512)
    if not resp.ok or payload is None:
        return None, resp

    try:
        gen = GeneratedReply(
            decision=payload.get("decision", "escalate"),
            reply=payload.get("reply", ""),
            reason=payload.get("reason", ""),
            unsupported_claims=payload.get("unsupported_claims", []) or [],
        )
    except Exception as e:
        return None, LLMResponse(status="invalid_output", error=str(e))

    return gen, resp


if __name__ == "__main__":
    from src.retriever import Retriever

    r = Retriever()
    msg = "my package was supposed to arrive 3 days ago and it's still not here"
    hits = r.retrieve(msg, k=3)
    gen, resp = generate(msg, "delivery_delay_or_missing", hits)
    print(f"status: {resp.status}")
    print(gen)