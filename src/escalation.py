"""
Deterministic escalation policy. Ordered rules; first match wins.
No LLM call — this must be inspectable and cheap.

Sentinel for generation_result:
  "PENDING"  -> provisional pre-generation pass (skip generation checks)
  None       -> generation was attempted and failed -> escalate
  GeneratedReply -> normal post-generation evaluation
"""
import sys
from pathlib import Path

if __package__ in {None, ""}:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from src.thresholds import (
    CONFIDENCE_THRESHOLD,
    RETRIEVAL_THRESHOLD,
    SENSITIVE_INTENTS,
)

OUT_OF_SCOPE = {"not_support_related", "insufficient_context"}


def should_escalate(intent_result, retrieval_hits, generation_result="PENDING"):
    """
    intent_result:     {"intent": str, "confidence": float, "status": str} or None
    retrieval_hits:    list[RetrievedCase]
    generation_result: "PENDING" | GeneratedReply | None

    Returns (escalate: bool, reason: str).
    """
    # 1. Classifier did not succeed — cannot trust any downstream decision.
    if intent_result is None or intent_result.get("status") != "success":
        st = intent_result.get("status") if intent_result else "none"
        return True, f"classifier_error:{st}"

    intent = intent_result["intent"]

    # 2. Utility labels are out of scope.
    if intent in OUT_OF_SCOPE:
        return True, f"out_of_scope:{intent}"

    # 3. Sensitive intents always escalate.
    if intent in SENSITIVE_INTENTS:
        return True, f"sensitive_intent:{intent}"

    # 4. Low classifier confidence.
    if intent_result["confidence"] < CONFIDENCE_THRESHOLD:
        return True, f"low_confidence:{intent_result['confidence']:.2f}"

    # 5. No evidence retrieved.
    if not retrieval_hits:
        return True, "no_evidence"

    # 6. Weak evidence.
    top_score = retrieval_hits[0].score
    if top_score < RETRIEVAL_THRESHOLD:
        return True, f"weak_evidence:{top_score:.2f}"

    # 7. Generation was attempted and failed outright. Cannot handle.
    if generation_result is None:
        return True, "generator_failed"

    # 8. Generator said escalate or flagged unsupported claims.
    if generation_result != "PENDING":
        if generation_result.decision == "escalate":
            return True, f"generator_escalated:{generation_result.reason}"
        if generation_result.unsupported_claims:
            return True, "generator_flagged_unsupported"

    return False, "auto_handle"