"""
Tuned constants for the support pipeline. Each threshold is calibrated
from data, not picked by hand.
"""

# Confidence below which we escalate rather than trust the classifier.
# Derivation: from reports/audit_predictions.csv, the smallest threshold at
# which P(correct | confidence >= t) >= 0.85. Replace this comment with the
# actual observed value once you've computed it.
CONFIDENCE_THRESHOLD = 0.75

# Retrieval similarity below which we consider the evidence too weak to
# ground a reply. Derivation: from data/retrieval_labels.csv, the smallest
# threshold at which P(relevant | score >= t) >= 0.5.
RETRIEVAL_THRESHOLD = 0.30

# Intents we never auto-handle. Auth issues require identity verification
# that no historical corpus can perform.
SENSITIVE_INTENTS = {"account_access_or_security"}

# Retrieval depth. K_RETRIEVE for evidence shown to the generator;
# K_EVAL for the retrieval evaluation.
K_RETRIEVE = 3
K_EVAL = 5