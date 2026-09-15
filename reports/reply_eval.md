# Reply evaluation

Common examples between human and judge: 20

### All examples (n=20)

| Dimension | kappa | human mean | judge mean |
|---|---|---|---|
| correctness | 0.038 | 4.80 | 3.50 |
| relevance | -0.107 | 4.90 | 3.75 |
| helpfulness | -0.143 | 4.55 | 3.40 |
| groundedness | 0.028 | 4.75 | 3.50 |

| Dimension | agreement | human positives | judge positives |
|---|---|---|---|
| hallucination | 0.950 | NNYNYNNNNNNNNNNNNNNN/20 | NNNNYNNNNNNNNNNNNNNN/20 |
| escalation_correct | 0.700 | YYYYYYYYYYYYYYYYYYYY/20 | YYYNYYYNNYNNYYYYYNYY/20 |
### Handle-only (n=14)

| Dimension | kappa | human mean | judge mean |
|---|---|---|---|
| correctness | 0.759 | 4.71 | 4.57 |
| relevance | -0.105 | 4.86 | 4.93 |
| helpfulness | 0.533 | 4.36 | 4.43 |
| groundedness | 0.662 | 4.64 | 4.57 |

| Dimension | agreement | human positives | judge positives |
|---|---|---|---|
| hallucination | 0.929 | NNYYNNNNNNNNNN/14 | NNNYNNNNNNNNNN/14 |
| escalation_correct | 1.000 | YYYYYYYYYYYYYY/14 | YYYYYYYYYYYYYY/14 |
### Escalate-only (n=6)

| Dimension | kappa | human mean | judge mean |
|---|---|---|---|
| correctness | 0.000 | 5.00 | 1.00 |
| relevance | 0.000 | 5.00 | 1.00 |
| helpfulness | 0.000 | 5.00 | 1.00 |
| groundedness | 0.000 | 5.00 | 1.00 |

| Dimension | agreement | human positives | judge positives |
|---|---|---|---|
| hallucination | 1.000 | NNNNNN/6 | NNNNNN/6 |
| escalation_correct | 0.000 | YYYYYY/6 | NNNNNN/6 |
**Kappa interpretation:** <0.20 poor, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, >0.80 almost perfect.

## Notes

- The LLM judge scored correct escalations as 1/5 on numeric dimensions ("no reply given"). Human evaluation scores correct escalations as 5/5 ("the right output for this case"). This rubric-alignment issue inflates disagreement on escalate examples and depresses kappa on the all-examples set.
- The **handle-only** kappa table above is the more meaningful measure of reply quality agreement.
- The binary `escalation_correct` dimension measures whether the handle/escalate decision itself was right and is not affected by the reply-quality rubric mismatch.

## Interpretation

The reply evaluation was run on 20 examples: 14 that the system decided to
handle and 6 it decided to escalate.

### Handle-only: the meaningful number

On the 14 handle examples — where a reply was actually generated — the
human and judge agreed substantially:

- correctness: kappa 0.759 (substantial)
- groundedness: kappa 0.662 (substantial)
- helpfulness: kappa 0.533 (moderate)
- hallucination: 93% raw agreement (both flagged the same 1 case)

`relevance` shows a kappa of -0.105 despite means of 4.86 (human) and 4.93
(judge). This is a kappa paradox: when both raters give essentially every
example the same score, there is no variance for the statistic to measure
and kappa can go negative even under near-perfect agreement. The means are
the fairer signal here.

### Escalate-only: systematic disagreement, not noise

On the 6 escalation examples, human and judge disagreed on every dimension:

- Numeric (correctness/relevance/helpfulness/groundedness): human 5.00,
  judge 1.00. The human rubric treats "correct escalation" as the right
  output and scores 5. The judge rubric treats "no reply sent" as a failed
  reply and scores 1. This is a definition mismatch, not a quality finding.
- `escalation_correct`: human said Y for all 6, judge said N for all 6.
  The judge penalizes escalation regardless of context. Human evaluation
  agreed with every escalation the system made.

This indicates the judge has a bias toward handling — it interprets the
absence of a reply as a failure, even when a reply is not the correct
outcome. An improved judge prompt would separate two questions: "was
handle/escalate the right decision?" and (only if handled) "is the reply
good?".

### Overall agreement

| Metric | Value |
|---|---|
| Handle-only kappa (correctness) | 0.759 |
| Handle-only kappa (groundedness) | 0.662 |
| Handle-only `escalation_correct` | 1.000 (perfect) |
| All-examples `escalation_correct` | 0.700 (14/20) |
| Escalate-only numeric kappa | 0.000 |

The headline reply-quality metric should be the **handle-only** kappa. The
escalation decision quality is measured separately by the binary
`escalation_correct` column, on which human and judge agree 100% for
handle and disagree 100% for escalate.

### Known limitations

- n=20 is small. A kappa of 0.759 on 14 examples has wide confidence
  intervals.
- The judge is the same model family as the classifier and generator. Shared
  blind spots cannot be ruled out.
- The human ratings were produced by a single annotator. Inter-annotator
  agreement was not measured.