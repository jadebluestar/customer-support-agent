# What is misleading about my headline number?

Headline: LLM few-shot intent classifier, accuracy 0.720 ± 0.035,
macro F1 0.590 ± 0.080 (5-fold GroupKFold, 164 real-intent examples,
13 classes, Groq gpt-oss-120b).

## What makes the number look better than it is

1. **Several classes have 3–6 examples.** `how_to_or_feature_question` (3),
   `subscription_or_membership` (4), `account_access_or_security` (1).
   Per-class F1 on these is essentially noise.

2. **`general_complaint` is 27% of the dataset.** The majority baseline gets
   0.275 accuracy for free.

3. **The taxonomy was iterated during labelling.** `how_to_or_feature_question`
   and `availability_or_eligibility_query` were added after seeing specific
   golden-set examples. Their definitions were shaped by the data — mild
   optimistic bias.

4. **The ± is standard deviation across 5 folds, not a confidence interval.**
   With n=164 and 13 classes, a bootstrap 95% CI would be wider.

## What makes the number look worse than it is

1. **Some golden labels are wrong** (failure_modes #5). Fixing them raises
   every metric.

2. **Thin classes punish macro F1 heavily.** The Groq model is more accurate
   than gpt-4o-mini (0.720 vs 0.683) but its macro F1 is lower (0.590 vs
   0.653) because it collapses the 3-example `how_to` class while gpt-4o-mini
   got all 3 right.

## Cross-model comparison

| Backend | Model | Accuracy | Macro F1 |
|---|---|---|---|
| OpenRouter | gpt-4o-mini | 0.683 ± 0.054 | 0.653 ± 0.055 |
| Groq | gpt-oss-120b | 0.720 ± 0.035 | 0.590 ± 0.080 |

The relative ranking (LLM ≫ TF-IDF ≫ majority) is robust. The exact macro
F1 depends on the model and on how thin classes are weighted.

## What the number actually means

Directionally: the LLM beats TF-IDF by +0.38 macro F1 and beats majority
by +0.55. That ranking is stable across folds and across both backends.

Quantitatively: read 0.720 / 0.590 as "in the range 0.65–0.75 accuracy /
0.55–0.70 macro F1 under a different seed or slightly different labels."

## What I would measure next

1. Bootstrap 95% CI on macro F1.
2. Inter-annotator agreement on 30 examples (Cohen's kappa) to bound label
   noise.
3. Retrieval recall@5 before evaluating reply quality — a reply cannot be
   better than the evidence it was grounded in.

## Retrieval and reply numbers

Retrieval (25 hand-labelled queries, 25k-pair corpus, all-MiniLM-L6-v2):
Recall@1 0.88, Recall@3 1.00, MRR 0.94, Precision@5 0.82.

Reply quality (20 examples, LLM judge vs human):
- Handle-only kappa: correctness 0.759, groundedness 0.662, helpfulness 0.533.
- Escalate-only: human and judge disagreed systematically on whether
  escalation was correct (0% agreement on the binary dimension).
- The judge penalises escalation regardless of context; human evaluation
  does not. Handle-only kappa is the meaningful reply-quality number.