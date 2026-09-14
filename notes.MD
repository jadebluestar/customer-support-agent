## Known limitation: label audit
After labelling, a 20-row audit of the `general_complaint` bucket found ~70%
were arguably mislabelled into a specific intent (support, drivers, streaming,
packaging). I did not re-label the full set to preserve time; instead:
- `general_complaint` (45) overstates itself; per-class F1 for it will be
  optimistic because it absorbed examples that belong elsewhere.
- A real `packaging_complaint` intent surfaced in the audit (~15% of sampled
  GC rows) and is a documented taxonomy gap.
- This is failure mode #1 (see reports/failure_modes.md).

# Failure modes

Top 5 failure modes, each with real examples from the golden set and a hypothesis.

## 1. `general_complaint` acts as a magnet (biggest issue)

Of 52 misclassified examples, 20 were predicted `general_complaint` when the
true label was something else. Its recall is 23/45 = 51%.

Examples:
- "Amazon Logistics strikes again!"          true=delivery_delay   pred=general_complaint
- "what is going on"                          true=delivery_delay   pred=general_complaint
- "can you explain what 'external factors' mean"  true=order_status    pred=general_complaint

Hypothesis: the class definition is broad ("frustration with no specific named
issue"), and the LLM defaults to it when the tweet is short or ambiguous. It
also has the most training examples (45), so the model has learned a prior
toward it.

Fix in v2: split `general_complaint` into a small set of named sub-categories
or drop it entirely and force every message into a specific intent plus an
`ambiguous` flag.

## 2. `website_or_app_ux` has very low recall (2/7)

Examples:
- "there needs to be a way to report reviews"  true=how_to    pred=website_or_app_ux
- "the site is so slow today"                  (see test set)

Hypothesis: only 7 examples in the golden set. Per-class metric is noise.
Fix: label more examples or merge into `general_complaint`.

## 3. `order_status_or_tracking` vs `delivery_delay_or_missing` boundary is fuzzy

Confusion counts: 1→delivery, 1→general, 1→wrong_item, 3 correct.
Examples:
- "when will this be sent?" (with frustration)          → predicted delivery_delay
- "hasn't shipped yet but I want to keep trusting you"  → predicted delivery_delay

Hypothesis: the model sees "shipping" keywords and defaults to delivery_delay.
Fix: sharpen the definition, or accept the boundary is genuinely soft and
report it as an inherent taxonomy limit.

## 4. `customer_service_quality` bleeds both ways

5/10 correct; 7 general_complaint examples predicted as customer_service, and
2 delivery_delay examples predicted as customer_service.

Examples:
- "customer service is useless"      correct
- "Amazon Logistics strikes again"   not correct (should be delivery)

Hypothesis: "Amazon Logistics" and "customer service" both signal "Amazon
failed me" to the model.

## 5. Some labels are wrong; the model is sometimes right

Examples where the audit showed the model's prediction is arguably better than
the golden label:
- "what time are packages supposed to be delivered?"  labelled order_status,
  model predicts how_to. Model is right — this is informational, not about a
  specific order.
- "can you explain what 'external factors' means?"    same pattern.
- "was not in an Amazon box"                          labelled delivery_delay,
  model predicts wrong_or_damaged_item. Model is right.

Hypothesis: gold-standard labels were produced in a single pass without a
second reviewer. A relabelling pass with adjudication would raise every metric.
Direction of bias: reported accuracy is a lower bound on true model quality.

Headline: LLM few-shot intent classifier, accuracy 0.720 ± 0.035,
macro F1 0.590 ± 0.080 (5-fold GroupKFold, 164 real-intent examples, 13 classes,
Groq gpt-oss-120b backend).

## Cross-model comparison

Two backends were evaluated during development:

| Backend | Model | Accuracy | Macro F1 |
|---|---|---|---|
| OpenRouter | gpt-4o-mini | 0.683 ± 0.054 | 0.653 ± 0.055 |
| Groq | gpt-oss-120b | 0.720 ± 0.035 | 0.590 ± 0.080 |

The Groq model is more accurate overall but fails on thin classes that
gpt-4o-mini gets right (how_to_or_feature_question: 3/3 vs 0/3;
account_access_or_security: 1/1 vs 0/1). Macro F1 punishes thin-class
failure heavily; accuracy does not. This is a real trade-off, not noise:
the two models genuinely differ in how they generalize on rare classes.

The reported headline is the Groq number because that is the shipped
backend. The OpenRouter number is retained above for transparency.

## 16. Groq gpt-oss-120b over OpenRouter gpt-4o-mini

After OpenRouter's quota was exhausted mid-evaluation, migrated to Groq's
free tier with `openai/gpt-oss-120b`. Two things changed:
- Accuracy improved by 0.037.
- Macro F1 dropped by 0.063 because thin classes (<5 examples) went to zero
  on Groq while they were correct on gpt-4o-mini. This is a real
  generalization difference, not a metric artefact.

Reported the Groq numbers because that's the shipped backend. Documented
the OpenRouter comparison because hiding it would be misleading.

## 17. `llama-3.3-70b-versatile` deprecation

Groq retired this model on 16 Aug 2026. Every call returned 404. The
status contract surfaced the failure cleanly, no silent fallback occurred.
Migrated to `openai/gpt-oss-120b` per Groq's documented path.

## 18. Error-rate gate in evaluation

`cv_eval.py` and `audit.py` now refuse to report metrics if the LLM error
rate exceeds 5%. This is the fix for the earlier OpenRouter failure, where
402 errors silently became `general_complaint` predictions and the LLM row
matched the majority baseline exactly. The gate makes that class of failure
impossible to report.