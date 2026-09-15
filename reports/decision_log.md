# Decision log

Twenty-two non-obvious engineering decisions, why they were made, and what
was traded off.

1. **AmazonHelp, not Spotify.** Chosen after brand_exploration showed the
   richest conversation structure (38% multi-turn, 99% reply rate) and
   clustering surfaced 13 distinct intents from real messages. Trade-off:
   noisier topics.

2. **TF-IDF+KMeans for taxonomy discovery, not embeddings.** No new dep,
   runs in seconds, inspectable. Trade-off: no semantic grouping; sub-cluster 1
   stayed 56% mixed.

3. **Golden set from conversation openings only.** A deployed system sees
   the first message; later turns condition on context the model wouldn't have.

4. **Utility labels excluded from accuracy.** `not_support_related` and
   `insufficient_context` are not intents. 200 labelled → 164 evaluable.

5. **5-fold GroupKFold, not a single split.** With 164 examples a single
   test set has 1–3 examples per class. CV gives every example exactly one
   held-out prediction.

6. **Grouping by `conversation_id`.** Prevents the same conversation appearing
   in train and eval. Asserted in `split_golden.py`, not commented.

7. **LLM few-shot over fine-tuning.** Zero training infrastructure; the
   assignment warns against fine-tuning without demonstrated need.

8. **OpenRouter then Groq.** OpenRouter's quota was exhausted mid-evaluation
   (HTTP 402 on every call, silently turning into general_complaint
   predictions). Migrated to Groq's free tier.

9. **Structured JSON output, validated against `ALLOWED`.** Free-form text
   would make evaluation brittle. Retry-once on invalid, no fallback label.

10. **Duplicate `DEFS` bug found and fixed.** Two assignments to `DEFS`; the
    second (weaker) silently overwrote the first. Removing it raised accuracy
    from 0.51 to 0.68.

11. **13 intents, not 12.** `how_to_or_feature_question` and
    `availability_or_eligibility_query` were added mid-labelling when real
    examples didn't fit. Documented as a taxonomy-iteration weakness.

12. **`lost_or_stolen_package` removed after 0 golden examples.** Leaving it
    gives the model a class it can never win on.

13. **No vector DB.** 25k embeddings fit in 30MB. Matrix multiply is faster
    than any DB round-trip.

14. **Escalation is a rule, not an LLM call.** Inspectable, cheap,
    deterministic.

15. **Multi-intent flag rate 1.5%.** Almost certainly under-detected.
    Documented; not fixed because relabelling was out of budget.

16. **Groq gpt-oss-120b over OpenRouter gpt-4o-mini.** Accuracy improved
    (+0.037) but macro F1 dropped (−0.063) because thin classes collapsed
    on Groq. Reported both numbers.

17. **`llama-3.3-70b-versatile` deprecation.** Retired 16 Aug 2026. Every
    call returned 404. Migrated to `openai/gpt-oss-120b`.

18. **Error-rate gate in evaluation.** `cv_eval.py` and `audit.py` refuse to
    report metrics if error rate exceeds 5%. Fix for the earlier OpenRouter
    incident where 402s silently became general_complaint predictions.

19. **Reply generation error rate on Groq's free tier.** Initial run hit
    30% errors. Backoff wrapper recovered all 20. Pipeline marks every
    failure as `reply_status: "error"` — no silent substitution.

20. **Generation failure was silently not escalating.** Audit found one
    example with `reply_status: "error"` and `decision: "handle"` — a
    silent drop. Root cause: agent.handle() passed `"PENDING"` to the final
    should_escalate() call instead of the generation result. Fixed by
    treating `generation_result=None` as `generator_failed → escalate`.

21. **Exponential backoff on transient API errors.** `_call_raw` is wrapped
    with a 3-attempt exponential backoff (1s, 2s, 4s + jitter) for
    rate_limited, server_error, and network_error. Non-retryable statuses
    return immediately.

22. **Escalation is the correct failure mode.** Three paths lead to
    escalation: classifier failure, sensitive intent, generation failure.
    The cost of an unnecessary escalation is a few minutes of human time;
    the cost of a silent drop is a customer who never gets a reply.