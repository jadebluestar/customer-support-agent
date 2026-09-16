# Customer Support Agent (AmazonHelp)

A small, intentionally engineered customer-support agent for AmazonHelp
tweets. The pipeline classifies intent, retrieves similar historical
resolutions, drafts a grounded reply, or escalates to a human using a
deterministic, inspectable policy.

Plain Python. No LangChain, no LlamaIndex, no vector DB, no agent loop.

---

## Headline Results

### Intent Classification

5-fold GroupKFold by conversation, 164 examples, 13 classes.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority class | 0.275 ± 0.057 | 0.041 ± 0.010 |
| TF-IDF + LogReg | 0.373 ± 0.050 | 0.214 ± 0.044 |
| **LLM few-shot (Groq gpt-oss-120b)** | **0.726 ± 0.037** | **0.591 ± 0.079** |

The ± is standard deviation across folds, not a confidence interval.
See [What's misleading about the headline number](reports/misleading_number.md).

### Retrieval

25 hand-labelled queries, top-5 from a 25k-pair corpus.

| Metric | Value |
|---|---:|
| Recall@1 | 0.88 |
| Recall@3 | 1.00 |
| Recall@5 | 1.00 |
| MRR | 0.94 |
| Precision@5 | 0.82 |

### Reply Quality

LLM judge vs human ratings on 20 examples.

| Dimension | Cohen's Kappa (handle-only, n=14) |
|---|---:|
| Correctness | 0.759 |
| Groundedness | 0.662 |
| Helpfulness | 0.533 |

The judge systematically penalised correct escalations; human raters did
not. That disagreement is documented in
[Reply evaluation](reports/reply_eval.md), not hidden.

---

## Quickstart (< 15 minutes)

### Install

```bash
git clone https://github.com/jadebluestar/customer-support-agent.git
cd customer-support-agent

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Note on torch:** `requirements.txt` pins `torch==2.4.0+cpu` from
PyTorch's CPU wheel index. This keeps the install at ~300 MB instead of
~2.5 GB of CUDA libraries, and is sufficient for `all-MiniLM-L6-v2`
inference on the 25k-pair retrieval corpus. No GPU is required.

### Configure Groq

```bash
cp .env.example .env
# edit .env and paste your GROQ_API_KEY
```

The pipeline raises a clear `KeyError: 'GROQ_API_KEY'` if the key is
missing. There is no silent fallback.

### Reproduce headline numbers

All classification predictions are cached; this step makes no API calls.

```bash
python3 scripts/cv_from_cache.py
```

Expected output:

```text
LLM few-shot (from cached predictions)
  Accuracy  mean=0.726  std=0.037
  Macro F1  mean=0.591  std=0.079
```

### Run the end-to-end agent

The demo runs on 4 messages and makes approximately 5 API calls. The
first run downloads the `all-MiniLM-L6-v2` embedding model (~80 MB) from
HuggingFace; subsequent runs are instant.

```bash
python3 src/agent.py
```

Expected: four blocks, one per message, each with intent, decision, reason,
and reply (or an escalation with no reply).

---

## Problem Framing

### What "good" means for AmazonHelp

AmazonHelp's Twitter support is high-volume, high-repetition, and
frequently frustrated. A good system for this brand must:

- **Correctly identify the issue category** from a short, noisy tweet.
- **Ground its reply in a resolution AmazonHelp has actually used
  before**, not invent a policy.
- **Know when to hand off to a human.** Identity verification (account
  hacks) and issues with no matching historical resolution are worse to
  auto-reply to than to escalate.
- **Never silently drop a message.** A failed generation must escalate,
  not appear "handled."

### What I chose not to build

- **No vector DB.** 25k embeddings fit in roughly 40 MB as 384-dimensional
  float32 vectors; NumPy handles retrieval without database overhead at
  this scale.
- **No LangChain / LlamaIndex.** Plain Python is explainable, and this
  pipeline is five stages.
- **No fine-tuning.** No demonstrated need; the LLM already beats TF-IDF
  by approximately 0.38 macro F1.
- **No multi-brand taxonomy.** Cross-brand generalisation is a next-week
  item.
- **No frontend.** A dict-returning `handle()` is enough.
- **No autonomous agent loop.** This is a deterministic pipeline, not an
  agent in the looping sense.

---

## Architecture

```text
customer message
    │
    ▼
[llm_classifier]   intent + confidence + explicit status
    │
    ▼
[retriever]        top-3 historical (customer, reply) pairs
    │
    ▼
[escalation]       deterministic rules (order matters)
    │
    ▼
[reply_generator]  grounded reply via Groq, or escalate
    │
    ▼
SupportAgent.handle(message) -> dict
```

Every stage is a plain Python module. The 25k embeddings occupy roughly
40 MB as float32 vectors; retrieval is a cosine-similarity matrix
multiply.

---

## What's in the Repo

### Pipeline modules (`src/`)

| File | Purpose |
|---|---|
| `llm_classifier.py` | Few-shot intent classifier with structured JSON output and an explicit status on every call |
| `retriever.py` | Embeds queries with `all-MiniLM-L6-v2`; retrieves top-k by cosine similarity over a NumPy matrix |
| `escalation.py` | Ordered deterministic rules; no LLM call |
| `reply_generator.py` | Grounded reply generation with an `unsupported_claims` self-check |
| `agent.py` | `SupportAgent.handle()` tying the stages together |
| `thresholds.py` | Tuned constants with derivation comments |

### Data preparation, evaluation, audits (`scripts/`)

| File | Purpose |
|---|---|
| `brand_exploration.py` | Compares candidate brands by conversation depth |
| `extract_conversations.py` | Reconstructs conversations from the reply graph |
| `explore_intents.py`, `subcluster_cluster0.py` | Data-driven taxonomy discovery (TF-IDF + KMeans) |
| `build_golden_set.py`, `label_golden.py` | 200-example hand-labelled golden set |
| `split_golden.py` | Conversation-grouped evaluation splits |
| `cv_eval.py`, `cv_from_cache.py` | 5-fold grouped evaluation |
| `audit.py` | Confusion matrix, misclassified examples, near-duplicate checks |
| `build_retrieval_corpus.py` | Builds the historical retrieval corpus |
| `make_retrieval_eval_set.py`, `eval_retrieval.py` | Retrieval evaluation (Recall@K, MRR) |
| `make_reply_eval_set.py`, `llm_judge_replies.py`, `eval_reply.py` | Reply-quality evaluation with Cohen's kappa |

### Data and reports

- `golden_set/` — 200 hand-labelled examples and their splits
- `data/` — generated artefacts (retrieval corpus and embeddings committed for the reviewer quickstart; raw dataset gitignored)
- `reports/` — six Markdown reports (see below)

---

## Full Re-run From Raw Tweets

The complete pipeline can be reproduced from the raw Kaggle dataset.
The data preparation and automated evaluation take roughly 15 minutes;
manual labelling takes substantially longer and is interactive.

```bash
python3 scripts/extract_conversations.py
python3 scripts/explore_intents.py
python3 scripts/build_golden_set.py
python3 scripts/label_golden.py        # interactive, ~90 min
python3 scripts/split_golden.py
python3 scripts/cv_eval.py
python3 scripts/build_retrieval_corpus.py
python3 scripts/make_retrieval_eval_set.py
python3 scripts/eval_retrieval.py
python3 scripts/make_reply_eval_set.py
python3 scripts/llm_judge_replies.py
python3 scripts/eval_reply.py
```

`extract_conversations.py` downloads the dataset via `kagglehub` and
caches it. No Kaggle credentials are required for the cached version,
but a first-time download may require Kaggle access depending on the
environment.

---

## Reports

Six detailed reports in `reports/`:

- [Failure modes](reports/failure_modes.md) — five real failures with examples
- [What's misleading about the headline number](reports/misleading_number.md) — the required caveats section
- [Decision log](reports/decision_log.md) — full engineering decision history
- [Next week](reports/next_week.md) — realistic improvement priorities
- [Retrieval evaluation](reports/retrieval_eval.md) — Recall@K, MRR, Precision@K
- [Reply evaluation](reports/reply_eval.md) — Cohen's kappa, judge–human agreement

---

## Design Principles

**Data-grounded taxonomy.** Candidate intents were discovered by clustering
real AmazonHelp conversation openings, then refined during manual
labelling.

**No silent failures.** Every LLM call carries an explicit status
(`success` / `rate_limited` / `auth_error` / `invalid_output` / ...). A
402, 429, or 404 cannot silently become a `general_complaint` prediction.
The evaluation harness refuses to report if the error rate exceeds 5%.

**Conversation-grouped splits.** All tweets from the same conversation
stay in the same fold. Asserted programmatically, not just documented.

**No framework dependencies.** Every major component is implemented
directly in Python so its behaviour is inspectable.

**Honest numbers.** Standard deviation across 5 folds is reported, not
presented as a confidence interval. Thin classes are flagged. The
cross-model comparison (OpenRouter `gpt-4o-mini` vs Groq `gpt-oss-120b`)
is documented in full.

---

## Known Limitations

**Small per-class sample sizes.** The golden set contains 164 real-intent
examples across 13 classes. Three classes have ≤3 examples; their
per-class metrics are noisy.

**Broad `general_complaint` bucket.** 27% of the dataset. Documented in
[failure_modes.md](reports/failure_modes.md).

**Small reply evaluation.** 20 examples. Cohen's kappa is reported, and
the judge–human disagreement on escalations is documented, not hidden.

**Single-brand evaluation.** AmazonHelp only. Cross-brand generalisation
is untested.

---

## What Is Misleading About the Headline Number?

The headline classification result is:

> **0.726 ± 0.037 accuracy / 0.591 ± 0.079 macro F1**

This should not be read as a production success rate. The evaluation has:

- only 164 real-intent examples
- 13 unevenly represented classes, several with ≤5 examples
- a broad `general_complaint` category (27%)
- a taxonomy that was iterated during labelling (mild optimistic bias)
- single-brand evaluation
- standard deviation across folds, not a confidence interval
- some human labels that the audit showed were arguably wrong

The result is best read directionally: the LLM classifier substantially
outperformed both baselines on this evaluation. The macro-F1 improvement
over TF-IDF + LogReg is approximately **+0.38**.

See [misleading_number.md](reports/misleading_number.md) for the full
caveats.

---

## Retrieval Evaluation

25 hand-labelled queries against ~25,000 historical customer-support pairs.

A retrieved case is relevant if:

1. the retrieved customer message describes the same category of problem
   as the query, **and**
2. the historical brand reply shows an appropriate resolution approach.

| Metric | Value |
|---|---:|
| Recall@1 | 0.88 |
| Recall@3 | 1.00 |
| Recall@5 | 1.00 |
| MRR | 0.94 |
| Precision@5 | 0.82 |

---

## Reply Evaluation

20 examples: 14 handled, 6 escalated. LLM judge vs human ratings.

Handle-only agreement:

| Dimension | Cohen's Kappa |
|---|---:|
| Correctness | 0.759 |
| Groundedness | 0.662 |
| Helpfulness | 0.533 |

The judge penalised every escalation as 1/5 on numeric dimensions
("no reply given"), while human ratings scored correct escalations as
5/5 ("the right output for this case"). This is documented as a
**rubric-alignment issue on the judge's side**, not a system failure.

See [reply_eval.md](reports/reply_eval.md).

---

## Next Steps

With one more week, in priority order:

1. Relabel the golden set with a second annotator.
2. Bootstrap 95% confidence intervals for macro F1.
3. Evaluate retrieval Recall@5 on more queries.
4. Improve retrieval only if current recall proves insufficient.
5. Split `general_complaint` into more specific categories.
6. Fix the LLM judge rubric so decision-correctness is separated from
   reply-quality.
7. Tune escalation thresholds on a validation set.
8. Add a temporal train/eval split.
9. Test cross-brand generalisation.
10. Expand LLM response caching.

See [next_week.md](reports/next_week.md) for details.
