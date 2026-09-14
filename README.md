#  Customer Support Agent (based on AmazonHelp dataset)

A small, intentionally engineered customer-support agent for AmazonHelp
tweets. Classifies intent, retrieves similar historical resolutions, drafts
a grounded reply, or escalates to a human with a deterministic, inspectable
policy.

## Headline results

**Intent classification** (5-fold GroupKFold by conversation, 164 examples,
13 classes):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 0.275 ± 0.057 | 0.041 ± 0.010 |
| TF-IDF + LogReg | 0.373 ± 0.050 | 0.214 ± 0.044 |
| **LLM few-shot (Groq gpt-oss-120b)** | **0.720 ± 0.035** | **0.590 ± 0.080** |

**Retrieval** (25 hand-labelled queries, top-5 from a 25k-pair corpus):

| Metric | Value |
|---|---|
| Recall@1 | 0.88 |
| Recall@3 | 1.00 |
| Recall@5 | 1.00 |
| MRR | 0.94 |
| Precision@5 | 0.82 |

**Reply quality** (LLM-judge + human agreement, 19–20 examples): see
`reports/reply_eval.md`.

## Quickstart (< 15 minutes)

```bash
git clone <this repo>
cd hiver-support-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env, paste GROQ_API_KEY from https://console.groq.com/keys

# Reproduce headline numbers (all cached, no API spend)
python cv_from_cache.py

# End-to-end agent demo on 4 messages (~5 API calls)
python agent.py

Full re-run from raw tweets (downloads Kaggle dataset, ~15 min, ~$0.10
of Groq credit):

bash
python extract_conversations.py
python explore_intents.py
python build_golden_set.py
python label_golden.py        # interactive, ~90 min
python split_golden.py
python cv_eval.py
python build_retrieval_corpus.py
python make_retrieval_eval_set.py
python eval_retrieval.py
Architecture
text
customer message
    │
    ▼
[llm_classifier]   intent + confidence, status contract
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
Every stage is a plain Python module. No frameworks, no vector DB, no
agent loop. 25,000 embeddings fit in a 30MB numpy array; cosine
similarity is one matrix multiply.

What's in the repo
Pipeline modules

llm_classifier.py — few-shot intent classifier, structured JSON output,
explicit status on every call (no silent fallback).

retriever.py — embeds queries with all-MiniLM-L6-v2, retrieves top-k
by cosine similarity over a numpy matrix.

escalation.py — ordered if rules; no LLM.

reply_generator.py — grounded generation with unsupported_claims
self-check.

agent.py — SupportAgent.handle() ties the four stages together.

thresholds.py — tuned constants with derivation comments.

Scripts (data prep, eval, audits)

brand_exploration.py — compares candidate brands by conversation depth.

extract_conversations.py — reconstructs conversations from the reply
graph in twcs.csv.

explore_intents.py, subcluster_cluster0.py — data-driven taxonomy
discovery (TF-IDF + KMeans, no invented categories).

build_golden_set.py, label_golden.py — stratified 200-example
hand-labelled golden set.

split_golden.py, cv_eval.py, cv_from_cache.py — grouped evaluation.

audit.py — confusion matrix, misclassified examples, near-duplicate
check.

build_retrieval_corpus.py, make_retrieval_eval_set.py,
eval_retrieval.py — retrieval pipeline + Recall@K eval.

make_reply_eval_set.py, llm_judge_replies.py, eval_reply.py —
reply quality eval with Cohen's kappa.

Data (committed)

golden_set/ — the 200 hand-labelled examples and their splits.

reports/ — failure modes, misleading-number caveats, decision log,
next-week plan, retrieval eval, reply eval.

Reports
Failure modes

What's misleading about the headline number

Decision log

Next week

Retrieval evaluation

Reply evaluation

Design principles
Data-grounded taxonomy. All 13 intents came out of clustering real
AmazonHelp openings, not from an LLM's idea of what support intents
should be.

No silent failures. Every LLM call carries a status. A 402 or 429
or 404 cannot become a general_complaint prediction. The eval harness
refuses to report if error rate > 5%.

Conversation-grouped splits. All tweets from the same thread stay in
the same fold. Asserted, not commented.

No framework dependencies. No LangChain, no LlamaIndex, no vector
DB. Every line is explainable.

Honest numbers. Standard deviation across 5 folds, not a confidence
interval. Thin classes flagged as unreliable. Cross-model comparison
(OpenRouter gpt-4o-mini vs Groq gpt-oss-120b) reported in full.

Known limitations
Golden set is 164 real-intent examples across 13 classes. Three classes
have ≤3 examples; per-class metrics there are noise.

The general_complaint bucket absorbed some examples that arguably
belong to specific intents. Documented in failure_modes.md.

Reply evaluation is on 19–20 examples. Cohen's kappa is reported, not
hidden.

The evaluation was run on AmazonHelp only. Cross-brand generalisation
is untested.

