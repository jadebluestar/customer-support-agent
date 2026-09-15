# Failure modes

Five failure modes with real examples from the golden set and hypotheses.

## 1. `general_complaint` is a magnet

45 of 164 examples (27%) are labelled `general_complaint`. The LLM
predicts it for 20 misclassified examples whose true label was something
more specific. Recall is 33/45 = 73% on the Groq run.

Examples:
- "Amazon Logistics strikes again!"                true=delivery_delay   pred=general_complaint
- "what is going on"                                true=delivery_delay   pred=general_complaint
- "can you explain what 'external factors' mean"    true=order_status     pred=general_complaint

Hypothesis: the class is the largest in the training set, and its definition
("frustration with no named issue") is broad enough to absorb ambiguous
inputs. Fix in v2: split into named sub-categories or drop entirely.

## 2. Prompt/class mismatch for `availability_or_eligibility_query`

First audit run scored 1/18 because the label was in the golden set and
`ALLOWED` but not in `DEFS` — the model had no definition. After adding the
definition, accuracy went to 18/18. This is a silent failure: adding a class
without updating its prompt definition gives a model that structurally
cannot predict that class.

## 3. `how_to_or_feature_question` collapsed on Groq

On OpenRouter gpt-4o-mini: 3/3 correct. On Groq gpt-oss-120b: 0/3.
Samples:
- "there needs to be a way to report reviews"       → predicted website_or_app_ux
- "does amazon allow price alerts?"                 → predicted availability_or_eligibility_query
- "no speed change for video player?"               → predicted availability_or_eligibility_query

Hypothesis: with only 3 training examples, this class is too thin for the
model to learn a stable decision boundary. Groq's model collapses it into
nearby classes.

## 4. `order_status_or_tracking` vs `delivery_delay_or_missing` boundary

Confusion: 2/6 correct; the rest absorbed into delivery_delay or
availability. Samples:
- "what time are packages supposed to be delivered?"  → predicted availability
- "hasn't shipped yet but I want to keep trusting you" → predicted delivery_delay

Hypothesis: the model keys on shipping keywords and picks the more frequent
class. The two are operationally distinct (delay → "we'll expedite";
status → "here's your tracking") but lexically similar.

## 5. Some golden labels are wrong

Cases where the model is arguably right:
- "what time are packages supposed to be delivered?"  labelled order_status,
  model predicts how_to. Model is right — it's informational.
- "was not in an Amazon box"                          labelled delivery_delay,
  model predicts wrong_or_damaged_item. Model is right.

Hypothesis: labels were produced in a single pass without a second reviewer.
Reported accuracy is a lower bound. Direction of bias favours the model.

## Additional finding: silent-drop bug in the pipeline

The first reply-eval run exposed a case where `decision: handle` coexisted
with `reply_status: "error"` and `reply: null`. The message was marked as
handled while no reply was sent. Root cause: agent.handle() passed
`"PENDING"` to the final should_escalate() call instead of the generation
result. Fixed; generation failure now escalates. See decision_log entry 20.