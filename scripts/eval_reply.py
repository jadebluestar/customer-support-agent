"""
Reply-quality evaluation: Cohen's kappa between human and LLM-judge ratings.

Reads:
    data/reply_ratings_human.csv
    data/reply_ratings_judge.csv

Writes:
    reports/reply_eval.md

Handles:
- Missing examples (judge failure on one row) — intersects on example_id.
- Separate kappa for handle-only vs all-examples, so the rubric-mismatch on
  escalations doesn't swamp the headline number.
"""
import os
import pandas as pd
from sklearn.metrics import cohen_kappa_score

HUMAN = "data/reply_ratings_human.csv"
JUDGE = "data/reply_ratings_judge.csv"
AGENT = "data/reply_eval_agent_output.jsonl"
OUT   = "reports/reply_eval.md"

NUMERIC = ["correctness", "relevance", "helpfulness", "groundedness"]
BINARY  = ["hallucination", "escalation_correct"]


def load_decisions(path):
    """Map example_id -> decision (handle/escalate)."""
    import json
    decisions = {}
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            decisions[ex["example_id"]] = ex["decision"]
    return decisions


def kappa_table(h, j, mask_label, decisions=None, filter_decision=None):
    """Return markdown rows for numeric + binary kappa on a subset."""
    rows = []
    if filter_decision is not None and decisions is not None:
        keep = [eid for eid in h.index if decisions.get(eid) == filter_decision]
        h2, j2 = h.loc[keep], j.loc[keep]
    else:
        h2, j2 = h, j
    n = len(h2)
    if n == 0:
        return f"### {mask_label}\n\nNo examples.\n\n"

    out = [f"### {mask_label} (n={n})\n"]
    out.append("| Dimension | kappa | human mean | judge mean |")
    out.append("|---|---|---|---|")
    for dim in NUMERIC:
        k = cohen_kappa_score(h2[dim], j2[dim], weights="quadratic")
        out.append(f"| {dim} | {k:.3f} | {h2[dim].mean():.2f} | {j2[dim].mean():.2f} |")

    out.append("")
    out.append("| Dimension | agreement | human positives | judge positives |")
    out.append("|---|---|---|---|")
    for dim in BINARY:
        agree = (h2[dim] == j2[dim]).mean()
        out.append(f"| {dim} | {agree:.3f} | {h2[dim].sum()}/{n} | {j2[dim].sum()}/{n} |")
    out.append("")
    return "\n".join(out)


def main():
    if not (os.path.exists(HUMAN) and os.path.exists(JUDGE)):
        print(f"Missing files: {HUMAN} or {JUDGE}")
        return

    h = pd.read_csv(HUMAN).set_index("example_id").sort_index()
    j = pd.read_csv(JUDGE).set_index("example_id").sort_index()

    # Coerce numerics
    for df in (h, j):
        for col in NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in BINARY:
            df[col] = df[col].astype(str).str.upper().str.strip()

    common = sorted(set(h.index) & set(j.index))
    h, j = h.loc[common], j.loc[common]
    print(f"Common examples: {len(common)}")

    decisions = load_decisions(AGENT)

    os.makedirs("reports", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("# Reply evaluation\n\n")
        f.write(f"Common examples between human and judge: {len(common)}\n\n")
        f.write(kappa_table(h, j, "All examples"))
        f.write(kappa_table(h, j, "Handle-only", decisions, "handle"))
        f.write(kappa_table(h, j, "Escalate-only", decisions, "escalate"))
        f.write(
            "**Kappa interpretation:** <0.20 poor, 0.21–0.40 fair, "
            "0.41–0.60 moderate, 0.61–0.80 substantial, >0.80 almost perfect.\n\n"
        )
        f.write(
            "## Notes\n\n"
            "- The LLM judge scored correct escalations as 1/5 on numeric "
            "dimensions (\"no reply given\"). Human evaluation scores correct "
            "escalations as 5/5 (\"the right output for this case\"). This "
            "rubric-alignment issue inflates disagreement on escalate examples "
            "and depresses kappa on the all-examples set.\n"
            "- The **handle-only** kappa table above is the more meaningful "
            "measure of reply quality agreement.\n"
            "- The binary `escalation_correct` dimension measures whether the "
            "handle/escalate decision itself was right and is not affected by "
            "the reply-quality rubric mismatch.\n"
        )

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
