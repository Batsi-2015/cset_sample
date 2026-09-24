"""Step 5: LLM-assisted annotation, validated against hand labels.

Why: keyword queries are noisy. Some papers mention "machine learning" once in passing,
others build new methods. An LLM can annotate at scale, but its labels are only useful
if we measure how often it agrees with a careful human. This script does both.

Usage
  python src/llm_classify.py sample   # draws the LLM sample and writes a gold-label template
  (hand-label data/labels/gold_labels.csv, filling the `human_ai_role` column)
  python src/llm_classify.py run      # calls the LLM (cached), scores it against your labels

Labels
  ai_role:     ai_method       develops or substantially adapts an AI / ML method
               ai_application  applies existing AI / ML tools to a biological question
               not_ai          AI is incidental or absent (a false positive of the keyword query)
  bio_area:    genomics_transcriptomics | protein_structure_design | drug_discovery |
               gene_editing_synbio | clinical_genomics | other
  biosecurity: true if the work concerns pathogens, toxins, or other dual-use biology

Outputs
  data/labels/llm_cache.jsonl  raw model responses (so re-runs cost nothing)
  DuckDB tables: llm_labels, llm_eval
"""

from __future__ import annotations

import json
import math
import os
import sys

import pandas as pd

from common import ROOT, connect, get_logger, load_config

log = get_logger("llm_classify")
LABEL_DIR = ROOT / "data" / "labels"
AI_ROLES = ["ai_method", "ai_application", "not_ai"]
BIO_AREAS = ["genomics_transcriptomics", "protein_structure_design", "drug_discovery",
             "gene_editing_synbio", "clinical_genomics", "other"]

PROMPT = """You are annotating scientific papers for a bibliometric study of AI in biotechnology.
Read the title and abstract and return only a JSON object with these keys:

"ai_role": one of "ai_method" (the paper develops or substantially adapts an AI or machine
  learning method), "ai_application" (the paper applies existing AI or ML tools to answer a
  biological question), or "not_ai" (AI or ML is incidental, only mentioned, or absent).
"bio_area": one of {bio_areas}.
"biosecurity": true if the work concerns pathogens, toxins, or other dual-use biology, else false.
"rationale": one short sentence.

Title: {title}
Abstract: {abstract}
"""


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def cmd_sample(cfg: dict) -> None:
    lc = cfg["llm_classification"]
    con = connect(cfg, read_only=True)
    df = con.execute(f"""
        SELECT * FROM (
            SELECT work_id, year, title, abstract
            FROM works
            WHERE abstract IS NOT NULL AND length(abstract) > 200
            ORDER BY work_id
        )
        USING SAMPLE reservoir({int(lc['sample_size'])} ROWS) REPEATABLE ({cfg['topics']['random_state']})
    """).df()
    con.close()
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABEL_DIR / "llm_sample.csv", index=False)
    gold = df.head(lc["gold_size"]).copy()
    gold["human_ai_role"] = ""
    gold["human_notes"] = ""
    gold_path = LABEL_DIR / "gold_labels.csv"
    if gold_path.exists():
        log.warning("%s exists, not overwriting your labels", gold_path)
    else:
        gold.to_csv(gold_path, index=False)
    log.info("Wrote %s sample papers and a %s-paper gold template", len(df), len(gold))


def call_llm(client, model: str, row: pd.Series) -> dict:
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(
            bio_areas=", ".join(f'"{b}"' for b in BIO_AREAS),
            title=row.title, abstract=str(row.abstract)[:3500])}],
    )
    text = msg.content[0].text
    start, end = text.find("{"), text.rfind("}") + 1
    out = json.loads(text[start:end])
    if out.get("ai_role") not in AI_ROLES:
        out["ai_role"] = "invalid"
    return out


def cmd_run(cfg: dict) -> None:
    lc = cfg["llm_classification"]
    sample_path = LABEL_DIR / "llm_sample.csv"
    if not sample_path.exists():
        cmd_sample(cfg)
    sample = pd.read_csv(sample_path)

    cache_path = LABEL_DIR / "llm_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["work_id"]] = rec

    todo = sample[~sample.work_id.isin(cache)]
    if len(todo):
        if not os.getenv("ANTHROPIC_API_KEY"):
            log.warning("ANTHROPIC_API_KEY not set; %s papers left unlabeled. "
                        "Set the key and re-run to finish.", len(todo))
        else:
            import anthropic
            client = anthropic.Anthropic()
            with open(cache_path, "a", encoding="utf-8") as fh:
                for i, row in enumerate(todo.itertuples(index=False), 1):
                    try:
                        rec = {"work_id": row.work_id, **call_llm(client, lc["model"], row)}
                    except Exception as exc:  # keep going; failures are logged and retried next run
                        log.warning("Failed on %s: %s", row.work_id, exc)
                        continue
                    cache[row.work_id] = rec
                    fh.write(json.dumps(rec) + "\n")
                    if i % 50 == 0:
                        log.info("Labeled %s / %s", i, len(todo))

    labels = pd.DataFrame([cache[w] for w in sample.work_id if w in cache])
    if labels.empty:
        log.info("No LLM labels yet")
        return
    if "model" not in labels:
        labels["model"] = lc["model"]
    labels["model"] = labels["model"].fillna(lc["model"])

    # Keyword-query precision: how many retrieved papers are really about AI?
    n = len(labels)
    k = int((labels.ai_role != "not_ai").sum())
    lo, hi = wilson(k, n)
    rows = [{"metric": "query_precision_llm", "value": k / n, "ci_low": lo, "ci_high": hi, "n": n}]
    if "off_topic" in labels:
        # Stricter precision: about AI *and* about biology (not, e.g., materials science).
        k2 = int(((labels.ai_role != "not_ai") & ~labels.off_topic.fillna(False).astype(bool)).sum())
        lo2, hi2 = wilson(k2, n)
        rows.append({"metric": "query_precision_ai_and_bio", "value": k2 / n,
                     "ci_low": lo2, "ci_high": hi2, "n": n})
    if "biosecurity" in labels:
        k3 = int(labels.biosecurity.fillna(False).astype(bool).sum())
        lo3, hi3 = wilson(k3, n)
        rows.append({"metric": "share_biosecurity_relevant", "value": k3 / n,
                     "ci_low": lo3, "ci_high": hi3, "n": n})

    gold_path = LABEL_DIR / "gold_labels.csv"
    if gold_path.exists():
        gold = pd.read_csv(gold_path, keep_default_na=False)
        gold = gold[gold.human_ai_role.isin(AI_ROLES)]
        merged = gold.merge(labels[["work_id", "ai_role"]], on="work_id")
        if len(merged):
            agree = int((merged.human_ai_role == merged.ai_role).sum())
            lo, hi = wilson(agree, len(merged))
            rows.append({"metric": "agreement_with_human", "value": agree / len(merged),
                         "ci_low": lo, "ci_high": hi, "n": len(merged)})
            rows.append({"metric": "cohen_kappa",
                         "value": cohen_kappa(merged.human_ai_role, merged.ai_role),
                         "ci_low": None, "ci_high": None, "n": len(merged)})
            for role in AI_ROLES:
                tp = int(((merged.ai_role == role) & (merged.human_ai_role == role)).sum())
                pp = int((merged.ai_role == role).sum())
                ap = int((merged.human_ai_role == role).sum())
                rows.append({"metric": f"precision_{role}", "value": tp / pp if pp else None,
                             "ci_low": None, "ci_high": None, "n": pp})
                rows.append({"metric": f"recall_{role}", "value": tp / ap if ap else None,
                             "ci_low": None, "ci_high": None, "n": ap})
            confusion = pd.crosstab(merged.human_ai_role, merged.ai_role)
            log.info("Confusion matrix (rows = human, columns = LLM):\n%s", confusion)
        else:
            log.info("Gold labels not filled in yet; skipping agreement metrics")

    evaluation = pd.DataFrame(rows)
    log.info("\n%s", evaluation.to_string(index=False))
    con = connect(cfg)
    for name, df in {"llm_labels": labels, "llm_eval": evaluation}.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    con.close()


def main() -> None:
    cfg = load_config()
    if not cfg["llm_classification"]["enabled"]:
        log.info("LLM classification disabled in config.yaml")
        return
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"sample": cmd_sample, "run": cmd_run}[cmd](cfg)


if __name__ == "__main__":
    main()
