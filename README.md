# Mapping the Emergence of AI in Genomics and Biotechnology

Data analytics sample prepared for the Center for Security and Emerging Technology (CSET),
Walsh School of Foreign Service (SFS), Georgetown University.
Author: Batsirai Mabvakure

An end to end bibliometric pipeline on open data (OpenAlex) that measures how quickly AI is
being adopted in genomics and biotechnology research, detects emerging research fronts, and
maps where the work is done. It includes affiliation entity resolution and LLM-assisted
annotation, each with a quantitative validation step.

## Repository layout

```
config.yaml               corpus definition, windows, thresholds (edit here to re-scope)
src/
  fetch_openalex.py       1. ingest: OpenAlex API, cursor paging, seeded per-year sampling
  build_database.py       2. flatten JSON into DuckDB tables, rebuild abstracts, country credit
  entity_resolution.py    3. resolve raw affiliation strings; holdout precision / recall
  build_views.py          3b. reusable SQL views (sql/views.sql)
  topics.py               4. embeddings, k-means clusters, c-TF-IDF labels, emergence scores
  llm_classify.py         5. LLM annotation + agreement with hand labels
  run_pipeline.py         runs all steps in order
  report_style.py         shared chart style for the report
sql/                      every query used in the report, as standalone .sql files
dags/cset_sample_dag.py   the same pipeline as an Airflow DAG, with data quality checks
data/reference/           ISO country to region / EU27 lookup
report.qmd                Quarto report (renders to PDF)
```

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# optional: OpenAlex API key, if you have one (the pipeline also works without it)
export OPENALEX_API_KEY=...

python src/run_pipeline.py          # fetch -> build_db -> entity_resolution -> views -> topics -> llm
quarto render                       # builds report.pdf
```

A first run with the default cap (`max_records: 60000`) takes roughly 20 to 40 minutes,
most of it downloading and embedding. Re-runs skip years that are already downloaded
and reuse cached embeddings. Start from any step with `python src/run_pipeline.py topics`.

### LLM annotation (optional)

```bash
export ANTHROPIC_API_KEY=...
python src/llm_classify.py sample   # writes data/labels/gold_labels.csv
# hand-label the human_ai_role column: ai_method, ai_application, or not_ai
python src/llm_classify.py run      # labels the sample (cached) and scores agreement
quarto render
```

## Methods in brief

* **Corpus.** Title / abstract match on (AI terms) AND (genomics / biotech terms), 2010 to 2025.
* **Sampling.** When a year exceeds its share of the cap, a seeded random sample is drawn and
  results are reweighted by year, so estimates describe the full corpus.
* **Counting.** Fractional counting by author and country.
* **Impact.** OpenAlex field and year normalized citation percentiles (top 10% and top 1%).
* **Emergence.** Growth in a cluster's share of the corpus between two windows.
* **Validation.** Entity resolution is scored on a hidden holdout; LLM labels are scored
  against hand labels with agreement, Cohen's kappa, and per-class precision / recall.

## Data

OpenAlex data are released under CC0. See https://openalex.org.
