"""Step 1: Ingest publication records from the OpenAlex API.

Outputs (data/raw/):
  works_<year>.jsonl.gz   one OpenAlex work per line, only the fields we use
  counts_by_year.json     yearly totals for the AI x bio corpus and the bio baseline
  fetch_manifest.json     query, timestamp and record counts, for provenance

Design notes
  * If a year has more records than its share of the cap, we draw a seeded random
    sample (OpenAlex `sample` + `seed`), so the subset is unbiased and reproducible.
    Otherwise the whole year is paged with a cursor. Per-year sampling weights are
    saved so estimates can be scaled back to the full corpus.
  * The query is built from config.yaml so the corpus definition is versioned with the code.
  * Retries with exponential backoff handle 429 / 5xx responses.
  * Each year is its own file, so an interrupted pull resumes where it stopped.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from datetime import datetime, timezone

import requests

from common import get_logger, load_config, project_path

log = get_logger("fetch")

API = "https://api.openalex.org/works"
MAX_SAMPLE = 10_000  # OpenAlex upper limit for the sample parameter
FIELDS = [
    "id", "doi", "title", "publication_year", "publication_date", "type", "language",
    "cited_by_count", "fwci", "citation_normalized_percentile", "authorships",
    "primary_topic", "keywords", "abstract_inverted_index", "primary_location",
]


def or_group(terms: list[str]) -> str:
    return "(" + " OR ".join(f'"{t}"' for t in terms) + ")"


def build_filter(cfg: dict, terms_expr: str, year: int | None = None) -> str:
    c = cfg["corpus"]
    years = str(year) if year else f"{c['start_year']}-{c['end_year']}"
    parts = [
        f"title_and_abstract.search:{terms_expr}",
        f"publication_year:{years}",
        "type:" + "|".join(c["work_types"]),
    ]
    return ",".join(parts)


def base_params(cfg: dict) -> dict:
    params = {"mailto": cfg["project"]["contact_email"]}
    api_key = os.getenv("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def get_json(session: requests.Session, params: dict, max_tries: int = 6) -> dict:
    for attempt in range(max_tries):
        resp = session.get(API, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            log.warning("HTTP %s, retrying in %ss", resp.status_code, wait)
            time.sleep(wait)
            continue
        raise RuntimeError(f"OpenAlex error {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("OpenAlex request failed after retries")


def yearly_counts(session: requests.Session, cfg: dict, filt: str) -> dict[int, int]:
    params = {**base_params(cfg), "filter": filt, "group_by": "publication_year"}
    data = get_json(session, params)
    return {int(g["key"]): int(g["count"]) for g in data.get("group_by", [])}


def year_quotas(counts: dict[int, int], cap: int | None) -> dict[int, int]:
    """Allocate the record cap across years in proportion to each year's true size.

    Proportional allocation keeps the year distribution of the downloaded corpus
    faithful to OpenAlex even when we do not download everything.
    """
    if cap is None or cap >= sum(counts.values()):
        return dict(counts)
    total = sum(counts.values())
    return {y: max(1, min(n, round(cap * n / total), MAX_SAMPLE)) for y, n in counts.items()}


def fetch_year(session: requests.Session, cfg: dict, terms_expr: str, year: int,
               quota: int, available: int) -> int:
    """Download one publication year: everything if it fits, else a seeded random sample."""
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    out_path = raw_dir / f"works_{year}.jsonl.gz"
    if out_path.exists():
        log.info("%s already downloaded, skipping", year)
        with gzip.open(out_path, "rt", encoding="utf-8") as fh:
            return sum(1 for _ in fh)

    year_filter = build_filter(cfg, terms_expr, year)
    tmp_path = out_path.with_suffix(".tmp")
    n = 0
    with gzip.open(tmp_path, "wt", encoding="utf-8") as fh:
        if quota >= available:
            cursor = "*"
            while cursor:
                params = {**base_params(cfg), "filter": year_filter, "per-page": 200,
                          "cursor": cursor, "select": ",".join(FIELDS)}
                data = get_json(session, params)
                results = data.get("results", [])
                if not results:
                    break
                for work in results:
                    fh.write(json.dumps(work) + "\n")
                n += len(results)
                cursor = data.get("meta", {}).get("next_cursor")
                time.sleep(0.1)
        else:
            # OpenAlex random sampling: the same seed returns the same sample, so the pull is reproducible.
            page = 1
            while n < quota:
                params = {**base_params(cfg), "filter": year_filter, "per-page": 200,
                          "page": page, "sample": quota,
                          "seed": cfg["topics"]["random_state"], "select": ",".join(FIELDS)}
                data = get_json(session, params)
                results = data.get("results", [])
                if not results:
                    break
                for work in results:
                    fh.write(json.dumps(work) + "\n")
                n += len(results)
                page += 1
                time.sleep(0.1)
    tmp_path.rename(out_path)
    log.info("%s: %s of %s records", year, n, available)
    return n


def main() -> None:
    cfg = load_config()
    c = cfg["corpus"]
    corpus_expr = f"{or_group(c['ai_terms'])} AND {or_group(c['bio_terms'])}"
    corpus_filter = build_filter(cfg, corpus_expr)
    baseline_filter = build_filter(cfg, or_group(c["baseline_terms"]))

    session = requests.Session()
    session.headers["User-Agent"] = f"cset-sample (mailto:{cfg['project']['contact_email']})"

    log.info("Counting records by year")
    counts = {
        "ai_bio": yearly_counts(session, cfg, corpus_filter),
        "bio_baseline": yearly_counts(session, cfg, baseline_filter),
    }
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    (raw_dir / "counts_by_year.json").write_text(json.dumps(counts, indent=2))
    log.info("Corpus size in OpenAlex: %s", sum(counts["ai_bio"].values()))

    quotas = year_quotas(counts["ai_bio"], c.get("max_records"))
    downloaded = {}
    for year in sorted(quotas):
        downloaded[year] = fetch_year(session, cfg, corpus_expr, year,
                                      quotas[year], counts["ai_bio"][year])
    n = sum(downloaded.values())

    manifest = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "corpus_filter": corpus_filter,
        "baseline_filter": baseline_filter,
        "records_downloaded": n,
        "records_in_openalex": sum(counts["ai_bio"].values()),
        "max_records_cap": c.get("max_records"),
        "downloaded_by_year": downloaded,
        # Sampling weight = records in OpenAlex / records downloaded, per year.
        "sampling_weights": {y: counts["ai_bio"][y] / downloaded[y]
                             for y in downloaded if downloaded[y]},
    }
    (raw_dir / "fetch_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Done: %s records written", n)


if __name__ == "__main__":
    main()
