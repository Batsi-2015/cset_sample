"""Step 2: Flatten raw OpenAlex JSON into analysis-ready DuckDB tables.

Tables created
  works                 one row per paper (metadata, reconstructed abstract, impact, sampling weight)
  authorship_inst       one row per author x institution link (OpenAlex-resolved)
  affiliations_raw      one row per author x raw affiliation string (input to entity resolution)
  author_countries      fractional country credit per author (1 / n_authors / n_countries)
  keywords              OpenAlex keywords with scores
  yearly_counts         AI x bio and baseline totals from OpenAlex group_by
  country_regions       ISO2 country code to region / EU27 lookup

DuckDB is used as a local stand-in for BigQuery: same SQL dialect family, columnar,
and fast enough for a few hundred thousand rows on a laptop.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd

from common import ROOT, connect, get_logger, load_config, project_path

log = get_logger("build_db")


def reconstruct_abstract(inverted: dict | None) -> str | None:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the running text."""
    if not inverted:
        return None
    positions = [(pos, word) for word, idxs in inverted.items() for pos in idxs]
    return " ".join(word for _, word in sorted(positions))


def short_id(url: str | None) -> str | None:
    return url.rsplit("/", 1)[-1] if url else None


def parse_work(w: dict, weight: float):
    wid = short_id(w["id"])
    cnp = w.get("citation_normalized_percentile") or {}
    pt = w.get("primary_topic") or {}
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}

    work = {
        "work_id": wid,
        "doi": w.get("doi"),
        "title": w.get("title"),
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        "year": w.get("publication_year"),
        "pub_date": w.get("publication_date"),
        "type": w.get("type"),
        "language": w.get("language"),
        "cited_by_count": w.get("cited_by_count") or 0,
        "fwci": w.get("fwci"),
        "citation_percentile": cnp.get("value"),
        "top1pct": cnp.get("is_in_top_1_percent"),
        "top10pct": cnp.get("is_in_top_10_percent"),
        "source_name": src.get("display_name"),
        "source_type": src.get("type"),
        "topic": pt.get("display_name"),
        "subfield": (pt.get("subfield") or {}).get("display_name"),
        "field": (pt.get("field") or {}).get("display_name"),
        "domain": (pt.get("domain") or {}).get("display_name"),
        "n_authors": len(w.get("authorships") or []),
        "sampling_weight": weight,
    }

    inst_rows, raw_rows, country_rows = [], [], []
    authorships = w.get("authorships") or []
    n_auth = max(len(authorships), 1)
    for order, a in enumerate(authorships):
        author = a.get("author") or {}
        aid = short_id(author.get("id"))
        insts = a.get("institutions") or []
        for inst in insts:
            inst_rows.append({
                "work_id": wid, "author_order": order, "author_id": aid,
                "author_name": author.get("display_name"),
                "institution_id": short_id(inst.get("id")),
                "institution_name": inst.get("display_name"),
                "ror": inst.get("ror"),
                "country_code": inst.get("country_code"),
                "institution_type": inst.get("type"),
            })
        for raw in a.get("raw_affiliation_strings") or []:
            raw_rows.append({
                "work_id": wid, "author_order": order, "raw_affiliation": raw,
                "n_resolved_institutions": len(insts),
                "resolved_institution_id": short_id(insts[0]["id"]) if len(insts) == 1 else None,
            })
        countries = sorted(set(a.get("countries") or []))
        for cc in countries:
            country_rows.append({
                "work_id": wid, "author_order": order, "country_code": cc,
                "weight": 1.0 / n_auth / len(countries),
            })
    kw_rows = [{"work_id": wid, "keyword": k.get("display_name"), "score": k.get("score")}
               for k in (w.get("keywords") or [])]
    return work, inst_rows, raw_rows, country_rows, kw_rows


def main() -> None:
    cfg = load_config()
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    manifest = json.loads((raw_dir / "fetch_manifest.json").read_text())
    weights = {int(k): v for k, v in manifest.get("sampling_weights", {}).items()}

    works, insts, raws, countries, kws = [], [], [], [], []
    seen = set()
    for path in sorted(raw_dir.glob("works_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                w = json.loads(line)
                if w["id"] in seen:  # guard against duplicates across pages
                    continue
                seen.add(w["id"])
                parsed = parse_work(w, weights.get(w.get("publication_year"), 1.0))
                works.append(parsed[0])
                insts += parsed[1]
                raws += parsed[2]
                countries += parsed[3]
                kws += parsed[4]
    log.info("Parsed %s works, %s institution links, %s raw affiliations",
             len(works), len(insts), len(raws))

    counts = json.loads((raw_dir / "counts_by_year.json").read_text())
    years = sorted({int(y) for y in counts["ai_bio"]} | {int(y) for y in counts["bio_baseline"]})
    yearly = pd.DataFrame({
        "year": years,
        "ai_bio": [counts["ai_bio"].get(str(y), 0) for y in years],
        "bio_baseline": [counts["bio_baseline"].get(str(y), 0) for y in years],
    })

    frames = {
        "works": pd.DataFrame(works),
        "authorship_inst": pd.DataFrame(insts),
        "affiliations_raw": pd.DataFrame(raws),
        "author_countries": pd.DataFrame(countries),
        "keywords": pd.DataFrame(kws),
        "yearly_counts": yearly,
        "country_regions": pd.read_csv(ROOT / "data/reference/country_regions.csv",
                                       keep_default_na=False),
    }
    con = connect(cfg)
    for name, df in frames.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
        log.info("Table %-18s %8s rows", name, len(df))
    con.close()


if __name__ == "__main__":
    main()
