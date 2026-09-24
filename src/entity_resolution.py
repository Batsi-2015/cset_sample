"""Step 3: Resolve raw affiliation strings to institutions, and measure how well it works.

OpenAlex already links most authors to institutions, but some raw affiliation strings
are left unresolved. Those authors silently drop out of any institution or country
analysis. This step tries to recover them and, just as importantly, reports how
accurate the recovery is.

Method (a transparent cascade, cheapest and most precise first)
  1. Normalize text: lowercase, strip accents and punctuation, expand abbreviations.
  2. Exact lookup against strings OpenAlex has already resolved elsewhere in the corpus
     (a learned alias table, majority vote when a string maps to several institutions).
  3. Fuzzy match: split the string into comma-separated segments, keep segments that
     look like organizations, and match them to known institution names with rapidfuzz.
     Accept only matches at or above the configured threshold.

Evaluation
  A random holdout of strings with a known, single institution is hidden from the alias
  table, pushed through the cascade, and scored. Precision is the share of predictions
  that are correct; recall is the share of holdout strings correctly resolved.

Outputs (DuckDB)
  institutions             institution dimension (id, name, country, type)
  resolved_affiliations    recovered links for previously unresolved strings
  er_evaluation            precision / recall / coverage by method
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz import fuzz, process

from common import connect, get_logger, load_config

log = get_logger("entity_resolution")

ABBREVIATIONS = {
    r"\buniv\b": "university", r"\bunivers\b": "university", r"\binst\b": "institute",
    r"\bdept\b": "department", r"\bctr\b": "center", r"\bcentre\b": "center",
    r"\bhosp\b": "hospital", r"\bnatl\b": "national", r"\bacad\b": "academy",
    r"\bsci\b": "sciences", r"\bmed\b": "medical", r"\btech\b": "technology",
    r"\bcoll\b": "college", r"\blab\b": "laboratory", r"\bres\b": "research",
}
ORG_WORDS = ("university", "institute", "hospital", "college", "center", "academy",
             "school", "laboratory", "foundation", "council", "company", "inc", "ltd",
             "gmbh", "corporation", "agency", "ministry", "universite", "universidad",
             "universitat", "universita", "universidade", "klinikum", "clinic")
# Department-level segments carry little signal about which institution it is.
SKIP_PREFIXES = ("department", "division", "faculty", "unit", "section", "program")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9, ]+", " ", text)
    for pattern, repl in ABBREVIATIONS.items():
        text = re.sub(pattern, repl, text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def org_segments(norm: str) -> list[str]:
    segs = [s.strip() for s in norm.split(",") if s.strip()]
    keep = [s for s in segs if any(w in s.split() for w in ORG_WORDS)
            and not s.startswith(SKIP_PREFIXES)]
    return keep or segs[:2]


class Resolver:
    def __init__(self, alias_rows: pd.DataFrame, institutions: pd.DataFrame, threshold: int):
        votes: dict[str, Counter] = defaultdict(Counter)
        for norm, inst in zip(alias_rows["norm"], alias_rows["resolved_institution_id"]):
            votes[norm][inst] += 1
        self.alias = {k: c.most_common(1)[0][0] for k, c in votes.items()}

        self.names = institutions["norm_name"].tolist()
        self.ids = institutions["institution_id"].tolist()
        self.threshold = threshold

    def resolve(self, norm: str) -> tuple[str | None, str, float]:
        if norm in self.alias:
            return self.alias[norm], "exact_alias", 100.0
        best = (None, "unresolved", 0.0)
        for seg in org_segments(norm):
            if len(seg) < 6:
                continue
            hit = process.extractOne(seg, self.names, scorer=fuzz.token_sort_ratio,
                                     score_cutoff=self.threshold)
            if hit and hit[1] > best[2]:
                best = (self.ids[hit[2]], "fuzzy_name", float(hit[1]))
        return best


def resolve_unique(norms: pd.Series, resolver: Resolver) -> pd.Series:
    """Resolve each distinct string once; affiliation strings repeat a lot."""
    lookup = {n: resolver.resolve(n) for n in norms.unique()}
    return norms.map(lookup)


def evaluate(pred: pd.Series, truth: pd.Series, method: pd.Series) -> pd.DataFrame:
    rows = []
    df = pd.DataFrame({"pred": pred, "truth": truth, "method": method})
    for label, sub in [("overall", df)] + [(m, df[df.method == m]) for m in ("exact_alias", "fuzzy_name")]:
        predicted = sub.pred.notna()
        correct = (sub.pred == sub.truth) & predicted
        rows.append({
            "method": label,
            "n_holdout": len(df) if label == "overall" else None,
            "n_predicted": int(predicted.sum()),
            "n_correct": int(correct.sum()),
            "precision": correct.sum() / predicted.sum() if predicted.sum() else None,
            "recall": correct.sum() / len(df) if len(df) else None,
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    er = cfg["entity_resolution"]
    con = connect(cfg)

    con.execute("""
        CREATE OR REPLACE TABLE institutions AS
        SELECT institution_id,
               mode(institution_name)   AS institution_name,
               mode(country_code)       AS country_code,
               mode(institution_type)   AS institution_type,
               count(DISTINCT work_id)  AS n_works
        FROM authorship_inst
        WHERE institution_id IS NOT NULL
        GROUP BY institution_id
    """)
    institutions = con.execute("SELECT * FROM institutions").df()
    institutions["norm_name"] = institutions["institution_name"].fillna("").map(normalize)

    raw = con.execute("SELECT * FROM affiliations_raw").df()
    raw["norm"] = raw["raw_affiliation"].fillna("").map(normalize)
    known = raw[raw.resolved_institution_id.notna()]
    unresolved = raw[raw.n_resolved_institutions == 0]
    log.info("%s raw strings: %s resolved by OpenAlex, %s unresolved",
             len(raw), len(known), len(unresolved))

    # Holdout evaluation: hide a sample of known strings (by work, to avoid leakage
    # from the same paper) and see whether the cascade recovers the right institution.
    per_work = known.groupby("work_id").size().sample(frac=1.0, random_state=cfg["topics"]["random_state"])
    holdout_works = set(per_work.index[per_work.cumsum() <= er["holdout_size"]])
    holdout = known[known.work_id.isin(holdout_works)]
    train = known[~known.work_id.isin(holdout_works)]
    # Also remove holdout strings from the alias table, so the test is truly unseen text.
    train = train[~train.norm.isin(set(holdout.norm))]

    eval_resolver = Resolver(train, institutions, er["fuzzy_threshold"])
    res = resolve_unique(holdout.norm, eval_resolver)
    metrics = evaluate(res.map(lambda r: r[0]), holdout.resolved_institution_id,
                       res.map(lambda r: r[1]))
    log.info("Holdout evaluation:\n%s", metrics.to_string(index=False))

    # Production pass: full alias table, applied to the unresolved strings.
    resolver = Resolver(known, institutions, er["fuzzy_threshold"])
    out = unresolved[["work_id", "author_order", "raw_affiliation"]].copy()
    results = resolve_unique(unresolved.norm, resolver)
    out["institution_id"] = results.map(lambda r: r[0])
    out["method"] = results.map(lambda r: r[1])
    out["score"] = results.map(lambda r: r[2])

    coverage = out.method.value_counts().rename_axis("method").reset_index(name="n")
    coverage["share_of_unresolved"] = coverage.n / max(len(out), 1)
    log.info("Coverage on unresolved strings:\n%s", coverage.to_string(index=False))

    metrics["stage"] = "holdout"
    coverage["stage"] = "unresolved_pass"
    for name, df in {"resolved_affiliations": out, "er_evaluation": metrics,
                     "er_coverage": coverage}.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")

    # Enriched author-country table: OpenAlex links plus recovered links.
    con.execute("""
        CREATE OR REPLACE TABLE author_countries_enriched AS
        SELECT work_id, author_order, country_code, weight, 'openalex' AS source
        FROM author_countries
        UNION ALL
        SELECT r.work_id, r.author_order, i.country_code,
               1.0 / w.n_authors / count(*) OVER (PARTITION BY r.work_id, r.author_order) AS weight,
               'recovered' AS source
        FROM (SELECT DISTINCT work_id, author_order, institution_id
              FROM resolved_affiliations WHERE institution_id IS NOT NULL) r
        JOIN institutions i USING (institution_id)
        JOIN works w USING (work_id)
        WHERE i.country_code IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM author_countries a
                          WHERE a.work_id = r.work_id AND a.author_order = r.author_order)
    """)
    con.close()


if __name__ == "__main__":
    main()
