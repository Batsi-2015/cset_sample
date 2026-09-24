"""Step 4: Cluster papers into research topics and score which topics are emerging.

Pipeline
  1. Text = title + abstract.
  2. Embed with a sentence-transformer (falls back to TF-IDF + SVD if torch is unavailable).
  3. Cluster embeddings with k-means (MiniBatch for scale).
  4. Label each cluster with class-based TF-IDF (c-TF-IDF) terms plus the papers
     closest to its centroid, so a human can sanity-check every label.
  5. Emergence: compare each cluster's share of the corpus in a recent window against
     an earlier window (weighted by the per-year sampling weights). Fast-growing,
     reasonably large clusters are flagged as emerging.

This is a small-scale version of the idea behind CSET's Map of Science, where
research clusters are built from citation links and their growth is tracked over time.
Here clusters are built from text, which works on any corpus without a citation graph.

Outputs (DuckDB): work_topics, clusters, cluster_year, cluster_examples
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from common import connect, get_logger, load_config, project_path

log = get_logger("topics")

# Words that appear in nearly every AI x bio abstract and say nothing about the topic.
DOMAIN_STOP = {
    "learning", "machine", "deep", "model", "models", "data", "using", "based", "method",
    "methods", "approach", "results", "study", "analysis", "used", "use", "new", "proposed",
    "performance", "network", "networks", "neural", "artificial", "intelligence", "algorithm",
    "algorithms", "high", "different", "paper", "present", "show", "shows", "however",
    "also", "large", "important", "provide", "novel", "significant", "accuracy",
}


def embed(texts: list[str], cfg: dict) -> np.ndarray:
    t = cfg["topics"]
    cache = project_path(cfg["paths"]["processed_dir"]) / f"embeddings_{t['embedding_backend']}.npy"
    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(texts):
            log.info("Loaded cached embeddings %s", arr.shape)
            return arr
    if t["embedding_backend"] == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(t["embedding_model"])
            arr = model.encode(texts, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True)
            np.save(cache, arr)
            return arr
        except ImportError:
            log.warning("sentence-transformers not installed, falling back to TF-IDF + SVD")
    tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                            stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOP))
    X = tfidf.fit_transform(texts)
    n_comp = min(256, X.shape[1] - 1)
    arr = normalize(TruncatedSVD(n_comp, random_state=t["random_state"]).fit_transform(X))
    np.save(cache, arr)
    return arr


def layout_2d(emb: np.ndarray, seed: int) -> np.ndarray:
    try:
        import umap
        return umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                         random_state=seed).fit_transform(emb)
    except ImportError:
        log.warning("umap-learn not installed, using PCA for the 2D map")
        return PCA(2, random_state=seed).fit_transform(emb)


def ctfidf_labels(texts: pd.Series, labels: np.ndarray, top_n: int = 6) -> dict[int, list[str]]:
    docs = texts.groupby(labels).apply(" ".join)
    cv = CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=60_000,
                         stop_words=list(ENGLISH_STOP_WORDS | DOMAIN_STOP))
    tf = cv.fit_transform(docs.values).astype(float)
    tf = normalize(tf, norm="l1")
    idf = np.log(1 + tf.shape[0] / (np.asarray((tf > 0).sum(axis=0)).ravel() + 1))
    scores = tf.multiply(idf).tocsr()
    vocab = np.array(cv.get_feature_names_out())
    out = {}
    for i, cluster in enumerate(docs.index):
        row = scores.getrow(i).toarray().ravel()
        terms = []
        for term in vocab[row.argsort()[::-1]]:
            # skip a unigram already covered by a chosen bigram ("cell" after "single cell")
            # and simple plural variants ("cells" after "cell")
            stem = term.rstrip("s")
            if any(term in t.split() or t in term.split() or t.rstrip("s") == stem for t in terms):
                continue
            terms.append(term)
            if len(terms) == top_n:
                break
        out[int(cluster)] = terms
    return out


def main() -> None:
    cfg = load_config()
    t = cfg["topics"]
    con = connect(cfg)
    works = con.execute("""
        SELECT work_id, title, coalesce(abstract, '') AS abstract, year,
               sampling_weight, citation_percentile
        FROM works WHERE title IS NOT NULL ORDER BY work_id
    """).df()
    works["text"] = (works.title + ". " + works.abstract).str.slice(0, 4000)
    log.info("Embedding %s documents", len(works))

    emb = embed(works.text.tolist(), cfg)
    k = min(t["n_clusters"], max(2, len(works) // 20))
    km = MiniBatchKMeans(n_clusters=k, random_state=t["random_state"], batch_size=4096, n_init=5)
    works["cluster"] = km.fit_predict(emb)

    xy = layout_2d(emb, t["random_state"])
    works["x"], works["y"] = xy[:, 0], xy[:, 1]

    terms = ctfidf_labels(works.text.str.lower(), works.cluster.values)

    # Representative papers: nearest to each centroid.
    dist = np.linalg.norm(emb - km.cluster_centers_[works.cluster.values], axis=1)
    works["dist_to_centroid"] = dist
    examples = (works.sort_values("dist_to_centroid").groupby("cluster").head(3)
                [["cluster", "work_id", "title", "year"]])

    # Emergence scoring on sampling-weighted counts.
    cy = (works.groupby(["cluster", "year"]).sampling_weight.sum()
          .rename("n_weighted").reset_index())
    year_tot = cy.groupby("year").n_weighted.transform("sum")
    cy["share"] = cy.n_weighted / year_tot

    e0, e1 = t["early_window"]
    r0, r1 = t["recent_window"]
    def window_share(lo, hi):
        sub = cy[(cy.year >= lo) & (cy.year <= hi)]
        return sub.groupby("cluster").n_weighted.sum() / sub.n_weighted.sum()
    early, recent = window_share(e0, e1), window_share(r0, r1)

    clusters = pd.DataFrame({"cluster": range(k)})
    clusters["top_terms"] = clusters.cluster.map(lambda c: ", ".join(terms.get(c, [])))
    clusters["label"] = clusters.cluster.map(lambda c: " / ".join(terms.get(c, [])[:3]))
    clusters["n_works"] = clusters.cluster.map(works.cluster.value_counts()).fillna(0).astype(int)
    clusters["early_share"] = clusters.cluster.map(early).fillna(0)
    clusters["recent_share"] = clusters.cluster.map(recent).fillna(0)
    # Smoothed ratio: a 0.5 percentage point floor keeps clusters that barely existed
    # in the early window from producing meaningless ratios in the hundreds.
    eps = 0.005
    clusters["growth_ratio"] = (clusters.recent_share + eps) / (clusters.early_share + eps)
    clusters["median_year"] = clusters.cluster.map(works.groupby("cluster").year.median())
    clusters["mean_citation_pct"] = clusters.cluster.map(
        works.groupby("cluster").citation_percentile.mean())
    size_ok = clusters.n_works >= 100  # ignore tiny clusters whose growth is noise
    growth_cut = clusters.growth_ratio.quantile(0.75)
    clusters["emerging"] = size_ok & (clusters.growth_ratio >= growth_cut)
    clusters = clusters.sort_values("growth_ratio", ascending=False)
    log.info("Top clusters by growth:\n%s",
             clusters[["cluster", "label", "n_works", "growth_ratio"]].head(10).to_string(index=False))

    tables = {
        "work_topics": works[["work_id", "cluster", "x", "y"]],
        "clusters": clusters,
        "cluster_year": cy,
        "cluster_examples": examples,
    }
    for name, df in tables.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    con.close()


if __name__ == "__main__":
    main()
