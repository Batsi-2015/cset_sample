-- Reusable analytic views. Executed once after the pipeline builds the base tables.

-- Country credit per paper, fractional counting.
-- Each author carries 1 / n_authors of the paper; an author with several countries splits
-- that credit evenly. Credit is multiplied by the year's sampling weight so that
-- totals estimate the full OpenAlex corpus, not just the downloaded sample.
CREATE OR REPLACE VIEW v_country_credit AS
SELECT
    c.work_id,
    w.year,
    c.country_code,
    CASE
        WHEN c.country_code = 'US' THEN 'United States'
        WHEN c.country_code = 'CN' THEN 'China'
        WHEN r.eu27 = 'EU27'       THEN 'EU27'
        WHEN c.country_code = 'GB' THEN 'United Kingdom'
        WHEN c.country_code = 'IN' THEN 'India'
        WHEN c.country_code = 'JP' THEN 'Japan'
        WHEN c.country_code = 'KR' THEN 'South Korea'
        ELSE 'Rest of world'
    END                                  AS bloc,
    r.region,
    SUM(c.weight)                        AS credit,
    SUM(c.weight * w.sampling_weight)    AS credit_weighted,
    w.top10pct,
    w.top1pct,
    w.citation_percentile
FROM author_countries_enriched c
JOIN works w USING (work_id)
LEFT JOIN country_regions r USING (country_code)
GROUP BY ALL;

-- Paper-level collaboration flags.
CREATE OR REPLACE VIEW v_work_collab AS
SELECT
    work_id,
    year,
    ANY_VALUE(sampling_weight)                                         AS sampling_weight,
    COUNT(DISTINCT country_code)                                       AS n_countries,
    BOOL_OR(country_code = 'US')                                       AS has_us,
    BOOL_OR(country_code = 'CN')                                       AS has_cn,
    BOOL_OR(region = 'Africa')                                         AS has_africa,
    BOOL_OR(region IS DISTINCT FROM 'Africa')                          AS has_non_africa
FROM (
    SELECT c.work_id, w.year, w.sampling_weight, c.country_code, r.region
    FROM author_countries_enriched c
    JOIN works w USING (work_id)
    LEFT JOIN country_regions r USING (country_code)
)
GROUP BY work_id, year;
