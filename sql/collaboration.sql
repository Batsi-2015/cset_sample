-- International collaboration over time, including direct US-China co-authorship.
SELECT
    year,
    SUM(sampling_weight)                                                          AS papers,
    SUM(CASE WHEN n_countries > 1 THEN sampling_weight END) / SUM(sampling_weight) AS intl_collab_rate,
    SUM(CASE WHEN has_us AND has_cn THEN sampling_weight END)
        / NULLIF(SUM(CASE WHEN has_us OR has_cn THEN sampling_weight END), 0)     AS us_cn_share_of_us_or_cn,
    SUM(CASE WHEN has_us AND has_cn THEN sampling_weight END)
        / NULLIF(SUM(CASE WHEN has_us THEN sampling_weight END), 0)               AS us_papers_with_cn
FROM v_work_collab
WHERE year BETWEEN {start_year} AND {end_year}
GROUP BY year
ORDER BY year;
