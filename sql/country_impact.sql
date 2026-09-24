-- Output versus impact by bloc.
-- impact_ratio > 1 means a bloc's share of the top 10% most cited papers (field and year
-- normalized by OpenAlex) is larger than its share of all papers.
-- The most recent year is excluded because citations have not had time to accumulate.
WITH base AS (
    SELECT bloc,
           SUM(credit_weighted)                                        AS output,
           SUM(CASE WHEN top10pct THEN credit_weighted ELSE 0 END)     AS output_top10,
           SUM(CASE WHEN top1pct  THEN credit_weighted ELSE 0 END)     AS output_top1
    FROM v_country_credit
    WHERE year BETWEEN {impact_start} AND {impact_end}
    GROUP BY bloc
)
SELECT
    bloc,
    output,
    output / SUM(output) OVER ()                                       AS share_all,
    output_top10 / NULLIF(SUM(output_top10) OVER (), 0)                AS share_top10,
    output_top1  / NULLIF(SUM(output_top1)  OVER (), 0)                AS share_top1,
    (output_top10 / NULLIF(SUM(output_top10) OVER (), 0))
        / NULLIF(output / SUM(output) OVER (), 0)                      AS impact_ratio
FROM base
ORDER BY output DESC;
