-- Who leads each research cluster? Bloc shares of fractional output per cluster.
WITH credit AS (
    SELECT t.cluster, v.bloc, SUM(v.credit_weighted) AS output
    FROM v_country_credit v
    JOIN work_topics t USING (work_id)
    GROUP BY t.cluster, v.bloc
)
SELECT
    c.cluster,
    c.label,
    c.n_works,
    c.growth_ratio,
    c.emerging,
    credit.bloc,
    credit.output / SUM(credit.output) OVER (PARTITION BY c.cluster) AS bloc_share
FROM credit
JOIN clusters c USING (cluster)
ORDER BY c.growth_ratio DESC, credit.bloc;
