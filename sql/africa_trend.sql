-- Africa's share of world output over time, and the share of African-authored papers
-- that also include a non-African partner.
WITH credit AS (
    SELECT year,
           SUM(CASE WHEN region = 'Africa' THEN credit_weighted ELSE 0 END) AS africa_output,
           SUM(credit_weighted)                                              AS world_output
    FROM v_country_credit
    WHERE year BETWEEN {start_year} AND {end_year}
    GROUP BY year
),
collab AS (
    SELECT year,
           SUM(CASE WHEN has_africa AND has_non_africa THEN sampling_weight END)
             / NULLIF(SUM(CASE WHEN has_africa THEN sampling_weight END), 0) AS africa_with_external_partner
    FROM v_work_collab
    WHERE year BETWEEN {start_year} AND {end_year}
    GROUP BY year
)
SELECT credit.year,
       africa_output / world_output AS africa_share,
       africa_with_external_partner
FROM credit LEFT JOIN collab USING (year)
ORDER BY year;
