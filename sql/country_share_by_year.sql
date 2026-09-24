-- Share of world AI x biotech output by country bloc and year (fractional counting, weighted).
WITH yearly AS (
    SELECT year, bloc, SUM(credit_weighted) AS output
    FROM v_country_credit
    WHERE year BETWEEN {start_year} AND {end_year}
    GROUP BY year, bloc
)
SELECT
    year,
    bloc,
    output,
    output / SUM(output) OVER (PARTITION BY year) AS share
FROM yearly
ORDER BY year, bloc;
