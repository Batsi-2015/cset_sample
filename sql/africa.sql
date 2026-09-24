-- African participation: output by country, and how much of it involves partners outside Africa.
WITH af AS (
    SELECT country_code, SUM(credit_weighted) AS output, COUNT(DISTINCT work_id) AS papers_sampled
    FROM v_country_credit
    WHERE region = 'Africa' AND year BETWEEN {start_year} AND {end_year}
    GROUP BY country_code
)
SELECT
    af.country_code,
    r.country_name,
    af.output,
    af.papers_sampled,
    af.output / (SELECT SUM(credit_weighted) FROM v_country_credit
                 WHERE year BETWEEN {start_year} AND {end_year})   AS world_share
FROM af
JOIN country_regions r USING (country_code)
ORDER BY output DESC;
