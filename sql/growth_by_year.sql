-- How fast is AI x biotech research growing, in absolute terms and as a share of all biotech output?
SELECT
    year,
    ai_bio,
    bio_baseline,
    ai_bio / NULLIF(bio_baseline, 0)                               AS ai_share_of_bio,
    ai_bio / NULLIF(LAG(ai_bio) OVER (ORDER BY year), 0) - 1       AS yoy_growth
FROM yearly_counts
WHERE year BETWEEN {start_year} AND {end_year}
ORDER BY year;
