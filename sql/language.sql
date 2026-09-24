-- Publication language (OpenAlex detects language from title and abstract).
SELECT
    COALESCE(language, 'unknown')                           AS language,
    SUM(sampling_weight)                                    AS papers,
    SUM(sampling_weight) / SUM(SUM(sampling_weight)) OVER () AS share
FROM works
GROUP BY 1
ORDER BY papers DESC;
