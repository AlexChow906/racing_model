-- Standard-race feature store selection with explicit Unknown categories.
-- This query is point-in-time safe because it uses decision_cutoff_utc from races.

WITH base AS (
    SELECT
        ra.race_id,
        ru.runner_id,
        ra.decision_cutoff_utc,
        COALESCE(NULLIF(TRIM(ru.trainer_name), ''), 'Unknown') AS trainer_name,
        COALESCE(NULLIF(TRIM(ru.jockey_name), ''), 'Unknown') AS jockey_name,
        CASE
            WHEN COALESCE(NULLIF(TRIM(ru.trainer_name), ''), 'Unknown') = 'Unknown' THEN 1
            ELSE 0
        END AS trainer_is_unknown,
        CASE
            WHEN COALESCE(NULLIF(TRIM(ru.jockey_name), ''), 'Unknown') = 'Unknown' THEN 1
            ELSE 0
        END AS jockey_is_unknown
    FROM races ra
    JOIN runners ru ON ru.race_id = ra.race_id
    WHERE ra.is_standard_race = TRUE
)
SELECT *
FROM base;
