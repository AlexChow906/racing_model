CREATE OR REPLACE TABLE f008 AS
WITH base AS (
    SELECT
        ru.runner_id,
        ru.race_id,
        ru.horse_id,
        ru.weight_lbs,
        ru.age,
        ru.official_rating,
        ra.decision_cutoff_utc
    FROM runners ru
    JOIN races ra ON ra.race_id = ru.race_id
),

-- Field-level aggregates for relative features
field_stats AS (
    SELECT
        ru.race_id,
        AVG(ru.weight_lbs) AS field_avg_weight,
        MAX(ru.weight_lbs) AS field_top_weight,
        MIN(ru.weight_lbs) AS field_low_weight,
        AVG(ru.official_rating) FILTER (WHERE ru.official_rating IS NOT NULL) AS field_avg_rating,
        MAX(ru.official_rating) FILTER (WHERE ru.official_rating IS NOT NULL) AS field_top_rating,
        MIN(ru.official_rating) FILTER (WHERE ru.official_rating IS NOT NULL) AS field_low_rating
    FROM runners ru
    GROUP BY 1
),

-- Career stats from horse_history (all prior runs)
career AS (
    SELECT
        b.runner_id,
        b.horse_id,
        COUNT(*) FILTER (WHERE hh.finishing_position IS NOT NULL) AS career_runs,
        AVG(CASE WHEN hh.won THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE hh.finishing_position IS NOT NULL) AS career_win_rate,
        AVG(CASE WHEN hh.finishing_position <= 3 THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE hh.finishing_position IS NOT NULL) AS career_place_rate,
        STDDEV_SAMP(hh.finishing_position::DOUBLE)
            FILTER (WHERE hh.finishing_position IS NOT NULL) AS career_position_stddev
    FROM base b
    JOIN horse_history hh
        ON hh.horse_id = b.horse_id
       AND hh.scheduled_off_utc < b.decision_cutoff_utc
       AND hh.finishing_position IS NOT NULL
    GROUP BY 1, 2
),

-- Consistency: stddev of last 10 finishing positions
recent_consistency AS (
    SELECT
        runner_id,
        STDDEV_SAMP(finishing_position::DOUBLE) AS recent_position_stddev
    FROM (
        SELECT
            b.runner_id,
            hh.finishing_position,
            ROW_NUMBER() OVER (PARTITION BY b.runner_id ORDER BY hh.scheduled_off_utc DESC) AS rn
        FROM base b
        JOIN horse_history hh
            ON hh.horse_id = b.horse_id
           AND hh.scheduled_off_utc < b.decision_cutoff_utc
           AND hh.finishing_position IS NOT NULL
    )
    WHERE rn <= 10
    GROUP BY 1
    HAVING COUNT(*) >= 3
)

SELECT
    b.runner_id,
    b.race_id,

    -- Weight features
    b.weight_lbs,
    CASE WHEN b.weight_lbs IS NOT NULL AND fs.field_top_weight IS NOT NULL
         THEN b.weight_lbs - fs.field_top_weight
         ELSE NULL
    END AS weight_vs_top,
    CASE WHEN b.weight_lbs IS NOT NULL AND fs.field_avg_weight IS NOT NULL
         THEN b.weight_lbs - fs.field_avg_weight
         ELSE NULL
    END AS weight_vs_field_avg,

    -- Age
    b.age AS horse_age,

    -- Official rating relative to field
    b.official_rating AS runner_official_rating,
    CASE WHEN b.official_rating IS NOT NULL AND fs.field_top_rating IS NOT NULL
         THEN b.official_rating - fs.field_top_rating
         ELSE NULL
    END AS rating_vs_top,
    CASE WHEN b.official_rating IS NOT NULL AND fs.field_avg_rating IS NOT NULL
         THEN b.official_rating - fs.field_avg_rating
         ELSE NULL
    END AS rating_vs_field_avg,
    fs.field_avg_rating,

    -- Career stats
    COALESCE(c.career_runs, 0) AS career_runs,
    c.career_win_rate,
    c.career_place_rate,

    -- Consistency
    COALESCE(rc.recent_position_stddev, c.career_position_stddev) AS position_consistency,

    b.decision_cutoff_utc - INTERVAL 1 SECOND AS event_timestamp_utc,
    b.decision_cutoff_utc
FROM base b
LEFT JOIN field_stats fs ON fs.race_id = b.race_id
LEFT JOIN career c ON c.runner_id = b.runner_id
LEFT JOIN recent_consistency rc ON rc.runner_id = b.runner_id;
