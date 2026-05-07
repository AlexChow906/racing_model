CREATE OR REPLACE TABLE f006 AS
WITH base AS (
    SELECT
        ru.runner_id,
        ru.race_id,
        ru.horse_id,
        ra.race_date,
        ra.field_size,
        ra.surface,
        ra.race_type,
        ra.distance_furlongs,
        ra.going_code,
        ra.decision_cutoff_utc
    FROM runners ru
    JOIN races ra ON ra.race_id = ru.race_id
),
prior AS (
    SELECT
        b.runner_id,
        hh.scheduled_off_utc,
        hh.finishing_position,
        hh.field_size,
        ROW_NUMBER() OVER (PARTITION BY b.runner_id ORDER BY hh.scheduled_off_utc DESC) AS rn_desc
    FROM base b
    LEFT JOIN horse_history hh
        ON hh.horse_id = b.horse_id
       AND hh.scheduled_off_utc < b.decision_cutoff_utc
),
runner_style AS (
    SELECT
        p.runner_id,
        CASE
            WHEN AVG(CASE WHEN p.finishing_position <= 2 THEN 1.0 ELSE 0.0 END)
                FILTER (WHERE p.rn_desc <= 3) >= 0.5
                THEN 1
            ELSE 0
        END AS is_front_runner,
        CASE
            WHEN AVG(
                CASE
                    WHEN COALESCE(p.field_size, 0) > 0
                     AND p.finishing_position >= CEIL(p.field_size * 0.75)
                        THEN 1.0
                    ELSE 0.0
                END
            ) FILTER (WHERE p.rn_desc <= 3) >= 0.5
                THEN 1
            ELSE 0
        END AS is_hold_up,
        MAX(p.scheduled_off_utc) AS latest_hist_ts
    FROM prior p
    GROUP BY 1
),
race_style AS (
    SELECT
        b.race_id,
        SUM(COALESCE(rs.is_front_runner, 0)) AS pace_front_runners,
        SUM(COALESCE(rs.is_hold_up, 0)) AS pace_hold_up_horses,
        MAX(rs.latest_hist_ts) AS race_latest_hist_ts
    FROM base b
    LEFT JOIN runner_style rs ON rs.runner_id = b.runner_id
    GROUP BY 1
)
SELECT
    b.runner_id,
    b.race_id,
    b.field_size,
    rsty.pace_front_runners,
    rsty.pace_hold_up_horses,
    CASE
        WHEN COALESCE(b.field_size, 0) <= 0 THEN NULL
        ELSE rsty.pace_front_runners::DOUBLE / b.field_size::DOUBLE
    END AS pace_pressure_index,
    CASE
        WHEN LOWER(COALESCE(b.surface, '')) IN ('aw', 'allweather', 'all_weather') THEN 1
        WHEN LOWER(COALESCE(b.surface, '')) = 'jumps' OR LOWER(COALESCE(b.race_type, '')) LIKE '%jumps%' THEN 2
        ELSE 0
    END AS surface_encoded,
    b.distance_furlongs,
    CASE
        WHEN LOWER(COALESCE(b.going_code, '')) LIKE '%firm%' THEN 1
        WHEN LOWER(COALESCE(b.going_code, '')) LIKE '%good%' AND LOWER(COALESCE(b.going_code, '')) NOT LIKE '%soft%' THEN 2
        WHEN LOWER(COALESCE(b.going_code, '')) LIKE '%good%' AND LOWER(COALESCE(b.going_code, '')) LIKE '%soft%' THEN 3
        WHEN LOWER(COALESCE(b.going_code, '')) LIKE '%soft%' AND LOWER(COALESCE(b.going_code, '')) NOT LIKE '%heavy%' THEN 4
        WHEN LOWER(COALESCE(b.going_code, '')) LIKE '%heavy%' THEN 5
        WHEN LOWER(COALESCE(b.surface, '')) IN ('aw', 'allweather', 'all_weather') THEN 6
        ELSE NULL
    END AS going_encoded,
    COALESCE(b.race_type, 'Unknown') AS race_type_encoded,
    EXTRACT(MONTH FROM b.race_date)::INTEGER AS race_month,
    EXTRACT(DOW FROM b.race_date)::INTEGER AS race_day_of_week,
    COALESCE(rsty.race_latest_hist_ts, b.decision_cutoff_utc - INTERVAL 1 SECOND) AS event_timestamp_utc,
    b.decision_cutoff_utc
FROM base b
LEFT JOIN race_style rsty ON rsty.race_id = b.race_id;
