CREATE OR REPLACE TABLE f009 AS
WITH base AS (
    SELECT
        ru.runner_id,
        ru.race_id,
        ru.horse_id,
        ru.weight_lbs AS current_weight,
        ru.jockey_id AS current_jockey_id,
        ra.race_type,
        ra.distance_furlongs AS current_distance,
        ra.going_code,
        ra.decision_cutoff_utc
    FROM runners ru
    JOIN races ra ON ra.race_id = ru.race_id
),

-- Horse's prior runs with time, distance, weight, jockey
prior AS (
    SELECT
        b.runner_id,
        b.race_id,
        b.decision_cutoff_utc,
        b.current_weight,
        b.current_distance,
        b.current_jockey_id,
        b.race_type,
        b.going_code,
        hh.scheduled_off_utc AS prior_off,
        hh.distance_furlongs AS prior_distance,
        hh.weight_lbs AS prior_weight,
        hh.jockey_id AS prior_jockey_id,
        hh.finishing_position,
        hh.field_size AS prior_field_size,
        hh.btn_lengths AS prior_btn,
        hh.going_code AS prior_going,
        res.official_time_secs AS prior_time_secs,
        ROW_NUMBER() OVER (PARTITION BY b.runner_id ORDER BY hh.scheduled_off_utc DESC) AS rn
    FROM base b
    JOIN horse_history hh
        ON hh.horse_id = b.horse_id
       AND hh.scheduled_off_utc < b.decision_cutoff_utc
    LEFT JOIN results res
        ON res.race_id = hh.race_id
       AND res.horse_id = hh.horse_id
),

-- Speed figure: seconds per furlong, adjusted relative to going/distance median
-- Only for runs with valid time and distance
speed_raw AS (
    SELECT
        p.runner_id,
        p.prior_off,
        p.prior_time_secs,
        p.prior_distance,
        p.prior_going,
        p.rn,
        CASE
            WHEN p.prior_time_secs IS NOT NULL AND p.prior_distance IS NOT NULL AND p.prior_distance > 0
            THEN p.prior_time_secs / p.prior_distance
            ELSE NULL
        END AS secs_per_furlong
    FROM prior p
    WHERE p.rn <= 10
),

-- Going/distance median pace for normalization
going_dist_median AS (
    SELECT
        hh.going_code,
        ROUND(hh.distance_furlongs) AS dist_round,
        MEDIAN(res.official_time_secs / hh.distance_furlongs) AS median_spf
    FROM horse_history hh
    JOIN results res ON res.race_id = hh.race_id AND res.horse_id = hh.horse_id
    WHERE res.official_time_secs IS NOT NULL
      AND hh.distance_furlongs IS NOT NULL AND hh.distance_furlongs > 0
      AND res.official_time_secs / hh.distance_furlongs BETWEEN 5 AND 25
    GROUP BY 1, 2
    HAVING COUNT(*) >= 20
),

-- Normalized speed figure per run
speed_norm AS (
    SELECT
        sr.runner_id,
        sr.rn,
        CASE
            WHEN sr.secs_per_furlong IS NOT NULL AND gdm.median_spf IS NOT NULL
            THEN (gdm.median_spf - sr.secs_per_furlong) * 100
            ELSE NULL
        END AS speed_figure
    FROM speed_raw sr
    LEFT JOIN going_dist_median gdm
        ON gdm.going_code = sr.prior_going
       AND gdm.dist_round = ROUND(sr.prior_distance)
),

-- Jockey quality lookup (90-day win rate per jockey)
jockey_quality AS (
    SELECT
        jh.jockey_id,
        b.runner_id,
        AVG(CASE WHEN jh.won THEN 1.0 ELSE 0.0 END) FILTER (
            WHERE jh.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
              AND jh.scheduled_off_utc < b.decision_cutoff_utc
        ) AS jockey_win_rate
    FROM base b
    JOIN jockey_history jh ON jh.scheduled_off_utc < b.decision_cutoff_utc
    WHERE jh.jockey_id = b.current_jockey_id
    GROUP BY 1, 2
),

-- Horse's usual jockey quality (avg jockey quality across last 5 runs)
usual_jockey AS (
    SELECT
        p.runner_id,
        AVG(jq_hist.jockey_win_rate) AS avg_prior_jockey_quality
    FROM prior p
    LEFT JOIN LATERAL (
        SELECT AVG(CASE WHEN jh.won THEN 1.0 ELSE 0.0 END) AS jockey_win_rate
        FROM jockey_history jh
        WHERE jh.jockey_id = p.prior_jockey_id
          AND jh.scheduled_off_utc < p.prior_off
          AND jh.scheduled_off_utc >= p.prior_off - INTERVAL 90 DAY
    ) jq_hist ON TRUE
    WHERE p.rn <= 5 AND p.prior_jockey_id IS NOT NULL
    GROUP BY 1
),

-- Trainer 14-day form
trainer_14d AS (
    SELECT
        b.runner_id,
        AVG(CASE WHEN th.won THEN 1.0 ELSE 0.0 END) AS trainer_win_rate_14d,
        COUNT(*) AS trainer_runs_14d
    FROM base b
    JOIN runners ru ON ru.runner_id = b.runner_id
    JOIN trainer_history th
        ON COALESCE(NULLIF(TRIM(th.trainer_name), ''), 'Unknown') = COALESCE(NULLIF(TRIM(ru.trainer_name), ''), 'Unknown')
       AND th.scheduled_off_utc < b.decision_cutoff_utc
       AND th.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 14 DAY
    GROUP BY 1
),

-- Aggregate per runner
agg AS (
    SELECT
        p.runner_id,
        p.race_id,
        p.race_type,
        -- Trip change from last run
        CASE WHEN p.rn = 1 AND p.current_distance IS NOT NULL AND p.prior_distance IS NOT NULL
             THEN p.current_distance - p.prior_distance
             ELSE NULL
        END AS trip_change_furlongs,
        -- Weight change from last run
        CASE WHEN p.rn = 1 AND p.current_weight IS NOT NULL AND p.prior_weight IS NOT NULL
             THEN p.current_weight - p.prior_weight
             ELSE NULL
        END AS weight_change_lbs,
        -- Beaten lengths in last run (0 = won)
        CASE WHEN p.rn = 1 THEN COALESCE(p.prior_btn, 0.0) ELSE NULL END AS last_run_btn_lengths,
        -- Average beaten lengths last 3 runs
        AVG(COALESCE(p.prior_btn, 0.0)) FILTER (WHERE p.rn <= 3 AND p.prior_btn IS NOT NULL) AS avg_btn_last_3
    FROM prior p
    WHERE p.rn = 1
    GROUP BY 1, 2, 3, 4, 5, 6
),

-- Speed aggregates
speed_agg AS (
    SELECT
        runner_id,
        AVG(speed_figure) FILTER (WHERE rn <= 3) AS avg_speed_last_3,
        MAX(speed_figure) FILTER (WHERE rn <= 5) AS best_speed_last_5,
        MAX(speed_figure) FILTER (WHERE rn = 1) AS last_run_speed
    FROM speed_norm
    WHERE speed_figure IS NOT NULL
    GROUP BY 1
)

SELECT
    b.runner_id,
    b.race_id,

    -- Speed figures (more meaningful for flat, less for jumps)
    sa.avg_speed_last_3,
    sa.best_speed_last_5,
    sa.last_run_speed,
    CASE WHEN LOWER(b.race_type) IN ('chase', 'hurdle', 'nh flat') THEN 1 ELSE 0 END AS is_jumps,

    -- Trip change
    a.trip_change_furlongs,

    -- Weight change
    a.weight_change_lbs,

    -- Beaten lengths
    a.last_run_btn_lengths,
    a.avg_btn_last_3,

    -- Jockey booking signal: current jockey quality minus horse's usual jockey quality
    CASE WHEN jq.jockey_win_rate IS NOT NULL AND uj.avg_prior_jockey_quality IS NOT NULL
         THEN jq.jockey_win_rate - uj.avg_prior_jockey_quality
         ELSE NULL
    END AS jockey_upgrade_signal,

    -- Trainer 14-day hot streak
    t14.trainer_win_rate_14d,
    COALESCE(t14.trainer_runs_14d, 0) AS trainer_runs_14d,

    b.decision_cutoff_utc - INTERVAL 1 SECOND AS event_timestamp_utc,
    b.decision_cutoff_utc
FROM base b
LEFT JOIN agg a ON a.runner_id = b.runner_id
LEFT JOIN speed_agg sa ON sa.runner_id = b.runner_id
LEFT JOIN jockey_quality jq ON jq.runner_id = b.runner_id
LEFT JOIN usual_jockey uj ON uj.runner_id = b.runner_id
LEFT JOIN trainer_14d t14 ON t14.runner_id = b.runner_id;
