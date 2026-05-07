CREATE OR REPLACE TABLE f001 AS
WITH base AS (
    SELECT
        ru.runner_id,
        ru.race_id,
        ru.horse_id,
        ru.headgear AS current_headgear,
        ra.course_id,
        ra.going_code,
        ra.distance_furlongs,
        ra.race_class,
        ra.decision_cutoff_utc
    FROM runners ru
    JOIN races ra ON ra.race_id = ru.race_id
),
prior AS (
    SELECT
        b.runner_id,
        b.race_id,
        b.decision_cutoff_utc,
        b.current_headgear,
        b.course_id,
        b.going_code,
        b.distance_furlongs,
        b.race_class,
        hh.scheduled_off_utc AS prior_scheduled_off_utc,
        hh.finishing_position,
        hh.won,
        hh.going_code AS prior_going_code,
        hh.distance_furlongs AS prior_distance_furlongs,
        hh.course_id AS prior_course_id,
        hh.official_rating,
        hh.race_class AS prior_race_class,
        hh.headgear AS prior_headgear,
        ROW_NUMBER() OVER (
            PARTITION BY b.runner_id
            ORDER BY hh.scheduled_off_utc DESC
        ) AS rn_desc
    FROM base b
    LEFT JOIN horse_history hh
        ON hh.horse_id = b.horse_id
       AND hh.scheduled_off_utc < b.decision_cutoff_utc
),
agg AS (
    SELECT
        runner_id,
        race_id,
        decision_cutoff_utc,
        LIST(finishing_position ORDER BY prior_scheduled_off_utc DESC)
            FILTER (WHERE rn_desc <= 3) AS horse_runs_last_3_positions,
        AVG(finishing_position::DOUBLE)
            FILTER (WHERE rn_desc <= 5) AS horse_runs_last_5_positions,
        SUM(CASE WHEN won THEN 1 ELSE 0 END)
            FILTER (WHERE rn_desc <= 5) AS horse_wins_last_5,
        AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE rn_desc <= 10) AS horse_win_rate_last_10,
        DATE_DIFF('day', MAX(prior_scheduled_off_utc), decision_cutoff_utc) AS horse_days_since_last_run,
        SUM(CASE WHEN prior_scheduled_off_utc >= decision_cutoff_utc - INTERVAL 90 DAY THEN 1 ELSE 0 END)
            AS horse_runs_last_90_days,
        AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE prior_going_code = going_code) AS horse_going_affinity,
        AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE ABS(COALESCE(prior_distance_furlongs, 0.0) - COALESCE(distance_furlongs, 0.0)) <= 1.0)
            AS horse_distance_affinity,
        AVG(CASE WHEN won THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE prior_course_id = course_id) AS horse_course_affinity,
        MAX(official_rating)
            FILTER (WHERE rn_desc <= 5) AS horse_best_rpr_last_5,
        AVG(prior_race_class::DOUBLE)
            FILTER (WHERE rn_desc <= 3) AS horse_avg_class_last_3,
        AVG(finishing_position::DOUBLE)
            FILTER (WHERE rn_desc <= 3) AS horse_avg_finish_last_3,
        REGR_SLOPE(
            finishing_position::DOUBLE,
            (6 - rn_desc)::DOUBLE
        ) FILTER (WHERE rn_desc <= 5) AS horse_form_trend,
        MAX(prior_headgear)
            FILTER (WHERE rn_desc = 1) AS last_run_headgear,
        MAX(prior_scheduled_off_utc) AS latest_prior_scheduled_off_utc
    FROM prior
    GROUP BY 1, 2, 3
)
SELECT
    a.runner_id,
    a.race_id,
    a.horse_runs_last_3_positions,
    a.horse_runs_last_5_positions,
    COALESCE(a.horse_wins_last_5, 0) AS horse_wins_last_5,
    a.horse_win_rate_last_10,
    a.horse_days_since_last_run,
    COALESCE(a.horse_runs_last_90_days, 0) AS horse_runs_last_90_days,
    a.horse_going_affinity,
    a.horse_distance_affinity,
    COALESCE(a.horse_course_affinity, 0.0) AS horse_course_affinity,
    a.horse_best_rpr_last_5,
    a.horse_avg_class_last_3,
    CASE
        WHEN a.horse_avg_class_last_3 IS NULL OR ra.race_class IS NULL THEN NULL
        ELSE ra.race_class - a.horse_avg_class_last_3
    END AS horse_class_delta,
    a.horse_form_trend,
    CASE
        WHEN COALESCE(NULLIF(TRIM(ru.headgear), ''), '') <> ''
             AND COALESCE(NULLIF(TRIM(a.last_run_headgear), ''), '') = ''
             AND COALESCE(NULLIF(TRIM(ru.headgear), ''), '') <> COALESCE(NULLIF(TRIM(a.last_run_headgear), ''), '')
            THEN 1
        ELSE 0
    END AS horse_first_time_headgear,
    COALESCE(a.latest_prior_scheduled_off_utc, ra.decision_cutoff_utc - INTERVAL 1 SECOND) AS event_timestamp_utc,
    ra.decision_cutoff_utc
FROM agg a
JOIN runners ru ON ru.runner_id = a.runner_id
JOIN races ra ON ra.race_id = a.race_id;
