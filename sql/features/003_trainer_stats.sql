CREATE OR REPLACE TABLE f003 AS
WITH base AS (
    SELECT
        ru.runner_id,
        ru.race_id,
        COALESCE(NULLIF(TRIM(ru.trainer_name), ''), 'Unknown') AS trainer_name_norm,
        lower(normalise_course(ra.course_name)) AS course_key,
        ra.going_code,
        ra.distance_furlongs,
        ra.decision_cutoff_utc,
        CASE WHEN COALESCE(NULLIF(TRIM(ru.trainer_name), ''), 'Unknown') = 'Unknown' THEN 1 ELSE 0 END AS trainer_is_unknown
    FROM runners ru
    JOIN races ra ON ra.race_id = ru.race_id
),
hist AS (
    SELECT
        COALESCE(NULLIF(TRIM(th.trainer_name), ''), 'Unknown') AS trainer_name_norm,
        lower(normalise_course(ra.course_name)) AS course_key,
        th.going_code,
        th.distance_furlongs,
        th.days_since_last_run,
        th.scheduled_off_utc,
        CASE WHEN th.won THEN 1.0 ELSE 0.0 END AS won_num
    FROM trainer_history th
    JOIN races ra ON ra.race_id = th.race_id
),
agg AS (
    SELECT
        b.runner_id,
        AVG(h.won_num) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
        ) AS trainer_win_rate_90d,
        AVG(h.won_num) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
              AND h.course_key = b.course_key
        ) AS trainer_win_rate_course_90d,
        AVG(h.won_num) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
              AND COALESCE(h.going_code, '') = COALESCE(b.going_code, '')
        ) AS trainer_win_rate_going_90d,
        AVG(h.won_num) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
              AND ABS(COALESCE(h.distance_furlongs, 0.0) - COALESCE(b.distance_furlongs, 0.0)) <= 1.0
        ) AS trainer_win_rate_dist_band_90d,
        COUNT(*) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
        ) AS trainer_runs_90d,
        AVG(h.won_num) FILTER (
            WHERE h.days_since_last_run > 60
        ) AS trainer_fresh_win_rate,
        COUNT(*) FILTER (
            WHERE h.days_since_last_run > 60
        ) AS trainer_fresh_runs,
        MAX(h.scheduled_off_utc) AS latest_hist_ts
    FROM base b
    LEFT JOIN hist h
        ON h.trainer_name_norm = b.trainer_name_norm
       AND h.scheduled_off_utc < b.decision_cutoff_utc
    GROUP BY 1
)
SELECT
    b.runner_id,
    b.race_id,
    a.trainer_win_rate_90d,
    a.trainer_win_rate_course_90d,
    a.trainer_win_rate_going_90d,
    a.trainer_win_rate_dist_band_90d,
    COALESCE(a.trainer_runs_90d, 0) AS trainer_runs_90d,
    a.trainer_fresh_win_rate,
    COALESCE(a.trainer_fresh_runs, 0) AS trainer_fresh_runs,
    b.trainer_is_unknown,
    COALESCE(a.latest_hist_ts, b.decision_cutoff_utc - INTERVAL 1 SECOND) AS event_timestamp_utc,
    b.decision_cutoff_utc
FROM base b
LEFT JOIN agg a ON a.runner_id = b.runner_id;
