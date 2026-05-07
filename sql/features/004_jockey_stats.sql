CREATE OR REPLACE TABLE f004 AS
WITH base AS (
    SELECT
        ru.runner_id,
        ru.race_id,
        COALESCE(NULLIF(TRIM(ru.jockey_name), ''), 'Unknown') AS jockey_name_norm,
        COALESCE(NULLIF(TRIM(ru.trainer_name), ''), 'Unknown') AS trainer_name_norm,
        lower(normalise_course(ra.course_name)) AS course_key,
        ra.decision_cutoff_utc,
        CASE WHEN COALESCE(NULLIF(TRIM(ru.jockey_name), ''), 'Unknown') = 'Unknown' THEN 1 ELSE 0 END AS jockey_is_unknown
    FROM runners ru
    JOIN races ra ON ra.race_id = ru.race_id
),
hist AS (
    SELECT
        COALESCE(NULLIF(TRIM(jh.jockey_name), ''), 'Unknown') AS jockey_name_norm,
        COALESCE(NULLIF(TRIM(th.trainer_name), ''), 'Unknown') AS trainer_name_norm,
        lower(normalise_course(ra.course_name)) AS course_key,
        jh.scheduled_off_utc,
        CASE WHEN jh.won THEN 1.0 ELSE 0.0 END AS won_num
    FROM jockey_history jh
    LEFT JOIN trainer_history th ON th.race_id = jh.race_id AND th.trainer_id = jh.trainer_id
    JOIN races ra ON ra.race_id = jh.race_id
),
agg AS (
    SELECT
        b.runner_id,
        AVG(h.won_num) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
        ) AS jockey_win_rate_90d,
        AVG(h.won_num) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
              AND h.course_key = b.course_key
        ) AS jockey_win_rate_course_90d,
        AVG(h.won_num) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
              AND h.trainer_name_norm = b.trainer_name_norm
        ) AS jockey_trainer_combo_win_rate,
        COUNT(*) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
              AND h.trainer_name_norm = b.trainer_name_norm
        ) AS jockey_trainer_combo_runs,
        COUNT(*) FILTER (
            WHERE h.scheduled_off_utc >= b.decision_cutoff_utc - INTERVAL 90 DAY
        ) AS jockey_runs_90d,
        MAX(h.scheduled_off_utc) AS latest_hist_ts
    FROM base b
    LEFT JOIN hist h
        ON h.jockey_name_norm = b.jockey_name_norm
       AND h.scheduled_off_utc < b.decision_cutoff_utc
    GROUP BY 1
)
SELECT
    b.runner_id,
    b.race_id,
    a.jockey_win_rate_90d,
    a.jockey_win_rate_course_90d,
    a.jockey_trainer_combo_win_rate,
    COALESCE(a.jockey_trainer_combo_runs, 0) AS jockey_trainer_combo_runs,
    COALESCE(a.jockey_runs_90d, 0) AS jockey_runs_90d,
    b.jockey_is_unknown,
    COALESCE(a.latest_hist_ts, b.decision_cutoff_utc - INTERVAL 1 SECOND) AS event_timestamp_utc,
    b.decision_cutoff_utc
FROM base b
LEFT JOIN agg a ON a.runner_id = b.runner_id;
