CREATE OR REPLACE TABLE f002 AS
WITH base AS (
    SELECT
        ru.runner_id,
        ru.race_id,
        ru.draw,
        ra.field_size,
        ra.course_id,
        ra.going_code,
        ra.race_type,
        ra.surface,
        ra.decision_cutoff_utc,
        CASE
            WHEN ru.draw IS NULL OR COALESCE(ra.field_size, 0) <= 0 THEN NULL
            ELSE LEAST(1.0, GREATEST(0.0, ru.draw::DOUBLE / ra.field_size::DOUBLE))
        END AS draw_field_percentile,
        CASE
            WHEN ru.draw IS NULL THEN NULL
            ELSE NTILE(4) OVER (PARTITION BY ru.race_id ORDER BY ru.draw)
        END AS draw_quartile
    FROM runners ru
    JOIN races ra ON ra.race_id = ru.race_id
),
hist_draw AS (
    SELECT
        ra.race_id,
        ra.course_id,
        ra.going_code,
        ra.scheduled_off_utc,
        ru.draw,
        NTILE(4) OVER (PARTITION BY ra.race_id ORDER BY ru.draw) AS draw_quartile,
        CASE WHEN res.won THEN 1.0 ELSE 0.0 END AS won_num
    FROM races ra
    JOIN runners ru ON ru.race_id = ra.race_id
    JOIN results res ON res.runner_id = ru.runner_id
    WHERE ru.draw IS NOT NULL
      AND (
          LOWER(COALESCE(ra.race_type, '')) LIKE '%flat%'
          OR LOWER(COALESCE(ra.surface, '')) IN ('aw', 'allweather', 'all_weather')
      )
),
agg AS (
    SELECT
        b.runner_id,
        AVG(hd.won_num) FILTER (
            WHERE hd.course_id = b.course_id
              AND COALESCE(hd.going_code, '') = COALESCE(b.going_code, '')
        ) AS draw_course_going_win_rate_exact,
        AVG(hd.won_num) FILTER (
            WHERE hd.course_id = b.course_id
        ) AS draw_course_win_rate,
        AVG(hd.won_num) FILTER (
            WHERE COALESCE(hd.going_code, '') = COALESCE(b.going_code, '')
        ) AS draw_going_win_rate,
        AVG(hd.won_num) AS draw_global_quartile_win_rate,
        MAX(hd.scheduled_off_utc) AS latest_hist_ts
    FROM base b
    LEFT JOIN hist_draw hd
        ON hd.draw_quartile = b.draw_quartile
       AND hd.scheduled_off_utc < b.decision_cutoff_utc
    GROUP BY 1
)
SELECT
    b.runner_id,
    b.race_id,
    b.draw AS draw_position,
    CASE
        WHEN LOWER(COALESCE(b.race_type, '')) LIKE '%jumps%'
          OR LOWER(COALESCE(b.surface, '')) = 'jumps'
            THEN NULL
        ELSE b.draw_field_percentile
    END AS draw_field_percentile,
    CASE
        WHEN LOWER(COALESCE(b.race_type, '')) LIKE '%jumps%'
          OR LOWER(COALESCE(b.surface, '')) = 'jumps'
            THEN NULL
        ELSE COALESCE(
            a.draw_course_going_win_rate_exact,
            a.draw_course_win_rate,
            a.draw_going_win_rate,
            a.draw_global_quartile_win_rate
        )
    END AS draw_course_going_win_rate,
    CASE
        WHEN LOWER(COALESCE(b.race_type, '')) LIKE '%jumps%'
          OR LOWER(COALESCE(b.surface, '')) = 'jumps'
          OR COALESCE(
              a.draw_course_going_win_rate_exact,
              a.draw_course_win_rate,
              a.draw_going_win_rate,
              a.draw_global_quartile_win_rate
          ) IS NULL
          OR COALESCE(b.field_size, 0) <= 0
            THEN NULL
        ELSE COALESCE(
            a.draw_course_going_win_rate_exact,
            a.draw_course_win_rate,
            a.draw_going_win_rate,
            a.draw_global_quartile_win_rate
        ) / (1.0 / b.field_size::DOUBLE)
    END AS draw_bias_coefficient,
    CASE WHEN b.draw IS NULL THEN 1 ELSE 0 END AS draw_is_null,
    COALESCE(a.latest_hist_ts, b.decision_cutoff_utc - INTERVAL 1 SECOND) AS event_timestamp_utc,
    b.decision_cutoff_utc
FROM base b
LEFT JOIN agg a ON a.runner_id = b.runner_id;
