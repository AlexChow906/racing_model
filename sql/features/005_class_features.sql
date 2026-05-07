CREATE OR REPLACE TABLE f005 AS
SELECT
    ru.runner_id,
    ru.race_id,
    CASE
        WHEN ra.race_class BETWEEN 1 AND 7 THEN ra.race_class
        WHEN LOWER(COALESCE(ra.race_grade, '')) = 'group 1' THEN 1
        WHEN LOWER(COALESCE(ra.race_grade, '')) = 'group 2' THEN 2
        WHEN LOWER(COALESCE(ra.race_grade, '')) = 'group 3' THEN 3
        WHEN LOWER(COALESCE(ra.race_grade, '')) = 'listed' THEN 4
        ELSE NULL
    END AS race_class_encoded,
    COALESCE(ra.is_handicap, FALSE) AS is_handicap,
    CASE
        WHEN LOWER(COALESCE(ra.race_grade, '')) = 'group 1' THEN 'Group1'
        WHEN LOWER(COALESCE(ra.race_grade, '')) = 'group 2' THEN 'Group2'
        WHEN LOWER(COALESCE(ra.race_grade, '')) = 'group 3' THEN 'Group3'
        WHEN LOWER(COALESCE(ra.race_grade, '')) = 'listed' THEN 'Listed'
        WHEN ra.is_handicap THEN 'Handicap'
        WHEN LOWER(COALESCE(ra.race_type, '')) LIKE '%maiden%' THEN 'Maiden'
        WHEN LOWER(COALESCE(ra.race_type, '')) LIKE '%novice%' THEN 'Novice'
        WHEN LOWER(COALESCE(ra.race_type, '')) LIKE '%conditions%' THEN 'Conditions'
        WHEN LOWER(COALESCE(ra.race_type, '')) LIKE '%selling%' THEN 'Selling'
        ELSE 'Other'
    END AS race_grade,
    LN(COALESCE(ra.prize_money_gbp, 0.0) + 1.0) AS prize_money_log,
    CASE
        WHEN f1.horse_avg_class_last_3 IS NULL THEN NULL
        WHEN ra.race_class BETWEEN 1 AND 7 THEN ra.race_class - f1.horse_avg_class_last_3
        ELSE NULL
    END AS horse_class_delta,
    CASE
        WHEN f1.horse_avg_class_last_3 IS NULL THEN 0
        WHEN (ra.race_class BETWEEN 1 AND 7) AND (ra.race_class - f1.horse_avg_class_last_3) >= 2 THEN 1
        ELSE 0
    END AS is_class_dropper,
    COALESCE(f1.horse_first_time_headgear, 0) AS is_first_time_headgear,
    ra.decision_cutoff_utc - INTERVAL 1 SECOND AS event_timestamp_utc,
    ra.decision_cutoff_utc
FROM runners ru
JOIN races ra ON ra.race_id = ru.race_id
LEFT JOIN f001 f1 ON f1.runner_id = ru.runner_id;
