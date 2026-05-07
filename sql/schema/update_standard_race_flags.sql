-- Pattern-based non-standard market/race filter.
-- This is executed by quality.checks.ensure_standard_race_flag on every run.
UPDATE races
SET is_standard_race = FALSE
WHERE COALESCE(is_standard_race, TRUE)
  AND (
    race_type ILIKE '%forecast%'
    OR race_type ILIKE '%reverse%'
    OR race_type ILIKE '%arab%'
    OR race_type ILIKE '%how far%'
    OR race_type ILIKE '% v %'
    OR race_type ILIKE '%charity%'
    OR race_type ILIKE 'pa%'
    OR race_type ILIKE '% pa %'
    OR race_type ILIKE '% pa'
    OR race_type ~ '^\d+[mf]\d*[f]?\s+pa'
    OR race_type ILIKE '%each way%'
    OR race_type ILIKE '%each-way%'
    OR race_type ILIKE '%win dist%'
    OR race_type ILIKE '%daily odds%'
    OR race_type ILIKE '%specials%'
    OR race_id = 'bfsp_162366764_win'
    OR race_id = 'bfsp_201092786_win'
    OR race_id = 'bfsp_218291493_win'
    OR race_id = 'bfsp_218647872_win'
    OR race_id = 'bfsp_230193686_win'
    OR race_id = 'bfsp_245106872_win'
    OR race_id = 'bfsp_246247068_win'
    OR course_name ILIKE '%win dist%'
    OR course_name ILIKE '%daily odds%'
    OR course_name ILIKE '%specials%'
    OR field_size < 2
    OR field_size > 40
  );
