EXCLUDE = [
    "runner_id",
    "race_id",
    "race_date",
    "decision_cutoff_utc",
    "target",
    "event_timestamp_utc",
]

DROP_LOW_IMPORTANCE = [
    "collateral_franked_winners",
    "trainer_runs_90d",
    "race_class_encoded",
    "career_win_rate",
    "horse_course_affinity",
    "jockey_trainer_combo_runs",
    "horse_distance_affinity",
    "horse_win_rate_last_10",
    "horse_going_group_affinity",
    "race_type_encoded",
    "horse_wins_last_5",
    "draw_is_null",
    "horse_first_time_headgear",
    "surface_encoded",
    "race_day_of_week",
    "going_encoded",
    "horse_course_runs",
    "race_month",
    "field_size",
    "horse_place_rate_last_5",
    "horse_fall_rate",
    "horse_completion_rate",
    "pace_front_runners",
]

FLAT_DROP = DROP_LOW_IMPORTANCE + [
    "is_jumps",
]

JUMPS_DROP = DROP_LOW_IMPORTANCE + [
    "draw_position",
    "draw_field_percentile",
    "draw_course_going_win_rate",
    "draw_bias_coefficient",
    "is_jumps",
    "surface_encoded",
]
