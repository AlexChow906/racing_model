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
    "is_handicap",
    "race_grade",
    "prize_money_log",
    "pace_hold_up_horses",
    "pace_pressure_index",
    "distance_furlongs",
    "field_avg_rating",
    "horse_nc_last_5",
    "horse_pu_rate",
    "is_female",
    "sex_encoded",
]

FLAT_V2_FEATURES = [
    # Core form
    "horse_weighted_form",
    "horse_form_trend",
    "horse_place_rate_last_5",
    "horse_avg_position_pct_last_5",
    "position_consistency",
    # Speed figures
    "avg_speed_last_3",
    "best_speed_last_5",
    "last_run_speed",
    # RPR (Racing Post Rating)
    "horse_best_rpr_rp_last_5",
    "horse_avg_rpr_last_3",
    "horse_last_rpr",
    # Beaten lengths
    "last_run_btn_lengths",
    "avg_btn_last_3",
    # Class
    "horse_class_delta",
    "is_class_dropper",
    "prize_money_log",
    # Horse profile
    "horse_days_since_last_run",
    "horse_age",
    "career_runs",
    "career_win_rate",
    # Ratings and weight
    "rating_vs_top",
    "rating_vs_field_avg",
    "weight_vs_field_avg",
    "weight_vs_top",
    "weight_change_lbs",
    # Race context
    "is_handicap",
    "field_size",
    "distance_furlongs",
    "pace_pressure_index",
    "race_class_encoded",
    "pace_front_runners",
    "pace_hold_up_horses",
    # Draw
    "draw_position",
    "draw_field_percentile",
    "draw_bias_coefficient",
    # Jockey
    "jockey_win_rate_course_90d",
    "jockey_upgrade_signal",
    "jockey_win_rate_90d",
    "jockey_dist_win_rate_90d",
    "jockey_trainer_combo_win_rate",
    # Trainer
    "trainer_dist_alltime_win_rate",
    "trainer_win_rate_14d",
    "trainer_win_rate_90d",
    "trainer_win_rate_course_90d",
    "trainer_course_going_win_rate",
    "trainer_win_rate_dist_band_90d",
    "trainer_fresh_win_rate",
    "trainer_win_rate_going_90d",
    "trainer_runs_90d",
    # Collateral form
    "collateral_beaten_place_rate",
    "collateral_beaten_win_rate",
    # Affinity and additional form
    "horse_distance_affinity",
    "trip_change_furlongs",
    "horse_wins_last_5",
    "horse_win_rate_last_10",
    "horse_place_rate_last_10",
    "horse_distance_place_rate",
    "horse_runs_last_90_days",
    "career_place_rate",
    "going_encoded",
    "field_avg_rating",
    "horse_first_time_headgear",
    "collateral_franked_winners",
    "collateral_beaten_count",
    # Horse sex
    "sex_encoded",
    "is_female",
]
