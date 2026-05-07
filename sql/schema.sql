PRAGMA enable_progress_bar;

CREATE TABLE IF NOT EXISTS races (
    race_id VARCHAR PRIMARY KEY,
    source VARCHAR NOT NULL,
    meeting_name VARCHAR,
    course_id VARCHAR,
    course_name VARCHAR,
    country_code VARCHAR,
    race_date DATE NOT NULL,
    off_time_utc TIMESTAMP,
    race_type VARCHAR,
    surface VARCHAR,
    distance_m INTEGER,
    class_band VARCHAR,
    going VARCHAR,
    declared_runners INTEGER,
    is_standard_race BOOLEAN DEFAULT TRUE,
    created_at_utc TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runners (
    runner_id VARCHAR PRIMARY KEY,
    race_id VARCHAR NOT NULL,
    horse_id VARCHAR,
    horse_name VARCHAR,
    draw INTEGER,
    age INTEGER,
    weight_lbs DOUBLE,
    jockey_id VARCHAR,
    jockey_name VARCHAR,
    trainer_id VARCHAR,
    trainer_name VARCHAR,
    match_type VARCHAR,
    non_runner BOOLEAN DEFAULT FALSE,
    created_at_utc TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS results (
    race_id VARCHAR NOT NULL,
    runner_id VARCHAR NOT NULL,
    finish_position INTEGER,
    win_flag INTEGER,
    beaten_distance DOUBLE,
    official_time_sec DOUBLE,
    result_timestamp_utc TIMESTAMP,
    PRIMARY KEY (race_id, runner_id)
);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    race_id VARCHAR NOT NULL,
    runner_id VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    market_type VARCHAR DEFAULT 'WIN',
    snapshot_timestamp_utc TIMESTAMP NOT NULL,
    minutes_to_off INTEGER,
    decimal_odds DOUBLE NOT NULL,
    implied_prob_raw DOUBLE,
    traded_volume_gbp DOUBLE,
    market_status VARCHAR,
    ingest_timestamp_utc TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (race_id, runner_id, source, market_type, snapshot_timestamp_utc)
);

CREATE TABLE IF NOT EXISTS horse_history (
    horse_id VARCHAR NOT NULL,
    asof_timestamp_utc TIMESTAMP NOT NULL,
    runs_last_365d INTEGER,
    wins_last_365d INTEGER,
    places_last_365d INTEGER,
    avg_finish_last_5 DOUBLE,
    rest_days INTEGER,
    PRIMARY KEY (horse_id, asof_timestamp_utc)
);

CREATE TABLE IF NOT EXISTS trainer_history (
    trainer_id VARCHAR NOT NULL,
    asof_timestamp_utc TIMESTAMP NOT NULL,
    runs_last_365d INTEGER,
    wins_last_365d INTEGER,
    win_rate_last_365d DOUBLE,
    roi_last_365d DOUBLE,
    PRIMARY KEY (trainer_id, asof_timestamp_utc)
);

CREATE TABLE IF NOT EXISTS jockey_history (
    jockey_id VARCHAR NOT NULL,
    asof_timestamp_utc TIMESTAMP NOT NULL,
    runs_last_365d INTEGER,
    wins_last_365d INTEGER,
    win_rate_last_365d DOUBLE,
    roi_last_365d DOUBLE,
    PRIMARY KEY (jockey_id, asof_timestamp_utc)
);

CREATE TABLE IF NOT EXISTS feature_store (
    race_id VARCHAR NOT NULL,
    runner_id VARCHAR NOT NULL,
    decision_cutoff_utc TIMESTAMP NOT NULL,
    source_market VARCHAR,
    market_implied_prob DOUBLE,
    horse_form_score DOUBLE,
    trainer_form_score DOUBLE,
    jockey_form_score DOUBLE,
    draw_bias_score DOUBLE,
    pace_score DOUBLE,
    feature_version VARCHAR NOT NULL,
    created_at_utc TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (race_id, runner_id, decision_cutoff_utc, feature_version)
);

CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id BIGINT,
    race_id VARCHAR NOT NULL,
    runner_id VARCHAR NOT NULL,
    placed_timestamp_utc TIMESTAMP NOT NULL,
    bookmaker_source VARCHAR NOT NULL,
    price_taken DOUBLE NOT NULL,
    model_prob DOUBLE NOT NULL,
    market_prob_fair DOUBLE NOT NULL,
    edge DOUBLE NOT NULL,
    stake DOUBLE NOT NULL,
    result_win_flag INTEGER,
    pnl DOUBLE,
    PRIMARY KEY (trade_id)
);

CREATE OR REPLACE VIEW odds_snapshots_enriched AS
SELECT
    o.*,
    CASE WHEN o.decimal_odds > 0 THEN 1.0 / o.decimal_odds ELSE NULL END AS implied_prob_from_odds
FROM odds_snapshots o;
