from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quality.checks import run_all_checks
from quality.leakage_guard import check_no_leakage
from ingestion.db_connect import get_db

DB_PATH = ROOT / "racing.duckdb"
SQL_DIR = ROOT / "sql" / "features"
BASELINE_MIN_RACE_DATE = date(2015, 1, 1)

PHASE_FILES: list[tuple[str, str]] = [
    ("001_horse_form.sql", "f001"),
    ("002_draw_bias.sql", "f002"),
    ("003_trainer_stats.sql", "f003"),
    ("004_jockey_stats.sql", "f004"),
    ("005_class_features.sql", "f005"),
    ("006_race_context.sql", "f006"),
    ("007_collateral_form.sql", "f007"),
    ("008_runner_profile.sql", "f008"),
    ("009_speed_and_changes.sql", "f009"),
]


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _column_null_rates(con: duckdb.DuckDBPyConnection, table_name: str) -> list[tuple[str, float]]:
    cols = con.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = ?
        ORDER BY ordinal_position
        """,
        [table_name],
    ).fetchall()

    total = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    if total == 0:
        return [(c[0], 0.0) for c in cols]

    out: list[tuple[str, float]] = []
    for (col_name,) in cols:
        qcol = _quote_ident(col_name)
        null_count = int(con.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {qcol} IS NULL").fetchone()[0])
        out.append((col_name, null_count / total))
    return out


def _prepare_upstream_inputs(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        UPDATE horse_history hh
        SET
            going_code = COALESCE(hh.going_code, ra.going_code),
            distance_furlongs = COALESCE(hh.distance_furlongs, ra.distance_furlongs),
            race_class = COALESCE(hh.race_class, ra.race_class),
            is_handicap = COALESCE(hh.is_handicap, ra.is_handicap),
            field_size = COALESCE(hh.field_size, ra.field_size)
        FROM races ra
        WHERE hh.race_id = ra.race_id
        """
    )

    con.execute(
        """
        UPDATE horse_history hh
        SET
            finishing_position = COALESCE(hh.finishing_position, res.finishing_position),
            won = COALESCE(hh.won, res.won),
            btn_lengths = COALESCE(hh.btn_lengths, res.btn_lengths),
            rpr = COALESCE(hh.rpr, res.rpr)
        FROM results res
        JOIN runners ru ON res.runner_id = ru.runner_id
        WHERE ru.horse_id = hh.horse_id
          AND res.race_id = hh.race_id
        """
    )

    con.execute(
        """
        UPDATE horse_history hh
        SET headgear = COALESCE(hh.headgear, ru.headgear)
        FROM runners ru
        WHERE ru.horse_id = hh.horse_id
          AND ru.race_id = hh.race_id
        """
    )

    con.execute(
        """
        UPDATE horse_history hh
        SET days_since_prev_run = sub.days_since_prev_run
        FROM (
            SELECT
                history_id,
                DATE_DIFF(
                    'day',
                    LAG(scheduled_off_utc) OVER (PARTITION BY horse_id ORDER BY scheduled_off_utc),
                    scheduled_off_utc
                ) AS days_since_prev_run
            FROM horse_history
        ) sub
        WHERE hh.history_id = sub.history_id
        """
    )

    con.execute("DELETE FROM trainer_history")
    con.execute(
        """
        INSERT INTO trainer_history (
            history_id,
            trainer_id,
            trainer_name,
            race_id,
            race_date,
            scheduled_off_utc,
            course_id,
            race_type,
            going_code,
            distance_furlongs,
            race_class,
            days_since_last_run,
            won,
            finishing_position,
            field_size,
            event_timestamp_utc,
            decision_cutoff_utc,
            ingest_timestamp_utc
        )
        SELECT
            r.runner_id || '_tr' AS history_id,
            COALESCE(r.trainer_id, 'unknown') AS trainer_id,
            COALESCE(NULLIF(TRIM(r.trainer_name), ''), 'unknown') AS trainer_name,
            r.race_id,
            ra.race_date,
            ra.scheduled_off_utc,
            ra.course_id,
            ra.race_type,
            ra.going_code,
            ra.distance_furlongs,
            ra.race_class,
            hh.days_since_prev_run AS days_since_last_run,
            res.won,
            res.finishing_position,
            ra.field_size,
            ra.scheduled_off_utc AS event_timestamp_utc,
            ra.decision_cutoff_utc,
            NOW() AS ingest_timestamp_utc
        FROM runners r
        JOIN races ra ON r.race_id = ra.race_id
        JOIN results res ON r.runner_id = res.runner_id
        LEFT JOIN horse_history hh ON hh.race_id = r.race_id AND hh.horse_id = r.horse_id
        WHERE COALESCE(NULLIF(TRIM(r.trainer_name), ''), '') <> ''
        """
    )

    con.execute("DELETE FROM jockey_history")
    con.execute(
        """
        INSERT INTO jockey_history (
            history_id,
            jockey_id,
            jockey_name,
            trainer_id,
            race_id,
            race_date,
            scheduled_off_utc,
            course_id,
            race_type,
            going_code,
            won,
            finishing_position,
            field_size,
            event_timestamp_utc,
            decision_cutoff_utc,
            ingest_timestamp_utc
        )
        SELECT
            r.runner_id || '_jk' AS history_id,
            COALESCE(r.jockey_id, 'unknown') AS jockey_id,
            COALESCE(NULLIF(TRIM(r.jockey_name), ''), 'unknown') AS jockey_name,
            COALESCE(r.trainer_id, 'unknown') AS trainer_id,
            r.race_id,
            ra.race_date,
            ra.scheduled_off_utc,
            ra.course_id,
            ra.race_type,
            ra.going_code,
            res.won,
            res.finishing_position,
            ra.field_size,
            ra.scheduled_off_utc AS event_timestamp_utc,
            ra.decision_cutoff_utc,
            NOW() AS ingest_timestamp_utc
        FROM runners r
        JOIN races ra ON r.race_id = ra.race_id
        JOIN results res ON r.runner_id = res.runner_id
        WHERE COALESCE(NULLIF(TRIM(r.jockey_name), ''), '') <> ''
        """
    )


def _materialize_feature_store(con: duckdb.DuckDBPyConnection) -> int:
    con.execute(
        """
        CREATE OR REPLACE TABLE feature_store AS
        WITH base AS (
            SELECT
                r.runner_id,
                r.race_id,
                ra.race_date,
                ra.decision_cutoff_utc,
                res.won AS target,
                f001.horse_runs_last_3_positions,
                f001.horse_runs_last_5_positions,
                f001.horse_wins_last_5,
                f001.horse_win_rate_last_10,
                f001.horse_days_since_last_run,
                f001.horse_runs_last_90_days,
                f001.horse_going_group_affinity,
                f001.horse_going_group_place_rate,
                f001.horse_distance_affinity,
                f001.horse_distance_place_rate,
                f001.horse_course_affinity,
                f001.horse_course_place_rate,
                f001.horse_course_runs,
                f001.horse_weighted_form,
                f001.horse_place_rate_last_5,
                f001.horse_place_rate_last_10,
                f001.horse_improvement_index,
                f001.horse_avg_position_pct_last_5,
                f001.horse_best_rpr_last_5,
                f001.horse_best_rpr_rp_last_5,
                f001.horse_avg_rpr_last_3,
                f001.horse_last_rpr,
                f001.horse_avg_class_last_3,
                f005.horse_class_delta AS horse_class_delta,
                f001.horse_form_trend,
                f001.horse_first_time_headgear,
                f002.draw_position,
                f002.draw_field_percentile,
                f002.draw_course_going_win_rate,
                f002.draw_bias_coefficient,
                f002.draw_is_null,
                f003.trainer_win_rate_90d,
                f003.trainer_win_rate_course_90d,
                f003.trainer_course_going_win_rate,
                f003.trainer_dist_alltime_win_rate,
                f003.trainer_win_rate_going_90d,
                f003.trainer_win_rate_dist_band_90d,
                f003.trainer_runs_90d,
                f003.trainer_fresh_win_rate,
                f003.trainer_fresh_runs,
                f004.jockey_win_rate_90d,
                f004.jockey_win_rate_course_90d,
                f004.jockey_dist_win_rate_90d,
                f004.jockey_trainer_combo_win_rate,
                f004.jockey_trainer_combo_runs,
                f004.jockey_runs_90d,
                f005.race_class_encoded,
                COALESCE(f005.is_handicap, FALSE) AS is_handicap,
                f005.race_grade,
                f005.prize_money_log,
                f005.is_class_dropper,
                f006.field_size,
                f006.pace_front_runners,
                f006.pace_hold_up_horses,
                f006.pace_pressure_index,
                f006.surface_encoded,
                f006.going_encoded,
                f006.race_type_encoded,
                f006.race_month,
                f006.race_day_of_week,
                f007.collateral_beaten_win_rate,
                f007.collateral_beaten_place_rate,
                f007.collateral_franked_winners,
                f007.collateral_beaten_count,
                f008.weight_lbs,
                f008.weight_vs_top,
                f008.weight_vs_field_avg,
                f008.horse_age,
                f008.runner_official_rating,
                f008.rating_vs_top,
                f008.rating_vs_field_avg,
                f008.field_avg_rating,
                f008.career_runs,
                f008.career_win_rate,
                f008.career_place_rate,
                f008.position_consistency,
                f009.avg_speed_last_3,
                f009.best_speed_last_5,
                f009.last_run_speed,
                f009.is_jumps,
                f009.trip_change_furlongs,
                f009.weight_change_lbs,
                f009.last_run_btn_lengths,
                f009.avg_btn_last_3,
                f009.jockey_upgrade_signal,
                f009.trainer_win_rate_14d,
                f009.trainer_runs_14d,
                ra.surface AS race_surface,
                ra.race_type AS race_type_raw,
                COALESCE(f006.distance_furlongs, ra.distance_furlongs) AS distance_raw
            FROM runners r
            JOIN races ra ON r.race_id = ra.race_id
            JOIN results res ON r.runner_id = res.runner_id
            LEFT JOIN f001 ON r.runner_id = f001.runner_id
            LEFT JOIN f002 ON r.runner_id = f002.runner_id
            LEFT JOIN f003 ON r.runner_id = f003.runner_id
            LEFT JOIN f004 ON r.runner_id = f004.runner_id
            LEFT JOIN f005 ON r.runner_id = f005.runner_id
            LEFT JOIN f006 ON r.runner_id = f006.runner_id
            LEFT JOIN f007 ON r.runner_id = f007.runner_id
            LEFT JOIN f008 ON r.runner_id = f008.runner_id
            LEFT JOIN f009 ON r.runner_id = f009.runner_id
            WHERE ra.is_standard_race = TRUE
        ),
        parsed AS (
            SELECT
                b.*,
                CASE
                    WHEN b.distance_raw IS NOT NULL THEN b.distance_raw
                    WHEN NULLIF(regexp_extract(LOWER(COALESCE(b.race_type_raw, '')), '([0-9]+)m', 1), '') IS NOT NULL THEN
                        CAST(NULLIF(regexp_extract(LOWER(COALESCE(b.race_type_raw, '')), '([0-9]+)m', 1), '') AS DOUBLE) * 8.0
                        + COALESCE(CAST(NULLIF(regexp_extract(LOWER(COALESCE(b.race_type_raw, '')), '([0-9]+)f', 1), '') AS DOUBLE), 0.0)
                    WHEN NULLIF(regexp_extract(LOWER(COALESCE(b.race_type_raw, '')), '([0-9]+)f', 1), '') IS NOT NULL THEN
                        CAST(NULLIF(regexp_extract(LOWER(COALESCE(b.race_type_raw, '')), '([0-9]+)f', 1), '') AS DOUBLE)
                    ELSE NULL
                END AS distance_after_parse
            FROM base b
        ),
        medians AS (
            SELECT
                race_surface,
                race_type_raw,
                MEDIAN(distance_after_parse) AS median_distance
            FROM parsed
            WHERE distance_after_parse IS NOT NULL
            GROUP BY 1, 2
        ),
        global_median AS (
            SELECT MEDIAN(distance_after_parse) AS median_distance
            FROM parsed
            WHERE distance_after_parse IS NOT NULL
        )
        SELECT
            p.runner_id,
            p.race_id,
            p.race_date,
            p.decision_cutoff_utc,
            p.target,
            p.horse_runs_last_3_positions,
            p.horse_runs_last_5_positions,
            p.horse_wins_last_5,
            p.horse_win_rate_last_10,
            p.horse_days_since_last_run,
            p.horse_runs_last_90_days,
            COALESCE(p.horse_going_group_affinity) AS horse_going_group_affinity,
            COALESCE(p.horse_going_group_place_rate) AS horse_going_group_place_rate,
            p.horse_distance_affinity,
            p.horse_distance_place_rate,
            p.horse_course_affinity,
            p.horse_course_place_rate,
            p.horse_course_runs,
            p.horse_weighted_form,
            p.horse_place_rate_last_5,
            p.horse_place_rate_last_10,
            p.horse_improvement_index,
            p.horse_avg_position_pct_last_5,
            p.horse_best_rpr_last_5,
            p.horse_best_rpr_rp_last_5,
            p.horse_avg_rpr_last_3,
            p.horse_last_rpr,
            p.horse_avg_class_last_3,
            p.horse_class_delta,
            p.horse_form_trend,
            p.horse_first_time_headgear,
            p.draw_position,
            p.draw_field_percentile,
            p.draw_course_going_win_rate,
            p.draw_bias_coefficient,
            p.draw_is_null,
            p.trainer_win_rate_90d,
            p.trainer_win_rate_course_90d,
            p.trainer_course_going_win_rate,
            p.trainer_dist_alltime_win_rate,
            p.trainer_win_rate_going_90d,
            p.trainer_win_rate_dist_band_90d,
            p.trainer_runs_90d,
            p.trainer_fresh_win_rate,
            p.trainer_fresh_runs,
            p.jockey_win_rate_90d,
            p.jockey_win_rate_course_90d,
            p.jockey_dist_win_rate_90d,
            p.jockey_trainer_combo_win_rate,
            p.jockey_trainer_combo_runs,
            p.jockey_runs_90d,
            p.race_class_encoded,
            p.is_handicap,
            p.race_grade,
            p.prize_money_log,
            p.is_class_dropper,
            p.field_size,
            p.pace_front_runners,
            p.pace_hold_up_horses,
            p.pace_pressure_index,
            p.surface_encoded,
            LEAST(
                COALESCE(p.distance_after_parse, m.median_distance, g.median_distance),
                36.0
            ) AS distance_furlongs,
            p.going_encoded,
            p.race_type_encoded,
            p.race_month,
            p.race_day_of_week,
            p.collateral_beaten_win_rate,
            p.collateral_beaten_place_rate,
            p.collateral_franked_winners,
            p.collateral_beaten_count,
            p.weight_lbs,
            p.weight_vs_top,
            p.weight_vs_field_avg,
            p.horse_age,
            p.runner_official_rating,
            p.rating_vs_top,
            p.rating_vs_field_avg,
            p.field_avg_rating,
            p.career_runs,
            p.career_win_rate,
            p.career_place_rate,
            p.position_consistency,
            p.avg_speed_last_3,
            p.best_speed_last_5,
            p.last_run_speed,
            p.is_jumps,
            p.trip_change_furlongs,
            p.weight_change_lbs,
            p.last_run_btn_lengths,
            p.avg_btn_last_3,
            p.jockey_upgrade_signal,
            p.trainer_win_rate_14d,
            p.trainer_runs_14d
        FROM parsed p
        LEFT JOIN medians m
            ON COALESCE(m.race_surface, '') = COALESCE(p.race_surface, '')
           AND COALESCE(m.race_type_raw, '') = COALESCE(p.race_type_raw, '')
        CROSS JOIN global_median g
        """
    )
    return int(con.execute("SELECT COUNT(*) FROM feature_store").fetchone()[0])


def main() -> None:
    con = get_db(DB_PATH)
    try:
        _prepare_upstream_inputs(con)
        for sql_name, table_name in PHASE_FILES:
            sql_path = SQL_DIR / sql_name
            con.execute(sql_path.read_text(encoding="utf-8"))
            row_count = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
            null_rates = _column_null_rates(con, table_name)
            print(f"PHASE2_FILE={sql_name}")
            print(f"PHASE2_TABLE={table_name}")
            print(f"PHASE2_ROWS={row_count}")
            print("PHASE2_NULL_RATES_START")
            for col_name, rate in null_rates:
                print(f"{col_name}|{rate:.6f}")
            print("PHASE2_NULL_RATES_END")

            leak_result = check_no_leakage(table_name=table_name, db_path=DB_PATH)
            print(f"PHASE2_LEAKAGE={leak_result['leaking_count']}")
            run_all_checks(db_path=DB_PATH, min_race_date=BASELINE_MIN_RACE_DATE)
            print("PHASE2_DQ=PASS")

        feature_rows = _materialize_feature_store(con)
        print(f"FEATURE_STORE_ROWS={feature_rows}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
