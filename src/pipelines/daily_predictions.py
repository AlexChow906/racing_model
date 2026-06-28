"""
Daily predictions — score today's or tomorrow's races using Betfair API for race cards.
Flat uses CatBoost v2 (66 features, no calibration). Jumps uses LightGBM + isotonic.

Usage:
    python -m src.pipelines.daily_predictions --date tomorrow
    python -m src.pipelines.daily_predictions --date tomorrow --flat
    python -m src.pipelines.daily_predictions --date tomorrow --jumps
    python -m src.pipelines.daily_predictions --date tomorrow --min-edge 0.12
    python -m src.pipelines.daily_predictions --date 2026-05-15 --output bets.csv
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(ROOT / ".env")

from ingestion.db_connect import get_db
from ingestion.normalise import slugify, decision_cutoff_for_off_time
from constants.features import EXCLUDE, JUMPS_DROP, FLAT_V2_FEATURES
from pipelines.helpers import resolve_date, append_dated_csv, BETS_LOG_COLS

MODELS_DIR = ROOT / "models"

IRE_KEYWORDS = [
    "cork", "navan", "leopardstown", "curragh", "galway", "fairyhouse", "punchestown",
    "dundalk", "limerick", "tipperary", "killarney", "listowel", "gowran", "wexford",
    "sligo", "clonmel", "thurles", "downpatrick", "down royal", "kilbeggan", "ballinrobe",
    "roscommon", "naas", "tramore", "laytown", "bellewstown",
]

JUMPS_COURSES = [
    "hereford", "aintree", "cheltenham", "exeter", "fontwell", "huntingdon", "kempton",
    "ludlow", "newbury", "newton abbot", "plumpton", "sandown", "sedgefield", "stratford",
    "taunton", "towcester", "uttoxeter", "warwick", "wetherby", "wincanton", "worcester",
    "hexham", "cartmel", "bangor", "fakenham", "market rasen", "musselburgh", "perth",
    "kelso", "ayr", "carlisle", "catterick", "doncaster", "haydock", "leicester",
    "newcastle", "southwell", "ffos las", "chepstow",
]


def get_betfair_client():
    """Create and login Betfair API client."""
    import betfairlightweight
    cert_file = Path(os.environ["BETFAIR_CERT_FILE"])
    client = betfairlightweight.APIClient(
        username=os.environ["BETFAIR_USERNAME"],
        password=os.environ["BETFAIR_PASSWORD"],
        app_key=os.environ["BETFAIR_APP_KEY"],
        certs=str(cert_file.parent),
    )
    client.login()
    return client


def to_uk_time(dt):
    """Convert datetime to UK local time string (HH:MM)."""
    try:
        import zoneinfo
        uk = zoneinfo.ZoneInfo("Europe/London")
    except ImportError:
        from dateutil import tz
        uk = tz.gettz("Europe/London")
    try:
        if hasattr(dt, "tzinfo") and dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        return dt.astimezone(uk).strftime("%H:%M")
    except Exception:
        return str(dt)[:5]


def fetch_race_cards(target_date: date):
    """Fetch race cards from Betfair API with full runner metadata."""
    import betfairlightweight

    client = get_betfair_client()

    day_start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, tzinfo=timezone.utc)
    day_end = datetime(target_date.year, target_date.month, target_date.day, 23, 59, tzinfo=timezone.utc)

    markets = client.betting.list_market_catalogue(
        filter=betfairlightweight.filters.market_filter(
            event_type_ids=["7"],
            market_countries=["GB", "IE"],
            market_type_codes=["WIN"],
            market_start_time={"from": day_start.isoformat(), "to": day_end.isoformat()},
        ),
        market_projection=["RUNNER_DESCRIPTION", "RUNNER_METADATA", "EVENT", "MARKET_START_TIME"],
        max_results=200,
    )

    races = []
    runners = []

    for m in sorted(markets, key=lambda x: x.market_start_time):
        event_name = m.event.name if m.event else "Unknown"
        market_name = m.market_name if hasattr(m, "market_name") and m.market_name else ""
        scheduled_off = m.market_start_time
        decision_cutoff = decision_cutoff_for_off_time(scheduled_off)
        race_id = f"bfsp_{m.market_id}_win"
        course_id = slugify(event_name)
        course_lower = event_name.lower()
        market_lower = market_name.lower()
        country = "IE" if any(kw in course_lower for kw in IRE_KEYWORDS) else "GB"

        if any(kw in market_lower for kw in ("chase", "steeple", "steeplechase", "chs")):
            race_type = "Chase"
        elif any(kw in market_lower for kw in ("hurdle", "hdle", "hrd")):
            race_type = "Hurdle"
        elif any(kw in market_lower for kw in ("nh flat", "nhf", "inhf", "bumper", "national hunt flat")):
            race_type = "NH Flat"
        else:
            race_type = "Flat"


        races.append({
            "race_id": race_id,
            "market_id": m.market_id,
            "course_name": event_name,
            "course_id": course_id,
            "race_date": target_date,
            "scheduled_off_utc": scheduled_off,
            "decision_cutoff_utc": decision_cutoff,
            "field_size": len(m.runners),
            "country": country,
            "race_type": race_type,
        })

        for runner in m.runners:
            meta = runner.metadata or {}
            horse_name = runner.runner_name
            horse_id = slugify(horse_name)
            runner_id = f"{race_id}_{horse_id}"

            trainer = meta.get("TRAINER_NAME")
            jockey = meta.get("JOCKEY_NAME")
            draw = meta.get("STALL_DRAW")
            weight = meta.get("WEIGHT_VALUE")
            age = meta.get("AGE")
            official_rating = meta.get("OFFICIAL_RATING")
            headgear = meta.get("WEARING", "")
            form = meta.get("FORM")

            cloth_number = meta.get("CLOTH_NUMBER") or runner.sort_priority
            runners.append({
                "runner_id": runner_id,
                "race_id": race_id,
                "market_id": m.market_id,
                "selection_id": runner.selection_id,
                "cloth_number": int(cloth_number) if cloth_number and str(cloth_number).isdigit() else None,
                "horse_id": horse_id,
                "horse_name": horse_name,
                "trainer_name": trainer,
                "trainer_id": slugify(trainer) if trainer else None,
                "jockey_name": jockey,
                "jockey_id": slugify(jockey) if jockey else None,
                "draw": int(draw) if draw and str(draw).isdigit() else None,
                "weight_lbs": float(weight) if weight else None,
                "age": int(age) if age and str(age).isdigit() else None,
                "official_rating": int(official_rating) if official_rating and str(official_rating).isdigit() and int(official_rating) > 0 else None,
                "headgear": headgear if headgear else None,
                "form": form,
            })

    return races, runners


def fetch_live_odds(market_ids: list[str]) -> dict:
    """Fetch live exchange back/lay prices for all markets. Returns {selection_id: {back, lay, back_size}}."""
    import betfairlightweight

    client = get_betfair_client()
    odds = {}

    # Betfair allows max 40 markets per request
    for i in range(0, len(market_ids), 40):
        batch = market_ids[i:i+40]
        books = client.betting.list_market_book(
            market_ids=batch,
            price_projection=betfairlightweight.filters.price_projection(
                price_data=["EX_BEST_OFFERS"]
            ),
        )
        for book in books:
            for runner in book.runners:
                status = runner.status if hasattr(runner, "status") else None
                if status and status != "ACTIVE":
                    continue
                back = runner.ex.available_to_back[0].price if runner.ex.available_to_back else None
                lay = runner.ex.available_to_lay[0].price if runner.ex.available_to_lay else None
                back_size = runner.ex.available_to_back[0].size if runner.ex.available_to_back else 0
                odds[runner.selection_id] = {
                    "back": back,
                    "lay": lay,
                    "back_size": back_size,
                }

    return odds


def insert_into_db(races, runners, target_date):
    """Insert race cards into DB for feature computation."""
    con = get_db(str(ROOT / "racing.duckdb"))
    now_utc = datetime.now(timezone.utc)
    target_str = str(target_date)

    con.execute(f"DELETE FROM results WHERE race_id IN (SELECT race_id FROM races WHERE race_date = '{target_str}')")
    con.execute(f"DELETE FROM runners WHERE race_id IN (SELECT race_id FROM races WHERE race_date = '{target_str}')")
    con.execute(f"DELETE FROM races WHERE race_date = '{target_str}'")

    for race in races:
        con.execute("""INSERT INTO races (race_id, source_race_id, course_id, course_name, race_date,
            scheduled_off_utc, field_size, country, race_type,
            event_timestamp_utc, decision_cutoff_utc, ingest_timestamp_utc, is_standard_race)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,TRUE)""",
            [race["race_id"], race["market_id"], race["course_id"], race["course_name"],
             race["race_date"], race["scheduled_off_utc"], race["field_size"], race["country"],
             race["race_type"], race["scheduled_off_utc"], race["decision_cutoff_utc"], now_utc])

    for r in runners:
        con.execute("""INSERT INTO runners (runner_id, race_id, horse_id, horse_name,
            trainer_name, trainer_id, jockey_name, jockey_id,
            draw, weight_lbs, age, official_rating, headgear, headgear_first_time,
            event_timestamp_utc, decision_cutoff_utc, ingest_timestamp_utc)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,FALSE,?,?,?)""",
            [r["runner_id"], r["race_id"], r["horse_id"], r["horse_name"],
             r["trainer_name"], r["trainer_id"], r["jockey_name"], r["jockey_id"],
             r["draw"], r["weight_lbs"], r["age"], r["official_rating"], r["headgear"],
             races[0]["scheduled_off_utc"], races[0]["decision_cutoff_utc"], now_utc])

        con.execute("""INSERT INTO results (result_id, race_id, runner_id, horse_id,
            won, event_timestamp_utc, decision_cutoff_utc, ingest_timestamp_utc)
            VALUES (?,?,?,?,FALSE,?,?,?)""",
            [f"{r['runner_id']}_res", r["race_id"], r["runner_id"], r["horse_id"],
             races[0]["scheduled_off_utc"], races[0]["decision_cutoff_utc"], now_utc])

    con.close()
    return len(races), len(runners)


def rebuild_features():
    """Rebuild feature store including new data."""
    con = get_db(str(ROOT / "racing.duckdb"))

    from quality.checks import ensure_standard_race_flag
    ensure_standard_race_flag(con)

    from pipelines.run_phase2_feature_store import _prepare_upstream_inputs, _materialize_feature_store
    _prepare_upstream_inputs(con)

    for sql_file in sorted(os.listdir(ROOT / "sql" / "features")):
        if sql_file.endswith(".sql"):
            con.execute((ROOT / "sql" / "features" / sql_file).read_text())

    rows = _materialize_feature_store(con)
    con.close()
    return rows


def load_model(category, params="tuned"):
    model_dir = MODELS_DIR / params / category
    cbm_path = model_dir / "model.cbm"
    lgbm_path = model_dir / "model.lgbm"
    if cbm_path.exists():
        from catboost import CatBoost
        model = CatBoost()
        model.load_model(str(cbm_path))
        return model, None
    model = lgb.Booster(model_file=str(lgbm_path))
    calibrator = None
    cal_path = model_dir / "calibrator.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
    return model, calibrator


def race_softmax(scores, race_ids):
    out = np.zeros_like(scores, dtype=float)
    start = 0
    n = len(scores)
    while start < n:
        rid = race_ids[start]
        end = start + 1
        while end < n and race_ids[end] == rid:
            end += 1
        chunk = scores[start:end]
        chunk = chunk - np.max(chunk)
        exps = np.exp(chunk)
        out[start:end] = exps / np.sum(exps)
        start = end
    return out


def renormalize(probs, race_ids):
    out = np.zeros_like(probs)
    start = 0
    n = len(probs)
    while start < n:
        rid = race_ids[start]
        end = start + 1
        while end < n and race_ids[end] == rid:
            end += 1
        chunk = probs[start:end]
        s = chunk.sum()
        out[start:end] = chunk / s if s > 0 else 1.0 / (end - start)
        start = end
    return out


# Map raw feature names to punter-friendly signal labels (many features -> one label)
FEATURE_LABELS = {
    "horse_weighted_form": "recent form", "horse_form_trend": "improving form",
    "horse_place_rate_last_5": "place record", "horse_place_rate_last_10": "place record",
    "horse_avg_position_pct_last_5": "finishing positions", "position_consistency": "consistency",
    "horse_improvement_index": "improving profile",
    "avg_speed_last_3": "speed figures", "best_speed_last_5": "speed figures",
    "last_run_speed": "speed figures",
    "horse_best_rpr_rp_last_5": "Racing Post Rating", "horse_avg_rpr_last_3": "Racing Post Rating",
    "horse_last_rpr": "Racing Post Rating", "horse_best_rpr_last_5": "Racing Post Rating",
    "last_run_btn_lengths": "beaten distances", "avg_btn_last_3": "beaten distances",
    "horse_class_delta": "class drop", "is_class_dropper": "class drop",
    "prize_money_log": "race quality", "race_grade": "race grade",
    "horse_days_since_last_run": "freshness", "horse_age": "age profile",
    "career_runs": "experience", "career_win_rate": "career record",
    "career_place_rate": "career record",
    "runner_official_rating": "official rating", "rating_vs_top": "official rating",
    "rating_vs_field_avg": "official rating", "field_avg_rating": "official rating",
    "weight_vs_top": "weight", "weight_vs_field_avg": "weight", "weight_lbs": "weight",
    "weight_change_lbs": "weight change",
    "jockey_win_rate_90d": "jockey form", "jockey_win_rate_course_90d": "jockey course form",
    "jockey_dist_win_rate_90d": "jockey form", "jockey_trainer_combo_win_rate": "jockey/trainer combo",
    "jockey_upgrade_signal": "jockey upgrade", "jockey_runs_90d": "jockey form",
    "trainer_win_rate_90d": "trainer form", "trainer_win_rate_course_90d": "trainer course form",
    "trainer_course_going_win_rate": "trainer course form", "trainer_dist_alltime_win_rate": "trainer at distance",
    "trainer_win_rate_going_90d": "trainer on going", "trainer_win_rate_dist_band_90d": "trainer at distance",
    "trainer_fresh_win_rate": "trainer with fresh horses", "trainer_win_rate_14d": "trainer form",
    "collateral_beaten_win_rate": "form of beaten rivals", "collateral_beaten_place_rate": "form of beaten rivals",
    "collateral_beaten_count": "form of beaten rivals",
    "draw_position": "draw", "draw_field_percentile": "draw", "draw_bias_coefficient": "draw bias",
    "pace_pressure_index": "race pace", "pace_front_runners": "race pace", "pace_hold_up_horses": "race pace",
    "horse_going_group_place_rate": "going suitability", "going_encoded": "going suitability",
    "horse_distance_place_rate": "distance suitability", "horse_distance_affinity": "distance suitability",
    "trip_change_furlongs": "trip change", "distance_furlongs": "race distance",
    "horse_course_place_rate": "course record", "horse_course_affinity": "course record",
}


def _extract_signals(feature_names, shap_matrix, top_k=3):
    """Top positive SHAP drivers per row -> deduped human-readable signal strings."""
    n_feat = len(feature_names)
    contribs = np.asarray(shap_matrix)[:, :n_feat]
    out = []
    for row in contribs:
        order = np.argsort(row)[::-1]
        labels = []
        for idx in order:
            if row[idx] <= 0:
                break
            lab = FEATURE_LABELS.get(feature_names[idx])
            if lab and lab not in labels:
                labels.append(lab)
            if len(labels) >= top_k:
                break
        out.append(", ".join(labels))
    return out


def _compute_signals(model, X, category):
    """Per-runner top model drivers via SHAP. Returns list aligned with X rows."""
    try:
        if category == "flat":
            from catboost import Pool
            shap = model.get_feature_importance(data=Pool(X), type="ShapValues")
        else:
            shap = model.predict(X, pred_contrib=True)
        return _extract_signals(list(X.columns), shap)
    except Exception as exc:  # signals are a nice-to-have; never break scoring
        print(f"  (model signals unavailable: {exc})", flush=True)
        return [""] * len(X)


def load_and_score(target_date, category, params="tuned"):
    """Load features and score runners."""
    db = get_db(str(ROOT / "racing.duckdb"))

    type_filters = {
        "flat": "AND ra.race_type = 'Flat'",
        "jumps": "AND ra.race_type IN ('Chase', 'Hurdle', 'NH Flat')",
        "chase": "AND ra.race_type = 'Chase'",
        "hurdle": "AND ra.race_type IN ('Hurdle', 'NH Flat')",
    }
    type_filter = type_filters[category]

    df = db.execute(f"""
        SELECT fs.*, ra.race_type, ra.course_name, ra.scheduled_off_utc,
            ru.horse_name, ru.trainer_name, ru.jockey_name
        FROM feature_store fs
        JOIN races ra ON fs.race_id = ra.race_id
        JOIN runners ru ON fs.runner_id = ru.runner_id
        WHERE fs.race_date = '{target_date}'
        {type_filter}
        ORDER BY ra.scheduled_off_utc, fs.race_id
    """).df()
    db.close()

    if len(df) == 0:
        return None, None

    model, calibrator = load_model(category, params)

    if category == "flat":
        available = [f for f in FLAT_V2_FEATURES if f in df.columns]
        X = df[available].copy()
        for col in FLAT_V2_FEATURES:
            if col not in X.columns:
                X[col] = np.nan
        X = X[FLAT_V2_FEATURES]
    elif category in ("jumps", "chase", "hurdle"):
        expected_features = model.feature_name()
        meta_cols = ["race_type", "course_name", "scheduled_off_utc", "horse_name", "trainer_name", "jockey_name"]
        drop_list = JUMPS_DROP
        drop_cols = [c for c in EXCLUDE + drop_list + meta_cols if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore")

        non_numeric = [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
        for col in non_numeric:
            X[col] = pd.Categorical(X[col].astype(str)).codes

        for col in expected_features:
            if col not in X.columns:
                X[col] = np.nan
        X = X[expected_features]

    race_ids = df["race_id"].to_numpy()
    probs = race_softmax(model.predict(X), race_ids)

    if calibrator:
        probs = calibrator.transform(probs)
        probs = np.nan_to_num(probs, nan=1e-6)
        probs = np.clip(probs, 1e-6, 1.0)
        probs = renormalize(probs, race_ids)

    df = df.copy()
    df["model_signals"] = _compute_signals(model, X, category)

    return df, probs


def print_predictions(df, probs, category, runner_data, live_odds, min_edge=0.0):
    """Print formatted race cards with live exchange odds."""
    sel_lookup = {r["runner_id"]: r.get("selection_id") for r in runner_data}
    cloth_lookup = {r["runner_id"]: r.get("cloth_number") for r in runner_data}

    out = pd.DataFrame({
        "race_id": df["race_id"].values,
        "runner_id": df["runner_id"].values if "runner_id" in df.columns else None,
        "time": df["scheduled_off_utc"].values if "scheduled_off_utc" in df.columns else None,
        "course": df["course_name"].values if "course_name" in df.columns else None,
        "horse": df["horse_name"].values if "horse_name" in df.columns else None,
        "trainer": df["trainer_name"].values if "trainer_name" in df.columns else None,
        "jockey": df["jockey_name"].values if "jockey_name" in df.columns else None,
        "model_prob": probs,
        "model_odds": np.round(1.0 / np.clip(probs, 1e-6, 1.0), 1),
        "signals": df["model_signals"].values if "model_signals" in df.columns else "",
    })
    out = out.sort_values(["race_id", "model_prob"], ascending=[True, False])

    has_live = len(live_odds) > 0
    value_bets = []
    top_picks = []

    for race_id, group in out.groupby("race_id", sort=False):
        race = group.iloc[0]
        course = str(race.get("course", "Unknown")).split(" ")[0]
        time_val = race.get("time", "")
        uk_time = to_uk_time(time_val) if hasattr(time_val, "astimezone") else str(time_val)[:5]

        print(f"\n{'='*75}")
        print(f"  {uk_time}  {course}  [{category.upper()}]")
        print(f"{'='*75}")

        if has_live:
            print(f"  {'#':<3} {'Horse':<20} {'Model%':>6} {'MOdds':>5} {'Back':>5} {'Lay':>5} {'£Avl':>5} {'Edge':>6}")
            print(f"  {'-'*58}")
        else:
            print(f"  {'#':<3} {'Horse':<20} {'Trainer':<16} {'Jockey':<14} {'Model%':>6} {'MOdds':>5}")
            print(f"  {'-'*68}")

        for i, (_, row) in enumerate(group.iterrows(), 1):
            horse = str(row.get("horse", "?"))[:19]
            prob = f"{row['model_prob']*100:.1f}%"
            m_odds = f"{row['model_odds']:.1f}"
            back = lay = avl = edge_val = None

            if has_live:
                sel_id = sel_lookup.get(row.get("runner_id"))
                odds_data = live_odds.get(sel_id, {}) if sel_id else {}
                back = odds_data.get("back")
                lay = odds_data.get("lay")
                avl = odds_data.get("back_size", 0)

                back_str = f"{back:.1f}" if back else "-"
                lay_str = f"{lay:.1f}" if lay else "-"
                avl_str = f"£{avl:.0f}" if avl else "-"

                edge = ""
                marker = ""
                if back and back > 1:
                    implied = 1.0 / back
                    edge_val = row["model_prob"] - implied
                    edge = f"{edge_val*100:+.1f}%"
                    if edge_val > min_edge:
                        marker = " <<< BET"
                        cloth = cloth_lookup.get(row.get("runner_id"))
                        value_bets.append({
                            "cloth": cloth, "rank": i,
                            "horse": row.get("horse"),
                            "runner_id": row.get("runner_id"),
                            "race_id": race_id,
                            "course": course, "time": uk_time,
                            "model_prob": row["model_prob"], "back": back,
                            "edge": edge_val, "avl": avl,
                            "category": category,
                            "model_signals": row.get("signals", ""),
                        })

                print(f"  {i:<3} {horse:<20} {prob:>6} {m_odds:>5} {back_str:>5} {lay_str:>5} {avl_str:>5} {edge:>6}{marker}")
            else:
                trainer = str(row.get("trainer", ""))[:15]
                jockey = str(row.get("jockey", ""))[:13]
                print(f"  {i:<3} {horse:<20} {trainer:<16} {jockey:<14} {prob:>6} {m_odds:>5}")

            if i == 1:
                top_picks.append({
                    "race_id": race_id,
                    "runner_id": row.get("runner_id"),
                    "horse": row.get("horse"),
                    "course": course,
                    "time": uk_time,
                    "category": category,
                    "model_prob": row["model_prob"],
                    "model_odds": row["model_odds"],
                    "back": back,
                    "lay": lay,
                    "avl": avl,
                    "edge": edge_val,
                    "model_signals": row.get("signals", ""),
                })

    if value_bets:
        print(f"\n{'='*85}")
        print(f"  VALUE BETS — {category.upper()} ({len(value_bets)} selections, edge>{min_edge:.0%})")
        print(f"{'='*85}")
        print(f"  {'Time':<6} {'No.':<4} {'Horse':<20} {'Course':<12} {'Rank':>4} {'Model%':>6} {'Back':>5} {'Edge':>6} {'Avail':>6}")
        print(f"  {'-'*73}")
        for vb in sorted(value_bets, key=lambda x: x["edge"], reverse=True):
            cloth_str = str(vb['cloth']) if vb['cloth'] else "?"
            print(f"  {str(vb['time']):<6} {cloth_str:<4} {str(vb['horse'])[:19]:<20} {str(vb['course'])[:11]:<12} {vb['rank']:>4} {vb['model_prob']*100:.1f}% {vb['back']:>5.1f} {vb['edge']*100:>+5.1f}% £{vb['avl']:>4.0f}")

    return out, value_bets, top_picks


def save_bets_log(value_bets: list[dict], target_date: date, refresh: bool = False):
    """Append today's value bets to logs/daily_bets.csv (date-deduped)."""
    if not value_bets:
        return
    new_rows = pd.DataFrame([{
        "date": str(target_date),
        "race_id": vb.get("race_id"),
        "runner_id": vb.get("runner_id"),
        "horse": vb.get("horse"),
        "course": vb.get("course"),
        "time": vb.get("time"),
        "category": vb.get("category"),
        "model_prob": round(vb["model_prob"], 4),
        "back_odds": vb.get("back"),
        "edge": round(vb["edge"], 4),
        "stake": 1.0,
        "model_signals": vb.get("model_signals", ""),
    } for vb in value_bets])
    append_dated_csv(ROOT / "logs" / "daily_bets.csv", new_rows, target_date,
                     BETS_LOG_COLS, refresh=refresh, label="bets")


TOP_PICKS_LOG_COLS = [
    "date", "race_id", "runner_id", "horse", "course", "time", "category",
    "model_prob", "model_odds", "back", "lay", "avl", "edge", "model_signals",
]


def save_top_picks(top_picks: list[dict], target_date: date, refresh: bool = False):
    """Write the model's #1-rated horse per race to logs/daily_top_picks.csv.

    Unlike save_bets_log this records every race (not just value bets), so a private
    Discord channel can mirror the terminal's top-of-card view.
    """
    if not top_picks:
        return
    new_rows = pd.DataFrame([{
        "date": str(target_date),
        "race_id": tp.get("race_id"),
        "runner_id": tp.get("runner_id"),
        "horse": tp.get("horse"),
        "course": tp.get("course"),
        "time": tp.get("time"),
        "category": tp.get("category"),
        "model_prob": round(tp["model_prob"], 4),
        "model_odds": tp.get("model_odds"),
        "back": tp.get("back"),
        "lay": tp.get("lay"),
        "avl": tp.get("avl"),
        "edge": round(tp["edge"], 4) if tp.get("edge") is not None else None,
        "model_signals": tp.get("model_signals", ""),
    } for tp in top_picks])
    append_dated_csv(ROOT / "logs" / "daily_top_picks.csv", new_rows, target_date,
                     TOP_PICKS_LOG_COLS, refresh=refresh, label="top picks")


def main():
    parser = argparse.ArgumentParser(description="Daily race predictions from Betfair API")
    parser.add_argument("--date", type=str, required=True, help="Date: YYYY-MM-DD, 'today', or 'tomorrow'")
    parser.add_argument("--params", type=str, default="tuned", choices=["tuned", "default"])
    parser.add_argument("--min-edge", type=float, default=0.15)
    parser.add_argument("--flat", action="store_true", help="Flat races only")
    parser.add_argument("--jumps", action="store_true", help="Jumps races only")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-rebuild", action="store_true", help="Skip feature store rebuild")
    parser.add_argument("--refresh", action="store_true", help="Replace today's bets in the log (re-log with fresh signals)")
    args = parser.parse_args()

    target_date = resolve_date(args.date)
    print(f"Predictions for {target_date}", flush=True)

    print("Fetching race cards from Betfair API...", flush=True)
    races, runners = fetch_race_cards(target_date)
    print(f"  {len(races)} races, {len(runners)} runners", flush=True)

    if not races:
        print("No races found for this date.")
        return

    # Check if today's data is already in DB
    db_check = get_db(str(ROOT / "racing.duckdb"))
    existing = db_check.execute(f"SELECT COUNT(*) FROM races WHERE race_date = '{target_date}'").fetchone()[0]
    fs_existing = db_check.execute(f"SELECT COUNT(*) FROM feature_store WHERE race_date = '{target_date}'").fetchone()[0]
    db_check.close()

    if existing > 0 and fs_existing > 0:
        print(f"  Already in DB: {existing} races, {fs_existing} feature rows (skipping insert + rebuild)", flush=True)
    else:
        if existing == 0:
            print("Inserting into DB...", flush=True)
            n_races, n_runners = insert_into_db(races, runners, target_date)
            print(f"  {n_races} races, {n_runners} runners inserted", flush=True)
        else:
            print(f"  {existing} races already in DB (skipping insert)", flush=True)

        if not args.skip_rebuild:
            print("Rebuilding feature store (this takes a few minutes)...", flush=True)
            rows = rebuild_features()
            print(f"  Feature store: {rows:,} rows", flush=True)

    # Fetch live exchange odds
    market_ids = [r["market_id"] for r in races]
    print("Fetching live exchange odds...", flush=True)
    try:
        live_odds = fetch_live_odds(market_ids)
        print(f"  Got prices for {len(live_odds)} runners", flush=True)
    except Exception as e:
        print(f"  Could not fetch live odds: {e}", flush=True)
        live_odds = {}

    if args.flat:
        categories = ["flat"]
    elif args.jumps:
        categories = ["chase", "hurdle"]
    else:
        categories = ["flat", "chase", "hurdle"]
    all_outputs = []
    all_value_bets = []
    all_top_picks = []

    # Build set of active selection_ids (have odds = still running)
    active_sels = set(live_odds.keys()) if live_odds else set()
    sel_to_runner = {r.get("selection_id"): r.get("runner_id") for r in runners}
    non_runner_ids = set()
    if active_sels:
        for r in runners:
            sel = r.get("selection_id")
            if sel and sel not in active_sels:
                non_runner_ids.add(r.get("runner_id"))
        if non_runner_ids:
            print(f"  {len(non_runner_ids)} non-runners detected (no odds), excluding from scoring", flush=True)

    for category in categories:
        print(f"\nScoring {category} races...", flush=True)
        df, probs = load_and_score(target_date, category, args.params)

        if df is None:
            print(f"  No {category} runners found")
            continue

        if non_runner_ids and "runner_id" in df.columns:
            mask = ~df["runner_id"].isin(non_runner_ids)
            if mask.sum() < len(df):
                removed = len(df) - mask.sum()
                df = df[mask].reset_index(drop=True)
                probs = probs[mask.values] if hasattr(mask, 'values') else probs[mask]
                probs = renormalize(probs, df["race_id"].to_numpy())
                print(f"  Removed {removed} non-runners, re-normalised probabilities", flush=True)

        n_races = df["race_id"].nunique()
        print(f"  {len(df)} runners across {n_races} races", flush=True)

        out, value_bets, top_picks = print_predictions(df, probs, category, runners, live_odds, min_edge=args.min_edge)
        out["category"] = category
        all_outputs.append(out)
        all_value_bets.extend(value_bets)
        all_top_picks.extend(top_picks)

    if all_value_bets:
        save_bets_log(all_value_bets, target_date, refresh=args.refresh)

    if all_top_picks:
        save_top_picks(all_top_picks, target_date, refresh=args.refresh)

    if all_outputs and args.output:
        combined = pd.concat(all_outputs)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
