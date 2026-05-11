# Racing Model — Architecture and Design

## Mission

Price-blind horse racing probability engine. Predicts calibrated per-runner win probabilities from fundamentals only, then compares to Betfair SP to find value bets. Separate flat and jumps models.

### Why Price-Blind

Market prices are never a model input. The model produces a signal fully independent of the market, so any edge is structural — based on factors the market systematically underweights. Prices enter only at the betting stage as a benchmark.

### Non-Negotiables

- No random splits — temporal walk-forward validation only
- No leakage — every feature provably available at decision cutoff (60 min before off)
- No model acceptance without walk-forward ROI across multiple independent time windows
- No market price as a model input

---

## Data Pipeline

### Sources

1. **Betfair SP History** — Public CSV at promo.betfair.com. Race spine: event_id, horse, BSP, win/lose, course, date. Covers 2007-present.
2. **rpscrape** — Open-source Racing Post scraper. Trainer, jockey, draw, weight, age, official rating, headgear, race times, beaten lengths, going, distance, class, prize money. Monthly CSVs.
3. **Betfair Exchange API** — Live odds snapshots for forward predictions (requires API credentials).

### Pipeline Stages

```
1. SP History → DuckDB (races, runners, results, horse_history)
2. rpscrape CSVs → Enrichment (trainer, jockey, draw, weight, class, going, times, headgear, non-completions)
3. Rematch pass → Fuzzy matching for missed runners (Jaro-Winkler, date-shifted, card overlap)
4. Quality gates → DQ checks, leakage guard, coverage thresholds
5. Feature store → 9 SQL feature files → materialised feature_store table
6. Training → Separate flat/jumps LightGBM LambdaRank models
7. Calibration → Isotonic regression on held-out calibration set
8. Validation → Walk-forward across 4 independent time windows
```

### Database Schema (DuckDB)

7 core tables: `races`, `runners`, `results`, `odds_snapshots`, `horse_history`, `trainer_history`, `jockey_history`

Key columns:
- `decision_cutoff_utc` — 60 min before scheduled off. All features must use data available before this timestamp.
- `is_standard_race` — filters out non-race markets (forecasts, specials, etc.)
- `non_completion` — PU/F/UR/BD codes for jumps non-finishers

---

## Feature Engineering

9 SQL feature files producing ~80 features, ~65-69 used per model after dropping low-importance ones.

### Feature Files

| File | Table | Features |
|------|-------|----------|
| 001_horse_form.sql | f001 | Weighted form, positions, win/place rates, going/distance/course affinity, form trend, RPR, class delta, headgear |
| 002_draw_bias.sql | f002 | Draw position, field percentile, course/going win rate, bias coefficient |
| 003_trainer_stats.sql | f003 | 90-day win rates (overall, course, going, distance), 14-day hot streak, fresh horse record, course×going interaction |
| 004_jockey_stats.sql | f004 | 90-day win rates (overall, course), trainer combo, distance specialisation |
| 005_class_features.sql | f005 | Class encoding, handicap flag, prize money, class delta |
| 006_race_context.sql | f006 | Field size, pace pressure, surface, distance, going, month |
| 007_collateral_form.sql | f007 | Franked form — subsequent win/place rate of beaten opponents |
| 008_runner_profile.sql | f008 | Weight vs field, age, official rating vs field, career stats, consistency |
| 009_speed_and_changes.sql | f009 | Speed figures, trip change, weight change, beaten lengths, jockey upgrade signal, trainer 14-day form |

### Key Features (by importance)

1. **horse_weighted_form** — Recency-weighted finishing position / field size. #1 feature in both models.
2. **rating_vs_top** — Official rating gap to top-rated in field
3. **avg_btn_last_3** — Average beaten lengths in last 3 runs
4. **trainer_dist_alltime_win_rate** — Trainer × distance interaction
5. **horse_class_delta** — Current class vs recent average (dropping in class = positive signal)
6. **horse_days_since_last_run** — Freshness vs fitness
7. **jockey_win_rate_90d** — Jockey current form
8. **collateral_beaten_place_rate** — Franked form (opponents' subsequent performance)
9. **horse_completion_rate** — Jumps-specific: % of races completed (not fallen/PU)

### Data Fixes Applied

- `horse_history.finishing_position` backfilled from results (10.7% → 91.5% coverage)
- `is_handicap` populated from rpscrape race names via horse-name matching (0 → 89k races)
- `race_class` inferred from prize money for Irish races (74% → 99.6% coverage)
- `headgear` backfilled from rpscrape (0 → 494k runners)
- `official_time_secs` and `btn_lengths` backfilled from rpscrape (0 → 91% coverage)
- `non_completion` codes (PU/F/UR) backfilled from rpscrape (77k entries)
- Course venue normalisation (stripped date suffixes from Betfair course_id)
- Distance capped at 36 furlongs (fixed meters-as-furlongs parsing bug)

---

## Model Architecture

### Two Separate Models

| | Flat | Jumps |
|---|---|---|
| Race types | Flat only | Chase, Hurdle, NH Flat |
| Features | ~69 (includes draw) | ~65 (no draw features) |
| Key differences | Draw, speed figures matter | Completion rate, going affinity matter more |

### Training

- **Objective**: LightGBM LambdaRank (learning-to-rank within each race)
- **Calibration**: Isotonic regression on held-out calibration set, then per-race renormalisation
- **Early stopping**: On calibration set NDCG@1

### Tuned Hyperparameters (Optuna, 80 trials)

| Param | Flat | Jumps |
|-------|------|-------|
| learning_rate | 0.044 | 0.045 |
| num_leaves | 57 | 100 |
| min_child_samples | 69 | 63 |
| max_depth | 7 | 9 |
| subsample | 0.88 | 0.74 |
| colsample_bytree | 0.96 | 0.59 |

### Walk-Forward Split

```
Window 1: Train 2015-2020 → Cal 2021 → Test 2022
Window 2: Train 2015-2021 → Cal 2022 → Test 2023
Window 3: Train 2015-2022 → Cal 2023 → Test 2024
Window 4: Train 2015-2023 → Cal 2024 → Test 2025-2026
```

---

## Value Betting

### How It Works

1. Model outputs calibrated P(win) per runner
2. Compare to Betfair SP: `edge = P(win) - (1 / BSP)`
3. Bet when `edge > 5%`
4. Flat: no filters (outsiders are profitable)
5. Jumps: optionally filter novice longshots and outsiders 50+

### Results (Walk-Forward, Tuned Params)

| Window | Flat ROI | Jumps ROI |
|--------|----------|-----------|
| Test 2022 | +11.21% | +18.39% |
| Test 2023 | +6.14% | +11.79% |
| Test 2024 | +5.23% | +7.09% |
| Test 2025-26 | +8.31% | +8.32% |
| **Total** | **+7.69%** | **+11.58%** |

93,058 total bets, +9.33% combined ROI, all 8 windows profitable.

### Key Findings

- **Flat outsiders (SP 50+) are highly profitable** — don't filter them out
- **Jumps loses money on soft/heavy going** — the model's calibration struggles in extreme conditions
- **Non-completion features transformed jumps** — knowing which horses fall/pull up regularly is critical
- **Handicaps and non-handicaps behave differently** — non-handicaps have higher top-pick rates but handicaps have more value betting opportunities
- **The market is almost perfectly efficient at short prices (SP < 3)** — no edge to find there

---

## Future Work

- **Production pipeline** — Score tomorrow's races, compare to live exchange prices, output bet recommendations
- **Staking strategy** — Fractional Kelly criterion instead of level stakes
- **Place model** — Predict P(place) for each-way betting
- **Live monitoring** — Track actual P&L, detect model drift, know when to retrain
- **Form features** — More sophisticated weighted form, consistency metrics
- **Seasonal retraining** — Retrain quarterly with latest data
