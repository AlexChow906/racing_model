# Racing Model Grand Plan (Zero-Budget, Production-Grade)
# Version 2.0 — Price-Blind Fundamentals Engine

---

## 1. Mission and Non-Negotiables

### 1.1 Core Mission

Build a race-level probability engine that:

- Outputs calibrated per-runner win probabilities derived **exclusively from fundamental race factors** — never from market prices.
- Enforces probability mass conservation per race (sum = 1.0) via softmax normalisation at the race group level.
- Produces positive expected value opportunities by comparing model output against pre-race market prices **after** the model has produced its estimate — prices are never an input.
- Survives real-world deployment under strict out-of-sample walk-forward validation across multiple independent time windows.

### 1.2 Why Price-Blind

The model intentionally excludes all market pricing data as input features. This is a deliberate architectural decision, not a limitation.

A price-blind model produces a signal that is **fully orthogonal to the market**. This means:

- Edge is structural and explainable — it comes from factors the market systematically underweights, not from attempting to out-read the market's own pricing.
- The model cannot degenerate into learning to trust the market, which would add no independent information.
- Calibration failures are diagnosable — if the model is wrong it is a fundamentals problem, not a market-reading problem.
- Edge is durable — structural biases in how markets price draw, trainer intent, class adjustment, and going sensitivity tend to persist for years.

Market implied probabilities are used **only** in the value engine as a benchmark to compare against, never as a training feature.

### 1.3 Non-Negotiables

- No random splits. Only temporal walk-forward validation with strict chronological ordering.
- No leakage. Every feature must be provably available at the decision cutoff time.
- No model acceptance without calibration checks, reliability curves, and betting simulation.
- ROI is the primary objective metric. Calibration error and maximum drawdown are hard constraints — a model failing either is rejected regardless of ROI.
- No market price in any form as a model input. Prices enter only at the value engine stage.
- No threshold tuning on the final held-out test window. Thresholds are set on validation only.

---

## 2. System Architecture

### 2.1 Storage and Compute

- **Primary database:** DuckDB (single local file, columnar analytical engine). Handles 50GB+ comfortably on a laptop with 10-100x faster analytical queries than SQLite.
- **Raw market data:** Parquet files partitioned by date and source, stored immutably. DuckDB can query Parquet directly without importing.
- **Feature store:** Materialised DuckDB tables and views with point-in-time guarantees.
- **Model artifacts:** Serialised models with full metadata (hyperparameters, training window, feature set version, validation metrics).
- **Experiment tracking:** Local JSON logs in `experiments/` — one file per run, immutable after creation.
- **Paper trading log:** Append-only CSV tracking every flagged bet, market odds at flag time, and eventual result.

### 2.2 Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   DATA INGESTION LAYER                  │
│  Betfair API (results, form)                            │
│  Sporting Life scraper (cards, going, draw)             │
│  BHA public data (official ratings, results)            │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│              DATA QUALITY AND CONTRACT LAYER            │
│  Timestamp validation, entity key resolution            │
│  Leakage detection checks, schema validation            │
│  Deduplication, outlier flagging                        │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│               FEATURE ENGINEERING LAYER                 │
│  Point-in-time safe aggregations only                   │
│  Horse form, trainer/jockey, course/draw/going/class    │
│  All features logged with leakage risk rating           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            MODELLING AND CALIBRATION LAYER              │
│  LightGBM LambdaRank (race-group level)                 │
│  Softmax normalisation per race                         │
│  Isotonic regression calibration                        │
│  Ensemble with CatBoost ranker                          │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            VALUE ENGINE (PRICES ENTER HERE ONLY)        │
│  Load evening-before market odds (21:00 snapshot)       │
│  Compute overround-corrected implied probabilities      │
│  Compute edge = p_model_calibrated - p_market_fair      │
│  Apply bet trigger gates                                │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│           BACKTESTING AND RISK FRAMEWORK                │
│  Realistic simulation with commission and slippage      │
│  Walk-forward ROI, drawdown, CLV, Brier decomposition   │
│  Segment-level diagnostics by angle                     │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│            LIVE INFERENCE AND MONITORING                │
│  Pre-race scheduler at fixed decision times             │
│  Drift monitoring, calibration tracking, CLV rolling    │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Repository Structure

```
racing-model/
├── data/
│   ├── raw/                    # Immutable Parquet snapshots by date/source
│   └── intermediate/           # Normalised event/runner tables
├── sql/
│   ├── schema/                 # Table definitions
│   └── features/               # Point-in-time feature SQL
├── src/
│   ├── ingestion/              # Betfair API client, scrapers, normalisation
│   ├── quality/                # DQ checks, timestamp validation, leakage guards
│   ├── features/               # Feature generation pipelines
│   ├── modeling/               # Training, ranking, calibration, ensembling
│   ├── backtest/               # Simulation, ROI, drawdown, CLV
│   └── live/                   # Scoring, value engine, bet candidate output
├── models/                     # Trained artifacts by version ID
├── experiments/                # Immutable JSON logs per experiment run
├── reports/                    # Walk-forward and strategy reports
├── configs/                    # YAML strategy and model configs
├── logs/
│   └── paper_trades.csv        # Append-only paper trade log
└── racing.duckdb               # Primary analytical database
```

---

## 3. Data Contracts (Anti-Leakage Foundation)

### 3.1 Time Semantics

Every row in every table must carry three timestamps:

- `event_timestamp_utc` — when the real-world event occurred (source truth)
- `ingest_timestamp_utc` — when your pipeline captured it
- `decision_cutoff_utc` — the point-in-time used for prediction (default: 21:00 UTC the evening before the race date)

**A feature is legal if and only if its source data `event_timestamp_utc` is strictly less than `decision_cutoff_utc` for that race.**

This rule is enforced programmatically in the feature store — no manual judgement calls.

### 3.2 Entity Keys

Every record is keyed by a stable canonical set:

- `race_id` — unique race identifier, stable across sources
- `runner_id` — unique runner within a race
- `horse_id` — stable horse identifier across all races
- `trainer_id`, `jockey_id`
- `course_id`, `surface`, `race_type`, `distance_furlongs`, `going_code`

Source-specific IDs (Betfair selection ID, BHA race ID) are stored as foreign keys and mapped to canonical keys at ingest time.

### 3.3 Mandatory Base Tables

```
races               — one row per race (race_id, course, date, distance, going, class, field_size)
runners             — one row per runner per race (race_id, runner_id, horse_id, trainer_id, jockey_id, draw, weight)
results             — official finishing positions and times, ingest AFTER result confirmed
odds_snapshots      — market prices at fixed T-minus intervals (stored but NEVER used as model features)
horse_history       — point-in-time safe rolling form per horse
trainer_history     — rolling trainer stats by course/surface/class/going
jockey_history      — rolling jockey stats with trainer interaction
course_bias         — draw bias and pace bias coefficients per course per going band
feature_store       — materialised point-in-time safe feature rows, one per runner per race
```

### 3.4 Leakage Blacklist

The following must never appear as model input features under any circumstances:

- Starting Price (BSP or SP) — set at or after race off
- Post-race official ratings updated after result time
- Any odds, implied probability, market movement, or volume data
- Final finishing positions of other runners in the same race
- Any result-derived feature from the same race being predicted
- "Final" market states not available at the decision cutoff
- Betfair BSP (available only post-race in historical feeds)

Leakage checks run automatically as part of the feature build pipeline. Any feature with `feature_timestamp > decision_cutoff_utc` raises a hard error and halts the pipeline.

---

## 4. Ingestion Strategy

### 4.1 Free Data Sources

| Source | Data | Access Method |
|---|---|---|
| Betfair Exchange API | Historical results, runner metadata | Free with funded account, `betfairlightweight` |
| Sporting Life | Race cards, going, draw, form | Scraping with `requests` + `BeautifulSoup` |
| At The Races | Race cards, trainer/jockey stats | Scraping |
| BHA Official Site | Official ratings, going reports, results | Public, structured |
| GitHub / Kaggle | Historical result CSVs | One-time download for bootstrap |

Betfair is the most important source. Their historical results and runner data give you a clean, structured foundation. The exchange never had a meaningful overround, making it the cleanest benchmark for the value engine.

### 4.2 What to Collect Per Race

For each race:
- Course, date, distance, going, race class/grade, prize money, race type (flat/jumps/AW), field size, number of runners declared vs ran

For each runner:
- Horse name and stable ID, trainer, jockey, draw, weight carried, headgear, age, official rating, days since last run, number of career starts

For each horse's recent form (last 10 runs, point-in-time safe):
- Finishing position, field size, course, distance, going, class, weight, jockey, trainer, days between runs, headgear worn

### 4.3 Odds Collection (For Value Engine Only)

Collect Betfair odds snapshots at fixed intervals **stored but not fed to the model**:

- **Evening before (primary):** 21:00 UTC the night before — this is your decision point and the benchmark for the value engine
- **Morning of race day:** 09:00 UTC — used to detect major market moves since your prediction (non-runners, weather, significant money)
- **Near off:** T-5 minutes before scheduled off — used exclusively for CLV calculation post-race

The **21:00 evening snapshot** is the primary benchmark for the value engine — this is the price available to you when you make your decision. T-5 is used for CLV calculation only and never influences bet selection.

Store raw snapshots immutably in Parquet. Never overwrite. Append and version all transforms.

### 4.4 Data Quality Rules

Reject or flag records when:
- Missing `race_id` or `runner_id`
- Odds <= 1.01 or > 1000 (implausibly extreme)
- Duplicate runner entries at same source and timestamp
- Race has fewer than 2 active runners
- Timestamp ordering violation (ingest before event)
- Draw position missing for flat races on draw-sensitive courses
- Going code not in validated enum

---

## 5. Feature Engineering

### 5.1 The Signal — Where Edge Actually Lives

These are the documented structural inefficiencies the feature set is designed to capture. This is your competitive moat — the pipeline is commoditised, these features are not.

**Draw bias**
At Chester, Beverley, Carlisle, Windsor, and others, starting stall position is a dominant factor in sprint races. The effect is amplified on soft or heavy going. Build per-course per-going draw bias coefficients from historical results and apply as a feature. The market underweights this in non-obvious configurations (e.g. high draws on a course that normally favours low, after rail movement).

**Trainer intent and readiness**
Certain trainers run horses fit first time out after a lay-off. Others always need a run. Build per-trainer "fresh horse" win rate vs. "had a run" win rate, segmented by lay-off duration (0-30 days, 31-60, 61-90, 90+). The market systematically penalises absence without adjusting for trainer-specific patterns.

**Class-adjusted finishing position**
A 4th in a Group 1 represents better form than a 1st in a Class 5. Raw finishing positions without class adjustment are almost useless. Build a class-adjusted speed rating proxy: normalise finishing times (where available) against par times per course/distance/going, or use official ratings as a proxy where times are unavailable.

**Class trajectory**
Horses dropping significantly in class are systematically underrated, especially when their last run looks bad on paper but was in much stronger company. Build a class delta feature (current race class vs average class of last 3 runs).

**First-time headgear**
Blinkers or cheekpieces worn for the first time significantly increase win probability. The market partially prices this but not fully, especially for lower-grade races with less media attention.

**Going sensitivity**
Some horses have dramatically different form profiles on different going. Build per-horse going affinity scores: win rate and average finishing position on each going band. Weight recent runs more heavily. Flag mismatches between today's going and a horse's optimal going.

**Distance profile**
Build per-horse optimal distance curves. A horse's best distance may not be today's distance. Flag where today's distance is materially outside the horse's historical comfort zone.

**Pace setup / race shape**
Count the number of likely front-runners vs hold-up horses in each race. If all runners prefer to lead, they burn each other out. If all runners prefer to come from behind with no strong pace, hold-up horses will struggle. The market rarely models this explicitly.

**Trainer-jockey combination on specific courses**
Some combinations have anomalous strike rates at specific venues. Small sample sizes make the market lazy. Flag combinations with >20 runs and strike rate materially above overall baseline.

### 5.2 Feature Families

**Horse form features**
- Finishing positions in last 3, 5, 10 runs (weighted by recency)
- Class-adjusted position scores for last 3, 5, 10 runs
- Days since last run
- Number of runs in last 30, 90, 365 days
- Going affinity score (win rate and avg position on today's going)
- Distance affinity score (win rate and avg position at today's distance ± 1f)
- Course affinity score (win rate at today's course)
- Best speed/time rating from last 5 runs
- Form trend: improving, declining, or flat (slope of recent position scores)

**Trainer features**
- Rolling 90-day win strike rate overall
- Rolling 90-day win strike rate at today's course
- Rolling 90-day win strike rate on today's going band
- Rolling 90-day win strike rate at today's distance band
- Fresh horse win rate (>60 day lay-off) vs had-a-run win rate
- First-time headgear win rate

**Jockey features**
- Rolling 90-day win strike rate overall
- Rolling 90-day win strike rate at today's course
- Trainer-jockey combination rolling strike rate (90 days, min 10 runs)

**Race context features**
- Draw position (raw)
- Draw bias coefficient for this course × going combination
- Draw percentile within field (normalised 0-1)
- Field size
- Race class / grade (encoded ordinally)
- Class delta (today's class vs average of last 3 runs)
- Going code (encoded ordinally)
- Distance (furlongs, continuous)
- Surface (flat turf / flat AW / jumps)
- Race type (handicap / conditions / maiden / listed / group)
- Prize money (log-scaled)

**Horse profile features**
- Age
- Official rating (if available, as proxy for quality)
- Weight carried
- Headgear today vs last run (first time flag, removed flag)
- Days since last run

**Pace setup features**
- Count of likely front-runners in field
- Count of likely hold-up horses in field
- Pace pressure index (front-runners / field size)

### 5.3 Point-in-Time Guarantee

All rolling aggregations use windows ending strictly before `decision_cutoff_utc` for the race being predicted. The feature SQL enforces this with a strict `WHERE` clause on `event_timestamp_utc < decision_cutoff_utc`. This is tested automatically.

### 5.4 Feature Governance Log

For each feature, maintain a record in `configs/feature_registry.yaml`:

```yaml
horse_going_affinity_win_rate:
  definition: "Win rate of horse in last 20 runs on same going code as today's race"
  source_tables: [horse_history, races]
  timestamp_logic: "event_timestamp_utc < decision_cutoff_utc strictly"
  expected_range: [0.0, 1.0]
  null_policy: "impute with field average if fewer than 3 runs on going"
  leakage_risk: LOW
  first_introduced: "v1.0"
```

---

## 6. Modelling Strategy

### 6.1 Framing

This is not a binary classification problem. It is a **discrete choice / competing risks problem**. All runners in a race compete against each other simultaneously. The correct output is a probability distribution across all runners in a race that sums to exactly 1.0.

Treating each horse as an independent binary classification target ignores the fundamental structure of racing and produces probability estimates that cannot be meaningfully compared across a field.

### 6.2 Primary Model

**LightGBM with LambdaRank objective, grouped by `race_id`.**

This trains the model to rank the eventual winner highest within each race group — the correct optimisation target. It naturally handles variable field sizes.

```python
import lightgbm as lgb

model = lgb.LGBMRanker(
    objective="lambdarank",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
)

model.fit(
    X_train, y_train,
    group=train_group_sizes,         # number of runners per race
    eval_set=[(X_val, y_val)],
    eval_group=[val_group_sizes],
    callbacks=[lgb.early_stopping(50)]
)
```

### 6.3 Probability Conversion

Convert raw LightGBM scores to race-level win probabilities using softmax normalisation, applied per race group:

```python
import numpy as np
from scipy.special import softmax

def scores_to_probs(scores_df):
    """Convert raw model scores to calibrated race-level probabilities."""
    probs = (
        scores_df
        .groupby("race_id")["raw_score"]
        .transform(lambda x: softmax(x.values))
    )
    return probs

# Verify conservation
assert (scores_df.groupby("race_id")["win_prob"].sum() - 1.0).abs().max() < 1e-6
```

### 6.4 Calibration

Raw softmax probabilities are scores, not true probabilities. Calibrate using isotonic regression fitted on out-of-fold predictions:

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(oof_probs, oof_actuals)

calibrated_probs = calibrator.transform(raw_probs)
```

Re-normalise per race after calibration to restore sum-to-one:

```python
df["calibrated_prob"] = (
    df.groupby("race_id")["raw_calibrated"]
    .transform(lambda x: x / x.sum())
)
```

Validate calibration with a reliability diagram: bin predictions into deciles and verify that each bin's average predicted probability matches the empirical win rate. If your model assigns 25% to 100 horses, roughly 25 of them should win. Any systematic deviation indicates miscalibration.

Also compute:
- **Brier Score** (lower is better, measures calibration and resolution jointly)
- **ECE (Expected Calibration Error)** (target below 0.02)
- **Reliability diagram** visually

Segment calibrators by field size (2-5 runners, 6-10, 11-16, 17+) and by odds band if sample size allows.

### 6.5 Secondary Model for Ensemble

CatBoost ranker with equivalent hyperparameter structure. Blend calibrated probabilities using validation-window-weighted stacking:

```python
# Blend weights determined by validation ROI, not raw accuracy
ensemble_prob = 0.6 * lgbm_prob + 0.4 * catboost_prob

# Re-normalise per race
ensemble_prob = ensemble_prob / ensemble_prob.groupby(race_id).transform("sum")
```

### 6.6 Feature Importance and Explainability

Run SHAP analysis on the LightGBM model to understand which features drive predictions for specific races. This helps identify which structural angles are contributing the most edge and flags when a feature may be behaving unexpectedly.

---

## 7. Temporal Validation Design

### 7.1 Walk-Forward Windows

Never use a random split. Use rolling walk-forward windows with a gap between train end and validation start to prevent leakage from form features that span the boundary:

```
Window A:  Train 2018-2020  |  Gap 1m  |  Val 2021  |  Test 2022
Window B:  Train 2018-2021  |  Gap 1m  |  Val 2022  |  Test 2023
Window C:  Train 2018-2022  |  Gap 1m  |  Val 2023  |  Test 2024
Window D:  Train 2018-2023  |  Gap 1m  |  Val 2024  |  Test 2025
```

Hyperparameters and thresholds are tuned on validation windows only. Test windows are touched exactly once per model version, at the final evaluation stage.

### 7.2 Out-of-Sample Acceptance Criteria

A model version is promoted to shadow live only if ALL of the following are true:

- Out-of-sample ROI positive across at least 3 of 4 walk-forward test windows
- Maximum drawdown within tolerance (defined in config, e.g. -20% of bank)
- ECE below 0.02 on each test window
- Brier score improvement over naive market-implied baseline
- CLV non-negative trend (model prices beat close prices on average)
- No single angle (draw, trainer, class) driving all the ROI — must be diversified

### 7.3 Statistical Discipline

All ROI estimates are accompanied by bootstrap confidence intervals (1000 resamples of the bet sequence). A model with a 5% ROI point estimate but a 95% confidence interval spanning -3% to +13% is not ready for live betting.

Evaluate robustness across:
- Race type (flat turf, flat AW, jumps hurdles, jumps chase)
- Field size bands (2-5, 6-10, 11-16, 17+)
- Odds bands (1.5-3.0, 3.0-6.0, 6.0-12.0, 12.0+)
- Seasonal periods (Flat turf season, jumps season, all-weather winter)
- Course type (galloping, tight/turning, straight)

---

## 8. Value Engine

### 8.1 Prices Enter the System Here — Nowhere Else

At prediction time the model has already produced calibrated, price-blind win probabilities. The value engine loads the 21:00 UTC evening Betfair odds snapshot and performs a pure comparison.

### 8.2 Market Implied Probability

```python
def market_implied_prob(decimal_odds: float) -> float:
    """Raw implied probability from decimal odds."""
    return 1.0 / decimal_odds

def overround_correct(runners_df):
    """Remove bookmaker margin to get fair implied probabilities."""
    raw_implied = 1.0 / runners_df["decimal_odds"]
    overround = raw_implied.sum()
    runners_df["p_market_fair"] = raw_implied / overround
    return runners_df
```

### 8.3 Edge Calculation

```python
df["edge"] = df["p_model_calibrated"] - df["p_market_fair"]

# Expected value of a £1 bet
df["ev"] = (df["p_model_calibrated"] * df["decimal_odds"]) - 1.0
```

### 8.4 Bet Trigger Gates

A candidate bet must pass ALL of the following gates — failing any one gate means no bet:

1. **Edge threshold:** `edge >= 0.04` (start conservative, adjust after 500+ paper trades)
2. **Minimum EV:** `ev >= 0.05` (at least 5p expected return per £1 staked)
3. **Odds band:** Decimal odds between 2.5 and 15.0 (avoid odds-on and extreme outsiders initially)
4. **Liquidity:** Betfair traded volume at 21:00 evening snapshot above minimum threshold for the race type
5. **Calibration bucket confidence:** Horse falls in a calibration bucket with ECE below threshold
6. **No missing mandatory features:** All Tier 1 features present (no imputation on key inputs)
7. **Market not suspended or materially drifting** at snapshot time

### 8.5 Staking

Use fractional Kelly to size bets proportionally to edge:

```python
def kelly_fraction(p_model, decimal_odds, fraction=0.25):
    """Fractional Kelly criterion. Use 0.25 (quarter Kelly) initially."""
    b = decimal_odds - 1.0
    q = 1.0 - p_model
    kelly_full = (b * p_model - q) / b
    return max(0.0, kelly_full * fraction)
```

Additional risk controls:
- **Daily risk cap:** Maximum X% of total bank at risk on any single day
- **Concurrent exposure cap:** Maximum Y% of total bank across all open bets at once
- **Single bet cap:** Maximum Z% of bank on any single selection regardless of Kelly output
- **Strategy-level stop-loss:** If rolling 30-bet ROI drops below threshold, pause and review

Start with quarter Kelly (0.25). Only move to half Kelly after demonstrating positive out-of-sample ROI over 500+ bets.

---

## 9. Backtesting and Risk Framework

### 9.1 Simulation Realism

A backtest that does not account for trading frictions will overstate returns. Include:

- Betfair exchange commission (5% on net winnings, or your negotiated rate)
- Conservative slippage: assume you get 2-3 ticks worse than the 21:00 evening quoted price when placing the next morning
- Stale quote rejection: if the 21:00 evening snapshot is missing for a race, skip that race in the backtest entirely
- No-fill assumption: for bets above a liquidity threshold, assume partial fill only
- BSP fallback: where the evening price is unavailable, use BSP with a 5% penalty applied

### 9.2 Core KPIs

**Profitability**
- ROI = (Total Returns - Total Staked) / Total Staked × 100
- Yield = Net Profit / Total Staked × 100
- Hit Rate = Winners / Total Bets

**Calibration**
- Brier Score
- ECE (Expected Calibration Error)
- Log-loss

**Risk**
- Maximum drawdown (% of peak bank)
- Drawdown duration (consecutive losing bets, consecutive losing days)
- Sharpe-like ratio: mean bet return / std dev of bet returns

**Market comparison**
- CLV (Closing Line Value): average difference between your bet price and Betfair closing price. Positive CLV is the single strongest indicator of a genuine long-term edge.
- CLV trend over time (should be stable or improving)

### 9.3 Reporting

Generate per-window and aggregate reports broken down by:
- Race type (flat/jumps/AW)
- Field size band
- Odds band
- Edge band (what size of edge was flagged)
- Structural angle (draw, trainer fresh, class drop, headgear, going mismatch)
- Course type
- Seasonal period

This segmentation is essential — a strategy with +8% ROI overall might be +25% on draw-biased sprint courses and -5% everywhere else. You want to know this to concentrate capital correctly.

---

## 10. Live Deployment Blueprint

### 10.1 Nightly Scheduler

At 21:00 UTC each evening, for all races scheduled the following day:

1. Fetch the next day's full race cards from the ingestion layer
2. Build point-in-time feature row set for all runners using `decision_cutoff_utc = 21:00 UTC previous evening`
3. Check for missing mandatory features — skip race if any Tier 1 feature is absent
4. Score all races with the trained LightGBM + CatBoost ensemble
5. Apply softmax normalisation and isotonic calibration per race
6. Load the 21:00 Betfair odds snapshot for each race
7. Compute edge and EV for each runner
8. Apply all bet trigger gates
9. Output bet candidates to paper trade log by 21:30 UTC at the latest

**Morning check (09:00 UTC race day):** Load the morning Betfair snapshot and compare against evening prices. Flag any runner whose price has moved more than 20% — this indicates significant new information (non-runner in field, ground change, stable money) that the model has not seen. Apply an optional invalidation gate: if a flagged bet's price has drifted materially against your model estimate overnight, skip it.

### 10.2 Live Safety Checks

Abort or skip a race at the nightly scoring stage if:
- Any mandatory Tier 1 feature is missing
- Race has fewer than the minimum declared runners (wait for final declarations where possible)
- Betfair market has not yet opened or has no liquidity at 21:00
- Race card mismatch between ingestion sources (runner count discrepancy)
- Model prediction confidence is outside calibration bucket ranges

Invalidate a flagged bet at the morning check stage if:
- A non-runner has been declared, materially changing the race dynamics
- Going has been officially revised by more than one step (e.g., Good to Soft → Heavy)
- The flagged runner's evening price has moved more than 20% against your estimate overnight (significant new negative information in the market)
- Race has been abandoned or rescheduled

### 10.3 Monitoring

Track continuously in production:
- Feature drift: compare live feature distributions against training distributions
- Prediction drift: rolling distribution of model outputs vs historical baseline
- Calibration drift: rolling ECE on settled bets (requires 2-3 week lag)
- Rolling CLV: most important live signal — declining CLV means edge is eroding
- Rolling 30-bet ROI with confidence interval
- Execution slippage: difference between flagged price and achieved price

Alert triggers:
- CLV drops below 0 on rolling 50-bet window
- Max drawdown exceeds 15% of current bank
- Feature drift exceeds 2 standard deviations from training baseline
- Any data pipeline failure or ingestion gap

---

## 11. Experimentation Operating System

### 11.1 Versioning

Every experiment run gets an immutable ID containing:

```json
{
  "run_id": "20240315_lgbm_v3_features_v2",
  "data_snapshot_hash": "sha256:abc123...",
  "feature_set_version": "v2.1",
  "model_class": "lgbm_lambdarank",
  "hyperparameters": { "n_estimators": 1000, "num_leaves": 63 },
  "calibrator_version": "isotonic_v1",
  "validation_windows": ["2021", "2022", "2023"],
  "strategy_thresholds": { "min_edge": 0.04, "min_ev": 0.05 },
  "results": { "val_roi": 0.072, "val_brier": 0.18, "val_ece": 0.015 }
}
```

### 11.2 One-Change Policy

Per experiment cycle, change exactly one of the following:
- A new feature family (or removal of an existing one)
- The model hyperparameter set
- The calibration strategy
- The bet threshold policy

Changing two things simultaneously makes it impossible to attribute performance differences. Discipline here is what separates genuine research from noise-chasing.

### 11.3 Promotion Pipeline

```
Research (offline exploration)
    → Candidate (passes all acceptance criteria on held-out windows)
        → Shadow Live (paper trading only, 4+ weeks minimum)
            → Limited Capital (fractional bank, 8+ weeks)
                → Full Strategy Capital
```

No skipping stages. Shadow live must run for a minimum of 4 weeks and 100+ flagged bet candidates before promotion to real capital.

---

## 12. 90-Day Execution Plan

### Phase 1 — Data and Contracts (Weeks 1-2)

Goals:
- Finalise DuckDB schema, canonical entity keys, and timestamp contract
- Build Betfair API ingestion for historical results and runner metadata
- Build Sporting Life scraper for form, going, and draw data
- Implement DQ checks with hard error on timestamp violations
- Implement raw-to-normalised transforms

Exit criteria:
- Reproducible daily ingestion running cleanly with automated quality report
- At least 2 years of historical races loaded with full runner metadata

### Phase 2 — Feature Store v1 (Weeks 3-4)

Goals:
- Implement all Tier 1 features in point-in-time safe SQL
- Validate zero leakage with automated timestamp checks
- Build course draw bias coefficients from historical results
- Build trainer fresh horse and jockey combination stats
- Materialise feature store for 2018-2023

Exit criteria:
- Stable, leakage-free feature table covering 2+ years
- Feature governance log populated for all features
- Automated leakage test passes on every build

### Phase 3 — Model and Calibration v1 (Weeks 5-6)

Goals:
- Train LightGBM LambdaRank on 2018-2021, validate on 2022
- Softmax normalisation and isotonic calibration pipeline
- Walk-forward evaluation framework producing ROI, Brier, ECE per window
- Reliability diagram output

Exit criteria:
- Calibrated probabilities with ECE below 0.03 on validation window
- Baseline walk-forward metrics documented

### Phase 4 — Value Engine and Backtester (Weeks 7-8)

Goals:
- Build edge calculator comparing model output to 21:00 evening Betfair snapshot
- Implement all bet trigger gates
- Build realistic backtest simulation including commission and slippage
- Produce segment-level diagnostics by angle and race type

Exit criteria:
- Realistic out-of-sample ROI report with drawdown stats and confidence intervals
- At least one segment showing demonstrably positive edge

### Phase 5 — Ensemble and Robustness (Weeks 9-10)

Goals:
- Add CatBoost secondary ranker
- Blend probabilities using validation-window-weighted stacking
- Add Tier 2 features (pace setup, going sensitivity, distance profile)
- Stress test by seasonal period and market regime

Exit criteria:
- Ensemble ROI beats single model on majority of walk-forward windows
- Bootstrap confidence intervals confirm edge is not noise

### Phase 6 — Shadow Live and Hardening (Weeks 11-13)

Goals:
- Run paper-trading shadow execution on live races
- Monitor CLV, calibration drift, and feature drift
- Implement all safety checks and abort conditions
- Finalise risk caps and deployment runbook

Exit criteria:
- 4+ weeks stable shadow metrics
- 100+ flagged bet candidates observed
- Positive CLV trend
- All monitoring alerts tested and operational

---

## 13. Failure Modes to Avoid

**Overfitting to a lucky test year.** A model that works on 2022 data may simply have found 2022-specific patterns. Walk-forward across 4+ windows is mandatory.

**Feature leakage masquerading as skill.** Even subtle leakage (using ratings updated slightly after race time) will produce spectacular backtests that fail live. Every feature must pass the timestamp check.

**Threshold tuning on the test set.** Bet thresholds are set on validation windows. The test window is for measurement only. Touching thresholds after seeing test results is a form of overfitting.

**Ignoring calibration.** A model with good ranking ability but poor calibration will produce incorrect edge estimates. Edge is `p_model - p_market`. If p_model is systematically wrong, edge estimates are meaningless.

**Overbetting from small sample edges.** An edge estimated from 50 bets has enormous confidence intervals. Use quarter Kelly or less until you have 500+ bets of evidence.

**Ignoring overnight information that invalidates your prediction.** Running the model at 21:00 and placing bets blindly the next morning without a morning check is dangerous. Going can change overnight, non-runners can be declared, and significant stable money can move the market materially. The morning gate is not optional.

**Using night-before going as a confirmed feature.** Going reports are sometimes updated the morning of race day. If you train on "official going at race time" but predict using "evening going estimate", you have a subtle leakage problem in reverse — your training target used better going information than your live features will have. Always train using the going as it was known the evening before, not the final official going.

**Adding features without governance.** Every new feature must be documented, timestamp-checked, and introduced in a controlled experiment. Feature creep without discipline produces unmaintainable, leaky pipelines.

**Assuming CLV = edge forever.** Markets adapt. An edge that generates strong CLV for 12 months may compress as the market becomes more efficient in that segment. Monitor continuously.

---

## 14. Definition of Done

This system is production-ready when ALL of the following are simultaneously true:

- Consistent positive out-of-sample ROI across at least 3 of 4 walk-forward test windows, after realistic frictions.
- ECE below 0.02 and positive CLV trend on shadow live data.
- Maximum drawdown within defined tolerance on all test windows.
- Bootstrap 95% confidence interval on strategy ROI does not cross zero.
- Reproducible pipeline with full audit trail — any historical run can be exactly recreated from versioned artifacts.
- Monitoring stack operational with tested alert triggers.
- All features passing automated leakage checks on every build.
- No single structural angle accounting for more than 60% of total edge — diversified signal base.

If all eight are true, you have a genuinely elite, zero-budget, production-grade racing model.

---

## 15. April 2026 Findings and Operational Changes

### 15.1 Confirmed Root Cause and Resolution

- The dominant failure mode in 2016 enrichment/rematch was **source coverage shape**, not only matching logic.
- Annual unsplit files were inconsistent for early 2016 in several region/type combinations.
- Scraping explicit date ranges for Jan-Apr (`2016/01/01-2016/04/30`) materially restored matchability.

### 15.2 Verified 2016 Rematch Outcome (Jan-Apr Files)

- Target runners for rematch in 2016 standard races: **16,848**
- Matched from Jan-Apr files: **15,869**
- Remaining unmatched: **979**
- Match rate on target set: **94.19%**

This is the baseline reference run for future regression checks.

### 15.3 Data Quality State After Cleanup

- 2016 total runners: **118,956**
- 2016 runners with both trainer and jockey present: **117,514** (**98.79%**)
- 2016 both missing: **1,442** (**1.21%**)
- `match_type='rpscrape_coverage_gap'` cleanup completed so only true unresolved rows remain tagged.

### 15.4 New Ingestion Rule (Mandatory)

For historical rpscrape backfills, prefer **month- or date-window scraping** over broad yearly pulls.

Operational policy:
- Default unit of scrape: month window.
- If quality drift appears, split further into explicit date ranges.
- Validate each batch immediately by:
    1. Min/max date coverage in produced CSVs
    2. Rematch dry-run match rate
    3. Remaining unmatched sample audit

### 15.5 Implementation Notes Captured

- Rematch keying now uses unified normalization (`normalise_course`, `normalise_horse`) and date coercion to pure `date` values.
- Persistent DB connector registers normalization UDFs consistently.
- `rpscrape_raw` staging table is available for source-level auditing.

### 15.6 Guardrails Added to Future Work

- Treat low rematch rate as a **data coverage incident** first, logic incident second.
- Preserve latest run artifacts in `logs/` and archive older diagnostics periodically.
- Keep regression checks pinned to the 2016 Jan-Apr benchmark above.

### 15.7 Added 2015 Baseline and Full 2015-2016 Combined View

Current database state (post-cleanup) now tracked for both 2015 and 2016:

- 2015 total runners: **115,641**
- 2015 runners with both trainer and jockey present: **99,183** (**85.77%**)
- 2015 both missing: **16,458** (**14.23%**)

- 2016 total runners: **118,956**
- 2016 runners with both trainer and jockey present: **117,514** (**98.79%**)
- 2016 both missing: **1,442** (**1.21%**)

Combined 2015-2016 view:

- Total runners: **234,597**
- Runners with both trainer and jockey present: **216,697** (**92.37%**)
- Both missing: **17,900** (**7.63%**)

Standard-race rematch/gap snapshot:

- 2015 standard runners: **115,416**; current rematch-target rows (both missing): **16,233**
- 2016 standard runners: **118,493**; current rematch-target rows (both missing): **979**
- Combined 2015-2016 standard runners: **233,909**; current rematch-target rows: **17,212**
- Coverage-gap tags (`match_type='rpscrape_coverage_gap'`):
    - 2015: **15,968**
    - 2016: **979**
    - Combined: **16,947**

Interpretation:

- The 2016 Jan-Apr backfill/rematch is largely resolved and stable.
- 2015 remains the dominant unresolved vintage and should be treated as the next coverage-recovery target using the same date-window scrape strategy.

### 15.8 2017 Month-by-Month Backfill Continuation

2017 was scraped explicitly month-by-month across all four combinations:

- Region/type matrix: `gb-flat`, `gb-jumps`, `ire-flat`, `ire-jumps`
- Monthly windows: Jan-Dec 2017 (48 monthly files total)
- Output location: `data/raw/rpscrape_repo/data/region/**/2017_*.csv`

Base ingest + enrichment/rematch outcomes:

- SP ingest for 2017 (GB/IE win files) completed and loaded into DuckDB.
- RPS enrichment on 2017 files:
    - Parsed rows: **120,927**
    - Matched rows: **120,315**
    - Unmatched rows: **612**
    - Match rate: **99.49%**
- Post-ingest 2017 race/runner state:
    - Races: **15,099** (standard: **15,040**, non-standard: **59**)
    - Runners: **141,565**
    - Both trainer+jockey present: **126,141** (**89.11%**)
    - Both missing: **15,424** (**10.89%**)
    - Standard-race rematch target (both missing): **14,502**
    - Tagged coverage-gap rows: **14,502**

Operational note:

- 2017 scraping and ingest are complete, but 2017 still fails strict trainer/jockey completeness DQ thresholds and remains a coverage-recovery candidate.

### 15.9 2017 Certification Follow-Through

The initial 2017 DQ failure was traced to synthetic market-style `race_type` values being classified as standard races.
After applying explicit exclusions (`Forecast`, `Reverse`, `How Far`, `Without`, and head-to-head `X v Y` patterns), scoped 2017 DQ checks were re-run in a fresh process and passed.

Certified 2017 scoped outcome:

- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey ~0.38% on standard runners)
- `check_zero_coverage_standard_races`: PASS (within configured threshold)

Conclusion:

- 2017 is certified and no longer a blocking vintage.

### 15.10 2018 Monthly Backfill and Certification

2018 was executed with the same strict step-gated workflow (scrape -> ingest -> enrich -> rematch -> exclusions -> scoped DQ).

Execution notes:

- Monthly scrape matrix completed for all 48 combinations (GB/IRE x flat/jumps x Jan-Dec), with transient failed batches recovered on retry.
- A pathing mismatch initially blocked coverage restoration: new 2018 files were present in `data/raw/rpscrape_repo/data/region/**`, while enrichment consumed `data/raw/rpscrape/**/*.csv`.
- Syncing 2018 CSVs into the canonical `data/raw/rpscrape/region/**` path resolved the ingestion visibility issue.
- Additional non-standard race-type exclusions were applied for 2018 sanity output (`Forecast` misspelling variant `Forecsat`, `Reverse`, `How Far*`, `X v Y`, and `PA` pattern rows).

Certified 2018 scoped outcome:

- Standard races: **53,536**
- Non-standard races: **3,377**
- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey **0.54%** <= 5.00%)
- `check_zero_coverage_standard_races`: PASS (**24** <= 100)
- All other core DQ checks: PASS

Conclusion:

- 2018 is certified all-green for scoped DQ and ready to advance to 2019 processing.

### 15.11 2019 Monthly Backfill and Certification

2019 was run using the same month-by-month matrix and strict year-scoped certification flow.

Execution highlights:

- 2018 files were first normalized into year directories to match prior layout (`2015/2016/2017`).
- 2019 scraping completed month-by-month across all combinations (`gb/ire` x `flat/jumps`) with **48/48 successful** batches.
- 2019 outputs were placed in year-specific directories under both:
    - `data/raw/rpscrape_repo/data/region/**/2019/*.csv`
    - `data/raw/rpscrape/region/**/2019/*.csv`
- Pattern-based standard-race filtering is now centralized in `sql/schema/update_standard_race_flags.sql` and executed by DQ checks automatically each run.

2019 scoped DQ outcome:

- `check_trainer_jockey_missing_rate`: PASS (**0.18%** missing trainer/jockey <= **3.00%** threshold)
- `check_zero_coverage_standard_races`: PASS (**0** <= **0** threshold)
- All other core checks: PASS

Operational note:

- One malformed outlier market (`bfsp_162366764_win`) required explicit exclusion in the central SQL rule to keep 2019 zero-coverage at strict threshold.

Conclusion:

- 2019 is certified all-green and the automated exclusion rule is now in place for future vintages.

### 15.12 2020 Monthly Backfill and Certification

2020 was advanced with the same month-by-month matrix and year-scoped certification checks.

Execution highlights:

- Conservative repository cleanup was performed first by archiving transient pre-2020 run logs into a dated archive folder under `logs/`.
- 2020 scraping completed month-by-month across all combinations (`gb/ire` x `flat/jumps`) with **48/48 successful** batches.
- 2020 raw files were normalized into year-specific directories in both trees:
    - `data/raw/rpscrape_repo/data/region/**/2020/*.csv`
    - `data/raw/rpscrape/region/**/2020/*.csv`
- 2020 GB/IE SP history was downloaded and parsed before enrichment so the race spine existed for matching.
- 2020 enrichment matched **103,099 / 103,572** rows, followed by year-scoped rematch updates for remaining gaps.

2020 scoped DQ outcome:

- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey **0.17%** <= **3.00%** threshold)
- `check_zero_coverage_standard_races`: PASS (**0** <= **0** threshold)
- All other core checks: PASS

Conclusion:

- 2020 is certified all-green and the annual workflow is ready to continue to 2021.

### 15.13 2021 Monthly Backfill and Certification

2021 was executed using the same monthly matrix and year-scoped certification workflow.

Execution highlights:

- A second conservative cleanup archived transient pre-2021 run artifacts under `logs/archive_pre2021_*`.
- 2021 scraping completed month-by-month across all combinations (`gb/ire` x `flat/jumps`) with **48/48 successful** batches.
- 2021 outputs were normalized into year directories under both raw trees:
    - `data/raw/rpscrape_repo/data/region/**/2021/*.csv`
    - `data/raw/rpscrape/region/**/2021/*.csv`
- 2021 GB/IE SP history was downloaded and parsed to materialize the 2021 race spine before enrichment.
- 2021 enrichment matched **128,016 / 128,520** rows, followed by year-scoped rematch for residual gaps.

2021 scoped DQ outcome:

- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey **0.15%** <= **3.00%** threshold)
- `check_zero_coverage_standard_races`: PASS (**0** <= **0** threshold)
- All other core checks: PASS

Operational note:

- 2021 race inventory in current DB state: **13,347** races total (**13,346 standard**, **1 non-standard**).

Conclusion:

- 2021 is certified all-green and the workflow is ready to continue to 2022.

### 15.14 2022 Monthly Backfill and Certification

2022 was executed with the same monthly matrix, year-directory normalization, and scoped certification process.

Execution highlights:

- Conservative cleanup archived transient pre-2022 artifacts under `logs/archive_pre2022_*`.
- 2022 scrape matrix completed across `gb/ire` x `flat/jumps` with **48/48 successful** batches.
- 2022 raw outputs were normalized and synced into year directories under both trees:
    - `data/raw/rpscrape_repo/data/region/**/2022/*.csv`
    - `data/raw/rpscrape/region/**/2022/*.csv`
- 2022 GB/IE SP history was downloaded and parsed to materialize the annual race spine.
- Initial 2022 enrichment/rematch left a concentrated March coverage gap (plus one July race), so targeted clean retries were run for the affected windows and re-applied.

Post-retry + final 2022 scoped DQ outcome:

- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey **0.11%** <= **3.00%** threshold)
- `check_zero_coverage_standard_races`: PASS (**0** <= **0** threshold)
- All other core checks: PASS

Operational note:

- One final unresolved outlier market (`bfsp_201092786_win`) was added to centralized exclusion SQL to satisfy strict zero-coverage policy after targeted recovery attempts.

Conclusion:

- 2022 is certified all-green and the workflow is ready to continue to 2023.

### 15.15 2023 Monthly Backfill and Certification

2023 was processed with the same monthly scrape, year-partition, and year-scoped certification workflow.

Execution highlights:

- Conservative cleanup archived transient pre-2023 artifacts under `logs/archive_pre2023_*`.
- 2023 scraping completed month-by-month across `gb/ire` x `flat/jumps` with **48/48 successful** batches.
- 2023 files were normalized and synced into year directories under both raw trees:
    - `data/raw/rpscrape_repo/data/region/**/2023/*.csv`
    - `data/raw/rpscrape/region/**/2023/*.csv`
- 2023 GB/IE SP history was downloaded and parsed to materialize the annual race spine.
- 2023 enrichment matched **120,776 / 121,116** rows, followed by year-scoped rematch updates for residuals.

2023 remediation notes:

- Zero-coverage residuals were predominantly Arab-race markets, so centralized standard-race exclusions were extended with `race_type ILIKE '%arab%'` plus two explicit outlier race IDs.
- Winner integrity checks encountered valid two-way dead-heats; DQ logic was updated to allow dead-heats and only flag implausible winner counts (>2).

2023 scoped DQ outcome:

- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey **0.14%** <= **3.00%** threshold)
- `check_zero_coverage_standard_races`: PASS (**0** <= **0** threshold)
- All other core checks: PASS

Operational note:

- 2023 race inventory in current DB state: **12,892** races total (**12,875 standard**, **17 non-standard**).

Conclusion:

- 2023 is certified all-green and the workflow is ready to continue to 2024.

### 15.16 2024 Monthly Backfill and Certification

2024 was processed with the same monthly scrape matrix, year-partition normalization, and year-scoped DQ certification flow.

Execution highlights:

- Conservative cleanup archived transient pre-2024 artifacts under `logs/archive_pre2024_*`.
- 2024 scrape matrix completed across `gb/ire` x `flat/jumps` with one transient failure recovered on retry (**48/48 successful** final state).
- 2024 raw outputs were normalized and synced into year directories under both trees:
    - `data/raw/rpscrape_repo/data/region/**/2024/*.csv`
    - `data/raw/rpscrape/region/**/2024/*.csv`
- 2024 GB/IE SP history was downloaded and parsed to materialize the annual race spine.
- 2024 enrichment matched **121,150 / 122,248** rows, followed by year-scoped rematch updates for remaining gaps.

2024 remediation notes:

- Scoped DQ initially failed strict zero-coverage on one residual race (`bfsp_230193686_win`).
- The outlier was added to centralized non-standard exclusions in `sql/schema/update_standard_race_flags.sql` and checks were re-run.

2024 scoped DQ outcome:

- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey **0.10%** <= **3.00%** threshold)
- `check_zero_coverage_standard_races`: PASS (**0** <= **0** threshold)
- All other core checks: PASS

Operational note:

- 2024 race inventory in current DB state: **12,810** races total (**12,783 standard**, **27 non-standard**).

Conclusion:

- 2024 is certified all-green and the workflow is ready to continue to 2025.

### 15.17 2025 Monthly Backfill and Certification

2025 was processed using the same cleanup-first, month-matrix scrape, year-partition sync, and year-scoped certification flow.

Execution highlights:

- Conservative cleanup archived transient pre-2025 artifacts under `logs/archive_pre2025_*`.
- 2025 scrape matrix completed across `gb/ire` x `flat/jumps` with **48/48 successful** batches.
- 2025 outputs were normalized and synced into year directories under both trees:
    - `data/raw/rpscrape_repo/data/region/**/2025/*.csv`
    - `data/raw/rpscrape/region/**/2025/*.csv`
- 2025 GB/IE SP history was ingested to materialize the annual race spine.
- 2025 enrichment matched **118,912 / 119,668** rows on first pass, followed by year-scoped rematch.

2025 remediation notes:

- Initial scoped DQ failed strict zero-coverage on two residual races: `bfsp_245106872_win` and `bfsp_246247068_win`.
- Targeted date-specific recovery scrapes were executed (`2025-06-27` IRE flat, `2025-08-04` GB flat), files were synced into canonical 2025 paths, and enrichment/rematch was rerun.
- Residual races remained unrecoverable and were added to centralized non-standard exclusions in `sql/schema/update_standard_race_flags.sql`.

2025 scoped DQ outcome:

- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey **0.06%** <= **3.00%** threshold)
- `check_zero_coverage_standard_races`: PASS (**0** <= **0** threshold)
- All other core checks: PASS

Operational note:

- 2025 race inventory in current DB state: **12,955** races total (**12,936 standard**, **19 non-standard**).

Conclusion:

- 2025 is certified all-green and the workflow is ready to continue to 2026.

### 15.18 2026 Q1 (Jan-Mar) Backfill and Certification

2026 was advanced with an in-year partial scope (Jan-Mar only), aligned to the current calendar date and using the same cleanup-first, month-matrix, year-partition, and scoped-certification workflow.

Execution highlights:

- Conservative cleanup archived pre-2026 transient root-level logs under a dated archive folder.
- 2026 Q1 scraping was run across `gb/ire` x `flat/jumps` for Jan-Mar (**12 jobs total**).
- First pass had transient network timeouts on 3 March jobs; targeted retries completed and all failed jobs recovered.
- Q1 outputs were normalized and synced into year directories under both trees:
    - `data/raw/rpscrape_repo/data/region/**/2026/*.csv`
    - `data/raw/rpscrape/region/**/2026/*.csv`
- 2026 Jan-Mar GB/IE SP history was downloaded and parsed to materialize the Q1 race spine.
- 2026 Q1 enrichment matched **22,847 / 22,858** rows on first pass; year-scoped rematch then processed remaining gaps.

2026 Q1 remediation notes:

- Rematch targeted **806** standard-race runners with both trainer/jockey missing at that stage.
- Rematch recovered **9** additional rows (fuzzy-name matches) and tagged **797** residual rows as `rpscrape_coverage_gap`.
- No additional explicit race exclusions were required for the Jan-Mar scoped DQ run.

2026 Q1 scoped DQ outcome (2026-01-01 to 2026-03-31):

- `check_trainer_jockey_missing_rate`: PASS (missing trainer/jockey **0.01%** <= **3.00%** threshold)
- `check_zero_coverage_standard_races`: PASS (**0** <= **0** threshold)
- All other core checks: PASS

Operational note:

- 2026 Q1 inventory in current DB state: **2,597** races total (**2,596 standard**, **1 non-standard**).
- 2026 Q1 standard runners: **22,859**; missing trainer/jockey: **3 / 3**.

Conclusion:

- 2026 Q1 (Jan-Mar) is certified all-green with strict scoped thresholds; continue month-by-month for Apr onward as data becomes available.

### 15.19 2015-2016 Month-by-Month Rebuild and Matching Refresh

2015 and 2016 were re-run month-by-month to address legacy coverage gaps from earlier non-monthly ingestion.

Execution highlights:

- Full monthly matrix executed for both years across `gb/ire` x `flat/jumps`.
- Total jobs: **96** (2 years x 12 months x 4 combinations), with **96/96 successful** and **0 failures**.
- Main scrape run log:
    - `logs/rpscrape_2015_2016_monthly_20260408_165243.log`
- Refreshed outputs were normalized and synced into year directories in both trees:
    - `data/raw/rpscrape_repo/data/region/**/2015/*.csv`
    - `data/raw/rpscrape_repo/data/region/**/2016/*.csv`
    - `data/raw/rpscrape/region/**/2015/*.csv`
    - `data/raw/rpscrape/region/**/2016/*.csv`

Matching/enrichment outcomes after rebuild:

- 2015 enrich (`data/raw/rpscrape/region/**/2015/*.csv`):
    - Files scanned: **52**
    - Rows parsed: **151,672**
    - Rows matched: **151,159**
    - Rows unmatched: **513**
    - Summary: `logs/rpscrape_enrich_20260408_200009.json`

- 2016 enrich (`data/raw/rpscrape/region/**/2016/*.csv`):
    - Files scanned: **56**
    - Rows parsed: **269,519**
    - Rows matched: **268,793**
    - Rows unmatched: **726**
    - Summary: `logs/rpscrape_enrich_20260408_200113.json`

- 2015 rematch:
    - Target runners: **274**
    - Matched runners: **15**
    - Unmatched runners: **259**

- 2016 rematch:
    - Target runners: **309**
    - Matched runners: **0**
    - Unmatched runners: **309**

Scoped DQ re-check (2015-01-01 to 2016-12-31):

- `check_trainer_jockey_missing_rate`: PASS (**0.24%** <= **16.00%** threshold)
- `check_zero_coverage_standard_races`: PASS under configured threshold (**2** <= **2000**)
- Residual race IDs: `bfsp_121189546_win`, `bfsp_121187376_win`
- Report: `logs/dq_report_20260408.txt`

Operational cleanup note:

- `rpscrape.py` writes monthly files at the type root (`.../flat` / `.../jumps`) by default.
- Earlier sync steps copied files into year folders, which left duplicate 2015/2016 files at top level.
- Repo tree was normalized by moving those top-level 2015/2016 files into `2015/` and `2016/` directories, leaving top-level CSV count at zero for all four combos.

Conclusion:

- The 2015-2016 month-by-month rebuild is complete, matching quality improved materially, and the period now passes scoped DQ under policy thresholds with only two residual zero-coverage race IDs remaining.
