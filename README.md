# racing_model

Price-blind horse racing value model. Predicts win probabilities from fundamentals, compares to Betfair SP to find value bets. Flat uses CatBoost YetiRank (66 features, no calibration). Chase and Hurdle use separate LightGBM LambdaRank + isotonic calibration models (59 features each).

## Results (Walk-Forward Validation, 2022-2026, edge>15%)

| Model | Bets | ROI | P&L |
|-------|------|-----|-----|
| Flat  | 1,875 | +45.5% | +853u |
| Chase | 519 | +88.5% | +459u |
| Hurdle | 1,276 | +35.0% | +447u |

All windows positive. Jumps is split into separate chase and hurdle models with independently tuned hyperparameters (walk-forward Optuna).

## Quick Start

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Enrich database with rpscrape data (place CSVs in data/raw/rpscrape/)
python -m src.ingestion.rpscrape_enrich

# Build feature store
python -m src.pipelines.run_phase2_feature_store

# Train flat production model (saves to models/tuned/flat/model.cbm)
python -m src.modeling.train_split --flat-v2

# Walk-forward validation
python -m src.modeling.train_split --flat-v2 --walk-forward

# Daily predictions (requires Betfair API credentials)
python -m src.pipelines.daily_predictions --date tomorrow
python -m src.pipelines.daily_predictions --date tomorrow --flat
python -m src.pipelines.daily_predictions --date tomorrow --jumps
python -m src.pipelines.daily_predictions --date tomorrow --min-edge 0.12
```

## Data Sources

| Source | What You Get | Access |
|---|---|---|
| Betfair SP History | Race spine: BSP prices, win/lose results | Public CSV at promo.betfair.com |
| rpscrape | Trainer, jockey, draw, weight, going, class, times, headgear, RPR, sex, non-completions | Open-source scraper for Racing Post |
| Betfair Exchange API | Live odds for daily predictions | Free account + API credentials |

## Features

**Flat model (66 features, CatBoost YetiRank):**
- Horse form: weighted form, form trend, place rates, speed figures, RPR, beaten lengths
- Class: class delta, prize money, handicap flag
- Ratings/weight: official rating vs field, weight vs field, weight change
- Race context: field size, distance, pace pressure, race class, going
- Draw: position, field percentile, course+going bias coefficient
- Connections: trainer/jockey win rates (90d, course, distance, going, combo)
- Collateral form: subsequent win/place rate of beaten opponents
- Horse sex: sex encoded, is female

**Chase model (59 features, LightGBM + isotonic calibration):**
- Same as flat minus draw features, plus:
- Non-completion: pulled-up rate, recent non-completions (F/PU/UR/BD)
- Trained on chase races only — learns fence-jumping and fall-risk patterns

**Hurdle model (59 features, LightGBM + isotonic calibration):**
- Same feature set as chase, trained on hurdle + NH Flat races only
- Both models use Optuna-tuned hyperparameters (walk-forward optimised)

## Project Layout

```text
sql/
  schema/            DuckDB table definitions
  features/          Feature SQL (001-009)
src/
  constants/         Features, params, walk-forward windows
  ingestion/         Data download, parsing, enrichment
  pipelines/         DB init, feature store, daily predictions, backtest
  modeling/          Training, tuning, validation
models/
  tuned/flat/        Production CatBoost flat model (.cbm)
  tuned/chase/       Production LightGBM chase model (.lgbm)
  tuned/hurdle/      Production LightGBM hurdle model (.lgbm)
experiments/         Training metadata and validation results
data/raw/            Raw rpscrape CSVs (not committed)
racing.duckdb        Analytical database (not committed)
```

## Daily Pipeline (Live Testing)

Run the full daily pipeline (collect results, update P&L, score today's races):

```bash
./scripts/daily_run.sh
```

Or run each step individually:

```bash
# Collect yesterday's settled results (public SP CSVs, no auth needed)
python -m src.pipelines.collect_results --date yesterday

# Score today's races (requires Betfair API credentials)
python -m src.pipelines.daily_predictions --date today

# View P&L
python -m src.pipelines.track_pnl                              # all-time summary
python -m src.pipelines.track_pnl --from 2026-05-01 --to 2026-05-07  # date range
python -m src.pipelines.track_pnl --date 2026-05-15            # single day (bet details)
```

Automate with cron (runs daily at 9am):

```
0 9 * * * cd /path/to/racing_model && ./scripts/daily_run.sh >> logs/daily_run.log 2>&1
```

Value bets are logged to `logs/daily_bets.csv`. P&L summary is saved to `logs/pnl_tracker.csv`.

## Historical Data Pipeline

1. Download Betfair SP history:
```bash
python -m src.ingestion.betfair_historical \
    --use-sp-history \
    --sp-include pricesukwin,pricesirewin \
    --start-year 2015 --start-month 1 \
    --end-year 2026 --end-month 5
```

2. Enrich with rpscrape data (scrape CSVs live in `data/raw/rpscrape_repo/data/region/`):
```bash
python -m src.ingestion.rpscrape_enrich
```

3. Build feature store and train:
```bash
python -m src.pipelines.run_phase2_feature_store
python -m src.modeling.train_split --flat-v2
```

## Retraining Models

Run weekly or after adding significant new data. The retrain script validates results before updating production models.

```bash
# Step 1: Run walk-forward validation on latest data
./scripts/retrain.sh

# Step 2: Review results. If happy, train and save production models
./scripts/retrain.sh --approve
```

This will:
1. Rebuild the feature store on `racing.duckdb` (latest data)
2. Run walk-forward for flat and jumps (verify ROI hasn't degraded)
3. On `--approve`: snapshot `racing.duckdb` as the new `racing_backtest.duckdb`, then train production models

The old `racing_backtest.duckdb` is preserved until you approve.

## Value Betting Approach

The model is price-blind: it never sees current race odds. It predicts P(win) from fundamentals using a ranking model + softmax, then compares to Betfair SP post-race:

1. Model outputs P(win) per runner via race-level softmax
2. Compare to BSP: `edge = P(win) - (1 / BSP)`
3. Bet when `edge > 15%` (configurable with `--min-edge`)

No isotonic calibration for flat (raw softmax). Isotonic calibration for jumps.

## Betfair Setup

1. Create a Betfair Exchange account
2. Generate API SSL certificates (`.crt` and `.key`)
3. Copy `.env.example` to `.env` and fill in credentials + cert paths
