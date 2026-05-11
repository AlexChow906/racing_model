# racing_model

Price-blind horse racing value model. Predicts win probabilities from fundamentals, compares to Betfair SP to find value bets. Separate flat and jumps models trained with LightGBM LambdaRank + isotonic calibration.

## Results (Walk-Forward Validation, 2022-2026)

Tuned params, edge>5% threshold, £1 level stakes:

| Model | Windows | Total Bets | ROI | Profit |
|-------|---------|-----------|-----|--------|
| Flat  | 4/4 profitable | 53,905 | +7.69% | £+4,144 |
| Jumps | 4/4 profitable | 39,153 | +11.58% | £+4,534 |
| **Combined** | **8/8 profitable** | **93,058** | **+9.33%** | **£+8,678** |

## Data Sources

| Source | What You Get | Access |
|---|---|---|
| Betfair SP History | Race spine: BSP prices, win/lose results | Public CSV at promo.betfair.com |
| rpscrape | Trainer, jockey, draw, weight, going, class, times, headgear | Open-source scraper for Racing Post |
| Betfair Exchange API | Live odds snapshots | Free account + API credentials |

## Project Layout

```text
configs/             Settings, course aliases, feature registry
sql/
  schema/            DuckDB table definitions
  features/          Feature SQL (001-009)
src/
  ingestion/         Data download, parsing, enrichment, rematch
  pipelines/         DB init, historical rebuild, feature store
  modeling/          Training, tuning, validation
models/
  default/           Default-param flat + jumps models
  tuned/             Optuna-tuned flat + jumps models
experiments/         Training metadata and validation results
data/raw/            Raw data (not committed)
racing.duckdb        Analytical database (not committed)
```

## Features (81 total, ~65-69 used per model)

**Horse Form**: weighted form score, place rates, finishing positions, form trend, improvement index, beaten lengths, speed figures, career stats, consistency

**Affinity**: going (exact + group), distance, course (venue-normalised)

**Collateral Form**: subsequent win/place rate of beaten opponents (franked form)

**Runner Profile**: weight vs field, age, official rating vs field, career runs

**Trainer**: 90-day win rate, course/going/distance specialisation, 14-day hot streak, fresh horse record

**Jockey**: 90-day win rate, course/distance specialisation, trainer combo, upgrade signal

**Race Context**: class, handicap flag, prize money, pace pressure, draw bias

**Jumps-specific**: completion rate, fall rate, pulled-up rate, recent non-completions

## Quick Start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Initialize database
python -m src.pipelines.init_db

# Build feature store (requires populated DB)
python -m src.pipelines.run_phase2_feature_store

# Train split flat/jumps models
python -m src.modeling.train_split

# Walk-forward validation (4 windows, 2022-2026)
python -m src.modeling.walk_forward_final

# Optuna hyperparameter tuning
python -m src.modeling.tune_optuna
```

## Historical Data Pipeline

1. Download Betfair SP history (public, no auth):
```bash
python -m src.ingestion.betfair_historical \
    --use-sp-history \
    --sp-include pricesukwin,pricesirewin \
    --start-year 2015 --start-month 1 \
    --end-year 2026 --end-month 5
```

2. Place rpscrape CSV exports under `data/raw/rpscrape/`, then enrich:
```bash
python -m src.ingestion.rpscrape_enrich --input-glob "data/raw/rpscrape/**/*.csv"
python -m src.ingestion.rematch --input-glob "data/raw/rpscrape/**/*.csv"
```

3. Build feature store and train:
```bash
python -m src.pipelines.run_phase2_feature_store
python -m src.modeling.train_split
```

## Value Betting Approach

The model is price-blind: it never sees current race odds. It predicts P(win) from fundamentals, then compares to Betfair SP post-race to identify value:

1. Model outputs calibrated P(win) per runner
2. Convert to implied odds: `model_odds = 1 / P(win)`
3. Compare to BSP: `edge = P(win) - (1 / BSP)`
4. Bet when `edge > 5%`

## Betfair Setup

1. Create a Betfair Exchange account
2. Generate API SSL certificates (`.crt` and `.key`)
3. Copy `.env.example` to `.env` and fill in credentials + cert paths
