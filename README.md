# racing_model

Race-level horse racing value model project with DuckDB + Parquet.

## Source Map

| Source | What You Get | Access |
|---|---|---|
| Betfair Exchange API | Live and historical odds, volume, price movement | Free Betfair account + API credentials |
| Sporting Life | Race cards, results, form, going, draws | Scrape only if terms allow |
| At The Races | Race cards, runners, trainer/jockey stats | Scrape only if terms allow |
| BHA | Official results, handicap ratings, going | Public structured data where available |
| Historical dumps (GitHub/Kaggle) | Bulk historical CSV results | Free datasets |

## Project Layout

This repository is set up with the layout you asked for:

```text
betfair_odds_raw/
	2022/
		01/
			race_123.parquet
			race_124.parquet

racing.duckdb
models/
	lgbm_v1.pkl
	calibrator_v1.pkl
logs/
	paper_trades.csv
```

Current scaffold folders/files:
- `betfair_odds_raw/2022/01/`
- `models/`
- `logs/`
- `sql/schema.sql`
- `configs/sources.yaml`
- `configs/strategy.yaml`
- `src/ingestion/betfair_ingest.py`
- `src/ingestion/scrape_cards.py`
- `src/pipelines/init_project.py`

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Initialize folders, DuckDB schema, and paper trade log:

```bash
python src/pipelines/init_project.py
```

This creates:
- `racing.duckdb`
- core tables from `sql/schema.sql`
- `logs/paper_trades.csv` with headers

## Notes on Odds Collection

- Prefer Betfair API as the primary legal odds source.
- Keep raw odds snapshots immutable in parquet under `betfair_odds_raw/`.
- If scraping, verify site terms and robots rules before collection.
- Keep timestamps in UTC and maintain a fixed decision cutoff (for example T-60).

## Betfair Setup (What You Need To Do)

1. Create and verify a Betfair Exchange account.
2. Log in to Betfair developer portal and create an app key.
3. Generate Betfair API SSL client certificate files (`.crt` and `.key`).
4. Make sure your account has API access permissions enabled.
5. Copy `.env.example` to `.env` and fill in credentials + cert paths.
6. Export env vars in your shell (or load from `.env`).

Run collection job:

```bash
python src/pipelines/collect_betfair_snapshots.py --hours-ahead 6
```

This writes:
- raw race snapshots to `betfair_odds_raw/YYYY/MM/race_<market_id>.parquet`
- normalized odds rows to table `odds_snapshots` in `racing.duckdb`

## Why Betfair API

Betfair is the best free core market source for this stack because:
- It is legal and structured access (more reliable than brittle scraping).
- You get exchange-level prices and traded volume, not just headline bookmaker odds.
- It supports timestamped snapshots required for strict no-leakage modeling.
- It is strong for CLV validation and market-efficiency features.
- You can still compare against bookmaker feeds later for value overlays.

## Historical Stack (Free)

Recommended order:
1. Betfair SP promo history for BSP and win/loss spine.
2. rpscrape CSV exports for trainer, jockey, draw, weight, going, class, and ratings.
3. Betfair live API for forward snapshots.

Download and parse SP history (GB/IE WIN files):

```bash
python -m src.ingestion.betfair_historical \
	--use-sp-history \
	--start-year 2018 --start-month 1 \
	--end-year 2025 --end-month 12
```

Place rpscrape CSV exports under `data/raw/rpscrape/`, then enrich canonical tables:

```bash
python -m src.ingestion.rpscrape_enrich --input-glob "data/raw/rpscrape/**/*.csv"
```

Optional: organize rpscrape files into `YYYY/MM` folders:

```bash
python -m src.ingestion.organize_rpscrape
```

This creates paths like `data/raw/rpscrape/by_year_month/2025/04/gb_2025_04_01.csv`.

Optional: include region in the folder layout (`region/YYYY/MM`):

```bash
python -m src.ingestion.organize_rpscrape \
	--dest-root data/raw/rpscrape/by_region_year_month \
	--include-region
```

This creates paths like `data/raw/rpscrape/by_region_year_month/gb/2025/04/gb_2025_04_01.csv`.

To move files instead of copying:

```bash
python -m src.ingestion.organize_rpscrape --move
```

Run full historical pipeline (SP ingest + rpscrape enrichment + DQ checks):

```bash
python -m src.pipelines.run_historical --use-sp-history
```

The matcher logs unmatched and ambiguous rows under `logs/` for manual review.

### GB/IE-only preflight before full 2015+ run

Run a preflight gate on a narrow window first (fails with exit code 2 if thresholds are not met):

```bash
python -m src.pipelines.preflight_historical \
	--countries GB,IE \
	--start-year 2019 --start-month 1 \
	--end-year 2019 --end-month 1 \
	--parse-existing-only \
	--require-rps
```

If preflight is green, run full GB/IE backfill with SP history:

```bash
python -m src.ingestion.betfair_historical \
	--use-sp-history \
	--sp-include pricesukwin,pricesirewin \
	--start-year 2015 --start-month 1 \
	--end-year 2025 --end-month 12
```
