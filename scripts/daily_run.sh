#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .env 2>/dev/null || true

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python"
fi

YESTERDAY=$(date -v-1d '+%Y-%m-%d')
YESTERDAY_SLASH=$(echo "$YESTERDAY" | tr '-' '/')
YESTERDAY_YEAR=${YESTERDAY%%-*}
YESTERDAY_MONTH=$(echo "$YESTERDAY" | cut -d- -f2 | sed 's/^0//')
RPSCRAPE_DIR="$PROJECT_DIR/data/raw/rpscrape_repo/scripts"
RPSCRAPE_PYTHON=python

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "=========================================="
echo "  Daily Racing Pipeline — $TIMESTAMP"
echo "=========================================="

# 1. Collect yesterday's results (SP + won status)
echo ""
echo "[1/7] Collecting yesterday's results..."
$VENV_PYTHON -m src.pipelines.collect_results --date yesterday || {
    echo "  WARNING: Results collection failed (CSV may not be available yet)"
}

# 2. Scrape yesterday's rpscrape data (GB + IRE)
echo ""
echo "[2/7] Scraping yesterday's Racing Post data..."
(cd "$RPSCRAPE_DIR" && $RPSCRAPE_PYTHON rpscrape.py -d "$YESTERDAY_SLASH" -r gb) || {
    echo "  WARNING: rpscrape GB failed (data may not be available yet)"
}
(cd "$RPSCRAPE_DIR" && $RPSCRAPE_PYTHON rpscrape.py -d "$YESTERDAY_SLASH" -r ire) || {
    echo "  WARNING: rpscrape IRE failed (data may not be available yet)"
}

# 3. Ingest yesterday's SP CSV to populate horse_history
echo ""
echo "[3/8] Ingesting SP history for horse_history..."
$VENV_PYTHON -m src.ingestion.betfair_historical \
    --use-sp-history --sp-include pricesukwin,pricesirewin \
    --start-year "$YESTERDAY_YEAR" --start-month "$YESTERDAY_MONTH" \
    --end-year "$YESTERDAY_YEAR" --end-month "$YESTERDAY_MONTH" || {
    echo "  WARNING: SP history ingestion failed"
}

# 4. Enrich runners and horse_history with rpscrape data
echo ""
echo "[4/8] Enriching with rpscrape data..."
$VENV_PYTHON -m src.ingestion.rpscrape_enrich \
    --input-glob "data/raw/rpscrape_repo/data/region/*/all/*.csv" || {
    echo "  WARNING: rpscrape enrichment failed"
}

# 5. Backfill horse_history distance/going from enriched races
echo ""
echo "[5/8] Backfilling horse_history from races..."
$VENV_PYTHON -c "
import sys; sys.path.insert(0, 'src')
from ingestion.db_connect import get_db
con = get_db('racing.duckdb')
con.execute('''
    UPDATE horse_history
    SET distance_furlongs = COALESCE(r.distance_furlongs, horse_history.distance_furlongs),
        going_code = COALESCE(r.going_code, horse_history.going_code),
        event_timestamp_utc = r.scheduled_off_utc,
        decision_cutoff_utc = r.decision_cutoff_utc
    FROM races r
    WHERE horse_history.race_id = r.race_id
      AND (horse_history.distance_furlongs IS NULL
           OR horse_history.event_timestamp_utc != r.scheduled_off_utc)
''')
con.close()
" || {
    echo "  WARNING: horse_history backfill failed"
}

# 6. Update P&L tracker
echo ""
echo "[6/8] Updating P&L tracker..."
$VENV_PYTHON -m src.pipelines.track_pnl || {
    echo "  WARNING: P&L update failed (no bets logged yet?)"
}

# 7. Fetch today's cards and score
echo ""
echo "[7/8] Scoring today's races..."
$VENV_PYTHON -m src.pipelines.daily_predictions --date today

# 8. AI analysis + publish to Discord (picks + yesterday's results)
echo ""
echo "[8/8] Publishing to Discord..."
$VENV_PYTHON -m src.pipelines.publish_discord --date today || {
    echo "  WARNING: Discord publish failed (check bot token / channel IDs / LLM key)"
}

echo ""
echo "=========================================="
echo "  Pipeline complete — $(date '+%H:%M:%S')"
echo "  Check logs/daily_bets.csv for today's bets"
echo "  Check logs/pnl_tracker.csv for P&L"
echo "  FADE picks (if any): run scripts review — python -m src.pipelines.review_pending"
echo "=========================================="
