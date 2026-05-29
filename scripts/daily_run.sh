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

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "=========================================="
echo "  Daily Racing Pipeline — $TIMESTAMP"
echo "=========================================="

# 1. Collect yesterday's results (SP + won status)
echo ""
echo "[1/4] Collecting yesterday's results..."
$VENV_PYTHON -m src.pipelines.collect_results --date yesterday || {
    echo "  WARNING: Results collection failed (CSV may not be available yet)"
}

# 2. Update P&L tracker
echo ""
echo "[2/4] Updating P&L tracker..."
$VENV_PYTHON -m src.pipelines.track_pnl || {
    echo "  WARNING: P&L update failed (no bets logged yet?)"
}

# 3. Fetch today's cards and score
echo ""
echo "[3/4] Scoring today's races..."
$VENV_PYTHON -m src.pipelines.daily_predictions --date today

# 4. AI analysis + publish to Discord (picks + yesterday's results)
echo ""
echo "[4/4] Publishing to Discord..."
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
