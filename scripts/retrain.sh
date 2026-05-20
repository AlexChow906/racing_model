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
echo "  Retrain Pipeline — $TIMESTAMP"
echo "=========================================="

# Step 1: Rebuild feature store on racing.duckdb (has latest data)
echo ""
echo "[1/4] Rebuilding feature store on racing.duckdb..."
$VENV_PYTHON -c "
import sys, os
sys.path.insert(0, 'src')
from ingestion.db_connect import get_db
from quality.checks import ensure_standard_race_flag
from pipelines.run_phase2_feature_store import _prepare_upstream_inputs, _materialize_feature_store

con = get_db('racing.duckdb')
ensure_standard_race_flag(con)
_prepare_upstream_inputs(con)

for sql_file in sorted(os.listdir('sql/features')):
    if sql_file.endswith('.sql'):
        con.execute(open(f'sql/features/{sql_file}').read())

rows = _materialize_feature_store(con)
r = con.execute('SELECT MIN(race_date), MAX(race_date) FROM feature_store').fetchone()
print(f'Feature store: {rows:,} rows ({r[0]} to {r[1]})')
con.close()
"

# Step 2: Walk-forward validation (flat)
echo ""
echo "[2/4] Walk-forward: FLAT..."
RACING_DB=racing.duckdb $VENV_PYTHON -m src.pipelines.backtest \
    --walk-forward --flat-v2 --min-edge 0.15 --category flat --no-graphs

# Step 3: Walk-forward validation (jumps)
echo ""
echo "[3/4] Walk-forward: JUMPS..."
RACING_DB=racing.duckdb $VENV_PYTHON -m src.pipelines.backtest \
    --walk-forward --min-edge 0.15 --category jumps --no-graphs

# Step 4: Ask for approval
echo ""
echo "=========================================="
echo "  Review the results above."
echo "  If satisfied, run:"
echo "    ./scripts/retrain.sh --approve"
echo "=========================================="

if [ "${1:-}" = "--approve" ]; then
    echo ""
    echo "Approved. Training production models..."

    # Copy racing.duckdb as new backtest snapshot
    echo "  Copying racing.duckdb -> racing_backtest.duckdb..."
    cp racing.duckdb racing_backtest.duckdb

    # Train both models from backtest DB
    echo "  Training models..."
    RACING_DB=racing_backtest.duckdb $VENV_PYTHON -c "
import sys, os
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import pickle
from catboost import CatBoost, Pool
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from pathlib import Path

from ingestion.db_connect import get_db
from modeling.train_split import race_softmax, renormalize
from constants.params import CATBOOST_FLAT_PARAMS, TUNED_PARAMS
from constants.features import FLAT_V2_FEATURES, EXCLUDE, JUMPS_DROP

DB = os.environ['RACING_DB']
SETTLED_FILTER = 'AND NOT (res.won = FALSE AND res.sp_decimal IS NULL AND res.finishing_position IS NULL)'

def load_settled(con, start, end, race_type_filter):
    return con.execute(f'''
        SELECT fs.*, ra.race_type
        FROM feature_store fs
        JOIN races ra ON fs.race_id = ra.race_id
        JOIN results res ON fs.runner_id = res.runner_id
        WHERE fs.race_date >= \"{start}\" AND fs.race_date < \"{end}\"
        AND fs.target IS NOT NULL
        {race_type_filter}
        {SETTLED_FILTER}
        ORDER BY fs.race_date, fs.race_id
    ''').df()

# ── FLAT ──
print('=== FLAT (CatBoost) ===', flush=True)
con = get_db(DB)
df_train = load_settled(con, '2015-01-01', '2025-07-01', \"AND ra.race_type = 'Flat'\")
df_cal = load_settled(con, '2025-07-01', '2026-01-01', \"AND ra.race_type = 'Flat'\")
con.close()

for df in [df_train, df_cal]:
    df.drop(columns=['race_type'], inplace=True, errors='ignore')

def prep_flat(df):
    g = df.groupby('race_id', sort=False)['runner_id'].count().values
    y = df['target'].astype(int).values
    X = df[[f for f in FLAT_V2_FEATURES if f in df.columns]].copy()
    for col in FLAT_V2_FEATURES:
        if col not in X.columns:
            X[col] = np.nan
    return X[FLAT_V2_FEATURES], y, g

X_train, y_train, g_train = prep_flat(df_train)
X_cal, y_cal, g_cal = prep_flat(df_cal)
print(f'  train={len(X_train):,}, cal={len(X_cal):,}, {len(FLAT_V2_FEATURES)} features', flush=True)

train_pool = Pool(X_train, label=y_train, group_id=np.repeat(np.arange(len(g_train)), g_train))
cal_pool = Pool(X_cal, label=y_cal, group_id=np.repeat(np.arange(len(g_cal)), g_cal))
model_flat = CatBoost(CATBOOST_FLAT_PARAMS)
model_flat.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=200)

Path('models/tuned/flat').mkdir(parents=True, exist_ok=True)
model_flat.save_model('models/tuned/flat/model.cbm')
print(f'  Saved: {model_flat.best_iteration_} trees', flush=True)

# ── JUMPS ──
print('=== JUMPS (LightGBM + isotonic) ===', flush=True)
con = get_db(DB)
df_train = load_settled(con, '2015-01-01', '2025-01-01', \"AND ra.race_type IN ('Chase', 'Hurdle', 'NH Flat')\")
df_cal = load_settled(con, '2025-01-01', '2026-01-01', \"AND ra.race_type IN ('Chase', 'Hurdle', 'NH Flat')\")
con.close()

for df in [df_train, df_cal]:
    df.drop(columns=['race_type'], inplace=True, errors='ignore')

def prep_jumps(df):
    g = df.groupby('race_id', sort=False)['runner_id'].count().values
    y = df['target'].astype(int).values
    drop_cols = [c for c in EXCLUDE + JUMPS_DROP if c in df.columns]
    X = df.drop(columns=drop_cols)
    non_numeric = [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]
    for col in non_numeric:
        X[col] = pd.Categorical(X[col].astype(str)).codes
    return X, y, g, df

X_train, y_train, g_train, df_train_full = prep_jumps(df_train)
X_cal, y_cal, g_cal, df_cal_full = prep_jumps(df_cal)

cols = sorted(set(X_train.columns) & set(X_cal.columns))
for col in cols:
    if col not in X_train.columns: X_train[col] = np.nan
    if col not in X_cal.columns: X_cal[col] = np.nan
X_train, X_cal = X_train[cols], X_cal[cols]

print(f'  train={len(X_train):,}, cal={len(X_cal):,}, {len(cols)} features', flush=True)

params = TUNED_PARAMS['jumps']
model_jumps = lgb.LGBMRanker(**params)
model_jumps.fit(X_train, y_train, group=g_train,
                eval_set=[(X_cal, y_cal)], eval_group=[g_cal], eval_at=[1],
                callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(50)])

cal_ids = df_cal_full['race_id'].to_numpy()
cal_probs = race_softmax(model_jumps.predict(X_cal, num_iteration=model_jumps.best_iteration_), cal_ids)
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(cal_probs, y_cal)

Path('models/tuned/jumps').mkdir(parents=True, exist_ok=True)
model_jumps.booster_.save_model('models/tuned/jumps/model.lgbm')
with open('models/tuned/jumps/calibrator.pkl', 'wb') as f:
    pickle.dump(calibrator, f)
print(f'  Saved: {model_jumps.best_iteration_} trees, {len(model_jumps.feature_name_)} features', flush=True)
print('Done.', flush=True)
"

    echo ""
    echo "=========================================="
    echo "  Production models updated."
    echo "  racing_backtest.duckdb updated."
    echo "  Push to deploy: git add -A && git commit && git push"
    echo "=========================================="
fi
