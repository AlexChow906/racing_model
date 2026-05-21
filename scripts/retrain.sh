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
echo "[1/5] Rebuilding feature store on racing.duckdb..."
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
echo "[2/5] Walk-forward: FLAT..."
RACING_DB=racing.duckdb $VENV_PYTHON -m src.pipelines.backtest \
    --walk-forward --flat-v2 --min-edge 0.15 --category flat --no-graphs

# Step 3: Walk-forward validation (chase)
echo ""
echo "[3/5] Walk-forward: CHASE..."
RACING_DB=racing.duckdb $VENV_PYTHON -c "
import sys, os
sys.path.insert(0, 'src')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from ingestion.db_connect import get_db
from modeling.train_split import race_softmax, renormalize
from constants.windows import WALK_FORWARD_WINDOWS
from constants.params import TUNED_PARAMS
from constants.features import EXCLUDE, JUMPS_DROP

DB = os.environ['RACING_DB']
SETTLED = 'AND NOT (res.won = FALSE AND res.sp_decimal IS NULL AND res.finishing_position IS NULL)'

def load_type(start, end, types):
    tl = ','.join(f\"'{t}'\" for t in types)
    con = get_db(DB)
    df = con.execute(f'''SELECT fs.*, ra.race_type FROM feature_store fs
        JOIN races ra ON fs.race_id = ra.race_id
        JOIN results res ON fs.runner_id = res.runner_id
        WHERE fs.race_date >= \"{start}\" AND fs.race_date < \"{end}\"
        AND fs.target IS NOT NULL AND ra.race_type IN ({tl}) {SETTLED}
        ORDER BY fs.race_date, fs.race_id''').df()
    con.close()
    df.drop(columns=['race_type'], inplace=True, errors='ignore')
    g = df.groupby('race_id', sort=False)['runner_id'].count().values
    y = df['target'].astype(int).values
    drop_cols = [c for c in EXCLUDE + JUMPS_DROP if c in df.columns]
    X = df.drop(columns=drop_cols)
    for col in [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]:
        X[col] = pd.Categorical(X[col].astype(str)).codes
    return X, y, g, df

con2 = get_db(DB)
sp_df = con2.execute('SELECT runner_id, sp_decimal FROM results WHERE sp_decimal > 1').df()
con2.close()

for label, types, param_key in [('CHASE', ['Chase'], 'chase'), ('HURDLE', ['Hurdle', 'NH Flat'], 'hurdle')]:
    params = TUNED_PARAMS[param_key]
    all_bets = []
    for w in WALK_FORWARD_WINDOWS:
        X_tr, y_tr, g_tr, df_tr = load_type('2015-01-01', w['train_end'], types)
        X_ca, y_ca, g_ca, df_ca = load_type(w['cal_start'], w['cal_end'], types)
        X_te, y_te, g_te, df_te = load_type(w['test_start'], w['test_end'], types)
        if len(X_te) == 0: continue
        cols = sorted(set(X_tr.columns) & set(X_ca.columns) & set(X_te.columns))
        X_tr, X_ca, X_te = X_tr[cols], X_ca[cols], X_te[cols]
        model = lgb.LGBMRanker(**params)
        model.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_ca, y_ca)], eval_group=[g_ca], eval_at=[1],
                  callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(0)])
        cal_ids = df_ca['race_id'].to_numpy()
        cal_probs = race_softmax(model.predict(X_ca, num_iteration=model.best_iteration_), cal_ids)
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(cal_probs, y_ca)
        test_ids = df_te['race_id'].to_numpy()
        raw = race_softmax(model.predict(X_te, num_iteration=model.best_iteration_), test_ids)
        cal = calibrator.transform(raw)
        cal = np.clip(np.nan_to_num(cal, nan=1e-6), 1e-6, 1.0)
        probs = renormalize(cal, test_ids)
        a = df_te[['race_id','runner_id','race_date']].copy()
        a['model_prob'] = probs; a['target'] = y_te
        a = a.merge(sp_df, on='runner_id', how='left')
        a['implied'] = np.where(a['sp_decimal']>1, 1.0/a['sp_decimal'], np.nan)
        a['edge'] = a['model_prob'] - a['implied']
        a['profit'] = a['target']*(a['sp_decimal']-1) - (1-a['target'])
        all_bets.append(a)
    bets = pd.concat(all_bets)
    vb = bets[(bets['edge']>0.15) & bets['sp_decimal'].notna()]
    print(f'{label}: {len(vb)} bets, ROI={vb[\"profit\"].mean()*100:+.1f}%, P&L={vb[\"profit\"].sum():+.0f}', flush=True)
"

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

    # Train all models from backtest DB
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
from modeling.train_split import race_softmax
from constants.params import CATBOOST_FLAT_PARAMS, TUNED_PARAMS
from constants.features import FLAT_V2_FEATURES, EXCLUDE, JUMPS_DROP

DB = os.environ['RACING_DB']
SETTLED = 'AND NOT (res.won = FALSE AND res.sp_decimal IS NULL AND res.finishing_position IS NULL)'

def load_settled(con, start, end, race_type_filter):
    return con.execute(f'''
        SELECT fs.*, ra.race_type FROM feature_store fs
        JOIN races ra ON fs.race_id = ra.race_id
        JOIN results res ON fs.runner_id = res.runner_id
        WHERE fs.race_date >= \"{start}\" AND fs.race_date < \"{end}\"
        AND fs.target IS NOT NULL {race_type_filter} {SETTLED}
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
        if col not in X.columns: X[col] = np.nan
    return X[FLAT_V2_FEATURES], y, g

X_tr, y_tr, g_tr = prep_flat(df_train)
X_ca, y_ca, g_ca = prep_flat(df_cal)
print(f'  train={len(X_tr):,}, cal={len(X_ca):,}, {len(FLAT_V2_FEATURES)} features', flush=True)
train_pool = Pool(X_tr, label=y_tr, group_id=np.repeat(np.arange(len(g_tr)), g_tr))
cal_pool = Pool(X_ca, label=y_ca, group_id=np.repeat(np.arange(len(g_ca)), g_ca))
model = CatBoost(CATBOOST_FLAT_PARAMS)
model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=200)
Path('models/tuned/flat').mkdir(parents=True, exist_ok=True)
model.save_model('models/tuned/flat/model.cbm')
print(f'  Saved: {model.best_iteration_} trees', flush=True)

# ── CHASE + HURDLE ──
def train_jumps_model(race_types, model_name, param_key, train_end, cal_start, cal_end):
    params = TUNED_PARAMS[param_key]
    tl = ','.join(f\"'{t}'\" for t in race_types)
    con = get_db(DB)
    df_train = load_settled(con, '2015-01-01', train_end, f'AND ra.race_type IN ({tl})')
    df_cal = load_settled(con, cal_start, cal_end, f'AND ra.race_type IN ({tl})')
    con.close()
    for df in [df_train, df_cal]:
        df.drop(columns=['race_type'], inplace=True, errors='ignore')
    def prep(df):
        g = df.groupby('race_id', sort=False)['runner_id'].count().values
        y = df['target'].astype(int).values
        drop_cols = [c for c in EXCLUDE + JUMPS_DROP if c in df.columns]
        X = df.drop(columns=drop_cols)
        for col in [c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c]))]:
            X[col] = pd.Categorical(X[col].astype(str)).codes
        return X, y, g, df
    X_tr, y_tr, g_tr, df_tr = prep(df_train)
    X_ca, y_ca, g_ca, df_ca = prep(df_cal)
    cols = sorted(set(X_tr.columns) & set(X_ca.columns))
    X_tr, X_ca = X_tr[cols], X_ca[cols]
    print(f'  train={len(X_tr):,}, cal={len(X_ca):,}, {len(cols)} features', flush=True)
    m = lgb.LGBMRanker(**params)
    m.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_ca, y_ca)], eval_group=[g_ca], eval_at=[1],
          callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(50)])
    cal_ids = df_ca['race_id'].to_numpy()
    cal_probs = race_softmax(m.predict(X_ca, num_iteration=m.best_iteration_), cal_ids)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(cal_probs, y_ca)
    out = Path(f'models/tuned/{model_name}')
    out.mkdir(parents=True, exist_ok=True)
    m.booster_.save_model(str(out / 'model.lgbm'))
    with open(out / 'calibrator.pkl', 'wb') as f: pickle.dump(calibrator, f)
    print(f'  Saved: {m.best_iteration_} trees -> {out}/', flush=True)

print('=== CHASE (LightGBM + isotonic) ===', flush=True)
train_jumps_model(['Chase'], 'chase', 'chase', '2025-01-01', '2025-01-01', '2026-01-01')

print('=== HURDLE (LightGBM + isotonic) ===', flush=True)
train_jumps_model(['Hurdle', 'NH Flat'], 'hurdle', 'hurdle', '2025-01-01', '2025-01-01', '2026-01-01')

print('Done.', flush=True)
"

    echo ""
    echo "=========================================="
    echo "  Production models updated (flat, chase, hurdle)."
    echo "  racing_backtest.duckdb updated."
    echo "  Push to deploy: git add -A && git commit && git push"
    echo "=========================================="
fi
