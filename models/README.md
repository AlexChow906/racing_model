# Model Artifacts

## Directory Structure

```
models/
  default/          Default LightGBM params (lr=0.01, leaves=63)
    flat/
      model.lgbm
      calibrator.pkl
    jumps/
      model.lgbm
      calibrator.pkl
  tuned/            Optuna-tuned params
    flat/
      model.lgbm    (lr=0.044, leaves=57, depth=7)
      calibrator.pkl
    jumps/
      model.lgbm    (lr=0.045, leaves=100, depth=9)
      calibrator.pkl
```

## Model Details

Both models are LightGBM LambdaRank rankers with isotonic calibration.

- **Flat model**: Trained on 2015-2023 flat races, calibrated on 2024, ~69 features
- **Jumps model**: Trained on 2015-2023 jumps races, calibrated on 2024, ~65 features (no draw features)

The calibrator (isotonic regression) converts raw softmax probabilities into calibrated win probabilities. Both the `.lgbm` and `calibrator.pkl` are needed for inference.

## Tuned Hyperparameters

Found via Optuna (80 trials per model), optimising for value betting ROI:

| Param | Flat | Jumps |
|-------|------|-------|
| learning_rate | 0.04369 | 0.04460 |
| num_leaves | 57 | 100 |
| min_child_samples | 69 | 63 |
| max_depth | 7 | 9 |
| subsample | 0.88 | 0.74 |
| colsample_bytree | 0.96 | 0.59 |
| reg_alpha | 2.9e-06 | 0.272 |
| reg_lambda | 0.0013 | 0.0045 |

## Walk-Forward Results (Tuned, edge>5%)

| Window | Flat ROI | Flat P&L | Jumps ROI | Jumps P&L |
|--------|----------|----------|-----------|-----------|
| Test 2022 | +11.21% | £+1,389 | +18.39% | £+1,993 |
| Test 2023 | +6.14% | £+749 | +11.79% | £+987 |
| Test 2024 | +5.23% | £+732 | +7.09% | £+612 |
| Test 2025-26 | +8.31% | £+1,274 | +8.32% | £+941 |
| **Total** | **+7.69%** | **£+4,144** | **+11.58%** | **£+4,534** |
