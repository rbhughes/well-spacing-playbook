# Baselines (Phase 7) -- fold-cycled on train/val, test locked

6,904 wells, 5 folds. MAE in percentage points of peer-P50 pace, on the winsorized target.

| model | val MAE (mean over folds) | spread |
|---|---|---|
| B0 global median | 92.9 | +/-4.5 |
| B1 ridge, own-only | 94.3 | +/-3.4 |
| B2 HistGB own+nbr | 67.3 | +/-0.8 |

B3 quantile GBM: mean pinball 80.6; coverage P10=16%, P50=51%, P90=84%

Gate: B2 BEATS B1 -> neighbour/leg features carry signal
