# PERSIST v3 — full-run results of record

Source: `PERSIST_Phase5_v3.ipynb`, full run on Colab L4, `HORIZON = 3`,
`N_SEEDS = 3`, `RUN_BASELINES = True`, `RUN_ABLATIONS = True`.
All figures below are **test period 2018–2020**, ensemble of 3 seeds.
These are the numbers to quote in the paper. Do not re-derive them.

## 1 · Experimental setup as actually run

| Item | Value |
|---|---|
| Panel | 269,136 rows × 58 columns |
| Counties (N) | 1,068 |
| Months (T) | 252 (2000-01 – 2020-12) |
| Task | mean deseasonalised anomaly over months `[e, e+3)` |
| Lookback | 12 months |
| Targets | `lst_ds` (primary), `kndvi_ds` |
| Global sd of raw anomaly | LST 3.6050 °C; kNDVI 0.1134 |
| Dynamic / static / annual features | 32 / 8 / 64 |
| Graph snapshots (windows) | train 166, val 34, test 34 (embargo drops 2 per split) |
| Valid (county, window) targets | 249,401 |
| Flat per-county samples | train 176,894, val 36,297, test 36,210 |
| Spatial edges | 3,032 (undirected contiguity) |
| Hydrological edges | 3,032 (directed) |
| Node chunks | 3 × 356 counties; edges retained 30.8 / 29.9 / 31.3 % |
| Epochs cap | 60 (PERSIST and baselines), early stopping |
| **Between-county share of target variance** | **65.2 %** |

## 2 · Horizon selection (pre-registration evidence)

Ridge (AR-12 + climate + calendar) skill ceiling, embargoed test period.
Source: `phase5_proposed_model/horizon_scan.csv`, `within_scan.csv`.

| H | pooled R² | within-county R² | SeasonalNaive pooled | variance between-county |
|---|---|---|---|---|
| 1 | 0.733 | 0.488 | 0.594 | 49.5 % |
| 2 | 0.827 | 0.608 | 0.778 | 58.9 % |
| **3 (chosen)** | **0.872** | **0.669** | 0.844 | 65.2 % |
| 4 | 0.890 | 0.676 | 0.870 | 70.0 % |
| 6 | 0.909 | 0.630 | 0.900 | 79.4 % |
| 12 | 0.942 | **−0.704** | 0.935 | — |

Non-overlapping stride-3 check on the ridge: 0.8756 vs 0.8715 pooled → window
overlap does not inflate the result.

## 3 · Trivial references (H = 3)

| Model | RMSE | R² | within-county R² |
|---|---|---|---|
| Climatology | 0.9611 | −0.0394 | 0.0000 |
| CountyClimatology | 0.6154 | 0.5738 | 0.0000 |
| TrailingPersistence | 0.7991 | 0.2814 | −0.9907 |
| **SeasonalNaive** (the bar) | **0.4270** | **0.7948** | **0.5656** |

`Climatology` and `CountyClimatology` have within-county R² = 0 by
construction (they are constant in time within a county).
`TrailingPersistence` is *worse than useless* temporally at this horizon.

## 4 · Deep baselines (H = 3, 3-seed ensembles, same protocol as PERSIST)

| Model | Params | RMSE | MAE | R² | Pearson r | KGE | within R² | Moran's I | shock RMSE | stride-3 R² |
|---|---|---|---|---|---|---|---|---|---|---|
| DRSEI (AE+LSTM) | 142,274 | 0.3871 | 0.2878 | 0.8314 | 0.9126 | 0.8987 | 0.6906 | 0.8037 | 0.3614 | 0.8744 |
| STGCN | 101,730 | 0.4520 | 0.3446 | 0.7701 | 0.8780 | 0.8390 | 0.6068 | **0.5473** | 0.4659 | 0.8124 |
| TFT | 1,510,850 | 0.3970 | 0.3003 | 0.8227 | 0.9119 | 0.8067 | 0.6508 | 0.7565 | 0.3742 | 0.8399 |

Per-seed RMSE / R² / within-R²:
- DRSEI: 0.3969/0.8227/0.6593 · 0.4178/0.8036/0.6421 · 0.4148/0.8064/0.6673 (best val 0.3711, stop ep 12)
- STGCN: 0.4652/0.7565/0.5538 · 0.4674/0.7542/0.5595 · 0.5001/0.7185/0.5429 (best val 0.4257, stop ep 27)
- TFT: 0.4344/0.7876/0.6220 · 0.4213/0.8002/0.6390 · 0.4368/0.7853/0.5253 (best val 0.3949, stop ep 18)

## 5 · PERSIST v3 — headline result

**356,081 parameters. Best val RMSE 0.3526, early stop epoch 27.**

| Metric | Value |
|---|---|
| RMSE | **0.3633** |
| MAE | 0.2660 |
| R² (pooled) | **0.8515** |
| Pearson r | 0.9231 |
| Willmott d | 0.9596 |
| KGE | 0.9052 |
| Bias | +0.0126 |
| **within-county R²** | **0.7092** |
| residual Moran's I | 0.8172 |
| shock-window RMSE | 0.3556 |
| stride-3 R² (non-overlapping) | 0.8897 |
| R² in raw LST units (°C) | 0.9718 |
| Seed spread (RMSE) | 0.3662 ± 0.0045 |

Per-seed: 0.3614/0.8530/0.7103/Moran 0.8116 · 0.3651/0.8500/0.7072/0.8208 ·
0.3722/0.8441/0.6881/0.8185

### Margins that matter

| Comparison | PERSIST | Reference | Margin |
|---|---|---|---|
| RMSE vs SeasonalNaive | 0.3633 | 0.4270 | **−14.9 %** |
| RMSE vs best deep (DRSEI) | 0.3633 | 0.3871 | **−6.1 %** |
| RMSE vs TFT | 0.3633 | 0.3970 | −8.5 % |
| **within-county R² vs SeasonalNaive** | **0.7092** | 0.5656 | **+0.1436 (+25.4 % rel.)** |
| within-county R² vs DRSEI | 0.7092 | 0.6906 | +0.0186 |
| within-county R² vs TFT | 0.7092 | 0.6508 | +0.0584 |
| Parameters vs TFT | 356,081 | 1,510,850 | **4.24× fewer** |

### Wins / losses vs all seven references

Wins on RMSE, MAE, R², Pearson r, KGE, within-county R², shock RMSE, stride-3 R².
**Loses on residual Moran's I** — STGCN reaches 0.5473 against PERSIST's 0.8172.

### 0.82 audit

| Metric | Value | ≥ 0.82 |
|---|---|---|
| R² (pooled) | 0.8515 | **PASS** |
| Pearson r | 0.9231 | **PASS** |
| Willmott d | 0.9596 | **PASS** |
| KGE | 0.9052 | **PASS** |
| R² raw LST units | 0.9718 | **PASS** |
| stride-3 R² | 0.8897 | **PASS** |
| within-county R² | 0.7092 | below 0.82 |

Six of seven clear 0.82. The pre-run prediction for within-county R² was
0.70–0.75; it landed at **0.7092**, inside the predicted band.

## 6 · Ablations (3-seed ensembles)

Seed-noise band: PERSIST RMSE σ = 0.0045, so **|ΔRMSE| must exceed 0.0090**
to be interpretable.

| Variant | Params | RMSE | ΔRMSE | Significant | R² | within R² | Moran's I | shock RMSE |
|---|---|---|---|---|---|---|---|---|
| **PERSIST (full)** | 356,081 | **0.3633** | — | — | 0.8515 | 0.7092 | 0.8172 | 0.3556 |
| no_N1_DCRD | 356,081 | 0.3795 | **+0.0162** | **yes** | 0.8379 | 0.7057 | 0.7952 | 0.3593 |
| no_N5_annual | 309,232 | 0.3794 | **+0.0161** | **yes** | 0.8380 | 0.6874 | 0.7977 | 0.3353 |
| no_N3_moe | 298,142 | 0.3651 | +0.0018 | no | 0.8500 | 0.7023 | 0.8345 | 0.3564 |
| no_N2b_graph | 281,199 | 0.3642 | +0.0009 | no | 0.8507 | 0.7045 | 0.8143 | 0.3540 |
| no_N4_ecoloss | 356,081 | 0.3641 | +0.0008 | no | 0.8508 | 0.7075 | 0.8201 | 0.3585 |
| no_N6_attr | 355,309 | 0.3641 | +0.0008 | no | 0.8508 | 0.7007 | 0.8203 | 0.3580 |
| no_N2a_hydro | 318,640 | 0.3625 | −0.0008 | no | 0.8521 | 0.7052 | 0.8267 | 0.3592 |

ΔRMSE > 0 means removing the component **hurt**, i.e. the component helps.

Seed spreads: no_N1 0.3852±0.0036 · no_N2a 0.3657±0.0028 · no_N2b 0.3676±0.0039 ·
no_N3 0.3689±0.0026 · no_N4 0.3671±0.0047 · no_N5 0.3826±0.0053 ·
no_N6 0.3677±0.0052

**Only two of seven components are statistically defensible: N1 and N5.**

## 7 · Did the three v2 repairs work?

| Repair | Intent | Outcome |
|---|---|---|
| **R1** — N1 as structured prior + free residual | make N1 contribute instead of hurting | **Worked.** N1 went from ΔRMSE **−0.0094 (hurt)** in v2 to **+0.0162 (helps, significant)** in v3 |
| **R2** — graph scales init 0.3 | wake the inert graph | **Did not work.** `no_N2b_graph` ΔRMSE +0.0009, still inside noise. The dual graph contributes nothing measurable to accuracy |
| **R3** — per-chunk Laplacian every step, weight 0.15 | reduce residual Moran's I | **Did not work.** Moran's I 0.8172, *worse* than v2's 0.7765 and far worse than STGCN's 0.5473 |

## 8 · Task-change effect (context only, not a like-for-like comparison)

| Metric | v2 (H=1) | v3 (H=3) |
|---|---|---|
| RMSE | 0.6296 | 0.3633 |
| R² | 0.6655 | 0.8515 |
| within-county R² | 0.4515 | 0.7092 |
| residual Moran's I | 0.7765 | 0.8172 |
| shock RMSE | 0.7161 | 0.3556 |

v2 and v3 solve **different tasks**. This table documents the effect of the task
redefinition; it is not evidence of model improvement. The valid comparison is
§4–5, where every baseline is retrained on the H=3 target.

## 9 · Honest issues to address in the write-up

1. **Residual Moran's I is the weak result.** PERSIST 0.8172 vs STGCN 0.5473.
   STGCN's plain graph convolution absorbs spatial structure far better than
   PERSIST's gated graph attention. The paper must report this and not claim
   spatial-structure benefits. The N2/N4 spatial machinery does not deliver.

2. **Pooled R² 0.8515 sits *below* the linear ridge ceiling of 0.872.** A
   reviewer will ask why a 356k-parameter model does not beat AR-12 + climate.
   **Action: add a Ridge/linear AR baseline to the results table** — it is
   cheap, it is already implemented in `measure_horizon.py`, and omitting it
   after we measured it would look like selective reporting. On stride-3 R²
   PERSIST does lead (0.8897 vs 0.8756).

3. **Five of seven novelties are within seed noise.** The paper should claim
   two components (N1 dynamical decoder, N5 hierarchical annual branch) and
   present the other five as *tested and not supported at this scale* rather
   than as contributions. Claiming six novelties is not defensible against this
   table.

4. **SeasonalNaive at R² 0.7948 is a strong reference.** Pooled margin is
   +0.0567. The defensible headline is the **within-county margin (+0.1436,
   +25.4 % relative)** plus 4.24× fewer parameters than TFT.

5. **65.2 % of variance is between-county.** Pooled R² must never appear
   without within-county R² beside it.

## 10 · Artefact locations (in the user's Drive)

```
outputs_v3/
├── 00_horizon_choice.png, horizon_scan.csv
├── baselines/  trivial/ + DRSEI_AE_LSTM/ STGCN/ TFT/ + baseline_comparison_v3.csv
├── PERSIST/    25 figures, history.csv, per_seed_metrics.csv, metrics.json,
│               predictions/, interpretability/n1_coefficients.csv
├── ablations/  7 variants + ablation_comparison.csv
├── comparison_proposed_vs_baselines/   10 figures + proposed_vs_baselines.csv
└── comparison_proposed_vs_ablations/   12 figures
```

**Still needed for the resilience-assessment section of the paper:**
`outputs_v3/PERSIST/interpretability/n1_coefficients.csv` — the per-county
ρ, σ, κ values that yield recovery rate (1−ρ) and resistance (1−|κ|). None of
their summary statistics were printed in the run log, so the spatial resilience
results cannot be written until that file is available.
