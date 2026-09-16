# Phase 4 — Deep Learning Baselines

**`PERSIST_Phase4_Baselines.ipynb`** — single end-to-end notebook, Colab/L4 ready.

## How to run

1. Clone this repo into Google Drive.
2. Open the notebook in Colab, set **Runtime → GPU (L4)**.
3. **Edit the CONFIG cell only** — set `REPO_ROOT` to wherever the repo sits in your Drive:
   ```python
   REPO_ROOT = Path("/content/drive/MyDrive/Assesment_of_Ecological_resilience_Yangtze")
   ```
4. Run all cells.

Nothing else needs changing. Every other path is derived from `REPO_ROOT`.

## The three baselines

| # | Baseline | Architecture | Isolates |
|---|---|---|---|
| 1 | **DRSEI** | Autoencoder + LSTM (Gong et al. 2025, *Remote Sensing* 17(3):558) | N2 dual graph, N3 regime MoE — temporal only, no spatial structure |
| 2 | **STGCN** | Graph conv + gated temporal conv | N1 disturbance-response, N4 ecological loss — same information and graph, generic objective |
| 3 | **TFT** | Temporal Fusion Transformer | N1, N5 hierarchy, N6 attribution — attention + static covariates + built-in interpretability |

All deep learning, no tree ensembles.

STGCN deliberately uses **only the spatial graph**. Dual-graph fusion is PERSIST's N2 and is not given away to a baseline.

## Task

Forecast next-month ecological state anomalies (`lst_z`, `kndvi_z`) from a 12-month lookback of state + forcing + static context.

**Split is by time, never randomly** — random splits leak spatially and temporally autocorrelated information:

| Split | Years |
|---|---|
| Train | 2000–2014 |
| Validation | 2015–2017 |
| Test | 2018–2020 |

## Metrics

Deliberately identical to those planned for PERSIST, so the comparison table transfers unchanged.

| Metric | Purpose |
|---|---|
| RMSE, MAE | error magnitude |
| R² (1 − SSE/SST) | variance explained |
| Pearson r | pattern agreement independent of bias |
| Willmott's d | index of agreement, standard in ecological modelling |
| KGE | Kling–Gupta; decomposes correlation / variability / bias |
| Bias (ME) | systematic over/under-prediction |
| **Residual Moran's I** | **cleanest single test of the graph contribution (N2)** — should approach 0. Harder to game than R². |
| **Shock-month RMSE** | error on `\|heat_z\| ≥ 1.5` months (N1) — resilience is defined by response to real disturbance |
| Per-reach RMSE | upstream / midstream / downstream generalisation (N3) |

KGE's bias term is normalised by the observed standard deviation rather than the mean, because targets are standardised anomalies with means near zero.

Metrics are reported overall **and** per target.

## Outputs

```
outputs/
├── DRSEI_AE_LSTM/
│   ├── history.csv                  epoch-wise train + val loss/RMSE/MAE/R²/lr/seconds
│   ├── metrics.json / metrics.csv   full test metric bundle
│   ├── model_best.pt                best checkpoint by val RMSE
│   ├── predictions/test_predictions.csv
│   └── plots/                       17 figures
├── STGCN/                           (same structure)
├── TFT/                             (same structure)
├── comparison_plots/                12 figures
└── baseline_comparison.csv
```

All plots at **font size 20, dpi 300**.

Per-baseline figures: loss curve, RMSE/MAE/R² curves, LR schedule, predicted-vs-observed hexbin per target, residual histogram, residual-vs-prediction, residual Q–Q, error by reach, error seasonality, error by test year, calm-vs-shock error, example county trajectories, spatial RMSE choropleth, metric summary.

Comparison figures: grouped bars for each metric, combined validation curves, per-reach RMSE, normalised radar profile, parameters-vs-accuracy.

Training and validation metrics are **printed every epoch** and **written to `history.csv` every epoch**, so an interrupted run still leaves a usable history.

## Methodological points built into the notebook

1. **Anomalies are rebuilt on train-only climatology.** The shipped `*_z` columns in `panel_monthly.parquet` use full-period county×month statistics, which leaks test information into the target definition. The notebook recomputes them from raw `lst_c` / `kndvi` using train-period means and standard deviations only.
2. **Feature scaling is fitted on train only.**
3. **`n_rel` is included as a covariate**, following the Phase 3e finding that LST clear-sky sampling leaks ≈0.20 correlation into the anomaly. Omitting it would let part of the apparent forcing response be a sampling artefact.
4. **Early stopping on validation RMSE** with `ReduceLROnPlateau`, best checkpoint restored before test evaluation.

## Files

| File | Purpose |
|---|---|
| `PERSIST_Phase4_Baselines.ipynb` | the notebook to run |
| `build_notebook.py` | generator that produces the notebook (edit this, not the `.ipynb`) |
| `smoke_test.py` | extracts cells, patches CONFIG for a tiny local subset, executes in order |

## Smoke test

Verified end-to-end on CPU with 60 counties × 2 epochs: all 22 code cells execute, all three models train and evaluate, 17 plots per baseline and 12 comparison plots are written, and `baseline_comparison.csv` is produced.

Metric *values* from the smoke run are meaningless (negative R² on 60 counties and 2 epochs) — the test verifies execution, not performance.

Two bugs were found and fixed during smoke testing:
- `Axes.boxplot(labels=...)` was renamed to `tick_labels` in matplotlib 3.9+; tick labels are now set manually so the notebook is version-portable.
- The TFT attention output is `(B, L, d)` and was being added to a `(B, d)` slice; the final position is now selected explicitly, which is the correct behaviour for single-horizon forecasting.

## Note on excluding classical ML

No tree-ensemble baseline is included, per the deep-learning-only constraint. This is defensible because the monthly panel has **269,136 observations**, where deep learning is genuinely appropriate — the earlier sample-size concern applied to the 22,428-row annual panel.

If a reviewer asks for a classical comparator, the cheapest response is a single XGBoost reference row appended to `baseline_comparison.csv` rather than a restructure.


---

# Revision 1 — fixing the negative-R² result

The first full run returned **negative R² and near-zero Pearson r for all three models**. Diagnosis: the fault was the **target definition**, not the code. Three unrelated architectures failing identically points at the data, not the models.

## What was wrong

| Problem | Evidence | Fix |
|---|---|---|
| **Per-county-month standardisation divided by near-zero standard deviations** | `kndvi_z` max reached **60.955** — a 61-sigma value. A handful of these dominated every squared-error metric. | Primary target now uses a **single global scale**, which cannot explode. Strict target has county sds **floored** at 10 % of the global sd and values **clipped at ±5σ**. |
| **County-demeaning removed nearly all predictable signal** | Measured monthly lag-1 autocorrelation is only **0.155** (LST) / **0.209** (kNDVI), capping achievable R² at ≈0.03. The variance decomposition confirms `lst_z` is **0.0 % between-county / 100 % within-county** — pure hard temporal noise. | Primary target is now **deseasonalised state**: the regional month climatology is removed (so the trivial seasonal cycle can't be exploited) but county structure is **retained**. `lst_ds` splits 41 % between / 59 % within. |
| **No floor** — a negative R² was uninterpretable | Nothing to compare against | **Four trivial baselines** added |
| **Train/test distribution shift** | Test-period target variance ≈2× train, from warming (+0.032 °C/yr) and greening (+0.00206/yr) trends. Produced the structural −0.42 to −0.53 bias. | `year_frac` trend feature added |
| **No way to separate spatial from temporal skill** | — | **Variance decomposition** + **within-county R²** now reported |

## Trivial reference baselines — the floor

| Reference | Prediction |
|---|---|
| **Climatology** | 0 — the regional month normal |
| **County climatology** | the county's own train-period mean offset |
| **Persistence** | last month's value |
| **Seasonal naive** | the same month one year earlier |

Written into `baseline_comparison.csv` tagged `type = trivial`, and drawn as a red dashed reference line on the RMSE comparison plots.

## Metrics now reported in three spaces

A model trained on the primary target is scored in all three via an **exact affine back-transform**, so no retraining is needed:

- **Primary** (`*_ds`) — deseasonalised, what the model optimises
- **Raw** (`raw_lst_c_*`) — original physical units
- **Strict** (`strict_lst_z_*`) — county-demeaned interannual anomaly, the hard view

Plus two new diagnostics:

- **`within_county_R2`** — R² after removing each county's mean. Reveals whether a model has learned any **temporal** signal or is merely ranking counties.
- **Variance decomposition** — between- vs within-county share of target variance, printed for every target.

## Result of the fix (smoke test: 60 counties, 2 epochs)

| Model | RMSE | R² | Pearson r | within-county R² |
|---|---|---|---|---|
| Climatology | 1.0200 | −0.147 | — | −0.000 |
| CountyClimatology | 0.7839 | +0.323 | 0.689 | −0.000 |
| **Persistence** | **0.7439** | **+0.390** | 0.694 | −0.552 |
| **SeasonalNaive** | **0.6168** | **+0.580** | 0.788 | −0.235 |
| DRSEI | 0.8290 | +0.242 | 0.652 | +0.004 |
| STGCN | 0.9287 | +0.049 | 0.592 | −0.005 |
| TFT | 0.6950 | +0.467 | 0.719 | −0.004 |

**R² is positive and Pearson r is 0.59–0.72**, versus negative R² and r ≈ 0 before. The metrics are now interpretable.

## Two honest caveats the fix exposed

**1. SeasonalNaive is the real bar, not Persistence.** At smoke-test scale, "same month last year" (RMSE 0.617) beats every deep model including TFT (0.695). The verdict line in the notebook checks against Persistence, which TFT clears by 6.6 % — but **SeasonalNaive is the stronger reference and should be the headline comparison.** A deep model that cannot beat a one-line seasonal lookup is not earning its complexity.

**2. `within_county_R2` ≈ 0 for all deep models.** Their positive R² comes almost entirely from the **between-county** component — i.e. from learning which counties are warm or green, not from predicting temporal dynamics. Since `lst_ds` is 41 % between-county variance, that spatial component is legitimately part of the target, but it must be reported honestly rather than presented as forecasting skill.

These are 60-county, 2-epoch numbers and should improve substantially at full scale (1,068 counties, 60 epochs). But both diagnostics will remain the honest tests, and **PERSIST must beat SeasonalNaive and show non-trivial within-county R²** to justify itself.
