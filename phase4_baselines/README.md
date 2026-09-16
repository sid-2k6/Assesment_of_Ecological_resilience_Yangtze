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
