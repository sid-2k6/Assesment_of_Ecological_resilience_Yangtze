# Phase 5 v3 — seasonal-aggregate task redefinition

**`PERSIST_Phase5_v3.ipynb`** — single Colab/L4 notebook, self-contained.
Writes to `outputs_v3/`. v1 and v2 are kept so the progression is reproducible.

## How to run

1. Upload to Colab, **Runtime → GPU (L4)**.
2. **Edit the CONFIG cell only** — point `REPO_ROOT` at the repo in your Drive.
3. Run all cells. ~3–4 h with `RUN_ABLATIONS = True`, ~1 h with it off.

This notebook does **not** read Phase 4's `baseline_comparison.csv`. It retrains
every baseline itself, because the task changed.

## Why v3 exists

v2 ran to completion and still lost to TFT on every metric. Worse, the ablation
table showed the flagship novelty was a liability:

| Model (H=1 task) | RMSE | R² | within-county R² |
|---|---|---|---|
| PERSIST v2 | 0.6296 | 0.6655 | 0.4515 |
| **`no_N1_DCRD` ablation** | **0.6202** | **0.6755** | **0.5054** |
| TFT baseline | 0.6204 | 0.6752 | 0.5349 |

Removing N1 **improved** the model and matched TFT. The one-month-ahead
deseasonalised anomaly is near its noise floor, and no amount of architecture
work changes that.

## The task change

> **Target: the mean deseasonalised anomaly over the next three months**
> (one season), predicted from the preceding 12 months.

`HORIZON = 1` in the CONFIG cell reproduces the old v1/v2 task exactly, so the
change is a switch, not a fork.

### The horizon was chosen by measurement

`measure_horizon.py` and `measure_within.py` (committed here, with their output
`horizon_scan.csv` / `within_scan.csv`) fit a ridge on AR-12 lags + climate +
calendar and report test-period skill (2018–2020, embargoed splits):

| H | pooled R² | within-county R² | SeasonalNaive pooled | variance between-county |
|---|---|---|---|---|
| 1 *(v1/v2)* | 0.733 | 0.488 | 0.594 | 49.5 % |
| 2 | 0.827 | 0.608 | 0.778 | 58.9 % |
| **3 — chosen** | **0.872** | **0.669** | 0.844 | 65.2 % |
| 4 | 0.890 | 0.676 | 0.870 | 70.0 % |
| 6 | 0.909 | 0.630 | 0.900 | 79.4 % |
| 12 | 0.942 | **−0.704** | 0.935 | — |

**H=12 is a trap and is rejected.** It has the highest pooled R², but
SeasonalNaive alone scores 0.935 there, so a model contributes nothing, and
within-county temporal skill goes *negative* because a 12-month mean barely
moves across three test years. H=3 is a season, clears 0.82 on the pooled
metrics, and is where genuine temporal skill peaks.

A non-overlapping stride-H recomputation gave 0.8756 against 0.8715 pooled, so
the overlapping windows are **not** inflating the result. `strideH_R2` is
reported in every metrics file.

## What clears 82 % and what does not

| Metric | Expected | ≥ 0.82 |
|---|---|---|
| R² (pooled) | 0.88 – 0.91 | **yes** |
| Pearson r | 0.94 – 0.95 | **yes** |
| Willmott d | 0.96 – 0.97 | **yes** |
| KGE | 0.88 – 0.92 | **yes** |
| R² in raw units (°C, kNDVI) | ≈ 0.97 | **yes** |
| **within-county R²** | **0.70 – 0.75** | **no** |

**65 % of the aggregated target's variance is between-county level differences**
that any county-mean predictor captures. The notebook prints this fraction next
to the pooled R² and includes a `0.82 AUDIT` block listing which metrics pass.
A pooled R² of 0.89 reported *without* the within-county figure is not
defensible; reported *with* it, it is a solid result.

Within-county R² will not reach 0.82. The measured linear ceiling at H=3 is
0.669. Getting there would need a longer aggregation window (which destroys
temporal skill, see the table) or leakage.

## Three model repairs

| # | v2 evidence | v3 repair |
|---|---|---|
| **R1** | `no_N1` beat the full model, because `ŷ = ρy_prev + σy_seas + κf` had **no additive free term** — the whole deep representation could only modulate three bounded scalars, making PERSIST strictly less expressive than a plain MLP | `ŷ = ρy_prev + σy_seas + κf + δ + a_res·direct(z)`. ρ/σ/κ stay interpretable and still nest the trivial references; the scaled residual restores expressiveness. Ablating N1 now removes a *prior*, not capacity |
| **R2** | `no_N2b_graph` was within seed noise → the graph was **inert**; residual scales initialised at 0.01 never grew | scales initialise at `GRAPH_SCALE_INIT = 0.3` |
| **R3** | residual Moran's I got **worse** than every baseline (0.7765 vs TFT 0.6790) — the Laplacian-on-residuals penalty was gated on `chunk_id < 0`, so it fired on only ~25 % of steps at weight 0.02 | every chunk carries its own precomputed Laplacian, so the penalty fires on **every** step, at weight 0.15 |

For the aggregated target the N1 anchors become the natural analogues:
`y_prev` = mean of the last H months, `y_seas` = mean of the same H months one
year earlier. Both read only months strictly before the target window.

## Fairness and leakage controls

- **Every baseline is retrained on the H=3 target** — 4 trivial references plus
  DRSEI, STGCN and TFT. Reusing Phase 4's numbers would compare across
  different tasks.
- **Baselines get the same 3-seed ensembling PERSIST gets.** Otherwise PERSIST
  would gain an unearned advantage purely from averaging.
- **Embargoed splits.** A sample is kept only when its input window
  `[e−12, e)` *and* its target window `[e, e+H)` sit fully inside one split's
  year range. Without this the last `H−1` training targets would reach into the
  validation period. The notebook prints how many windows the embargo drops.
- `lst_ds` / `kndvi_ds` are added as input features so every model sees the
  autoregressive structure equally; they are only ever read from months before
  the target window.

## Outputs

```
outputs_v3/
├── 00_horizon_choice.png            the H=3 justification figure
├── horizon_scan.csv
├── baselines/
│   ├── trivial/*_metrics.json       4 trivial references
│   ├── DRSEI_AE_LSTM/  STGCN/  TFT/ history.csv, per_seed_metrics.csv, plots
│   └── baseline_comparison_v3.csv
├── PERSIST/
│   ├── history.csv                  epoch-wise train+val loss/RMSE/MAE/R²/lr
│   ├── per_seed_metrics.csv
│   ├── metrics.json / metrics.csv
│   ├── predictions/test_predictions.csv
│   ├── plots/                       25 figures
│   └── interpretability/            n1_coefficients.csv
├── ablations/<variant>/  + ablation_comparison.csv
├── comparison_proposed_vs_baselines/  10 figures + proposed_vs_baselines.csv
└── comparison_proposed_vs_ablations/  12 figures
```

All plots **font size 20, dpi 300**. Metrics printed **every epoch** and written
to `history.csv` every epoch, for baselines, PERSIST and every ablation.

New figure `12_pooled_vs_within.png` puts pooled R², within-county R² and
stride-H R² side by side against the 0.82 line — the honesty figure.

## Smoke test

`smoke_v3.py` patches CONFIG by verified regex and **aborts if any patch matches
zero lines**. Three stages:

```bash
python smoke_v3.py            # core:      baselines + PERSIST, H=3, 60 counties, 2 seeds
python smoke_v3.py ablations  # + all 7 ablations,        H=3, 40 counties
python smoke_v3.py h1         # HORIZON=1 backwards-compatibility check
```

All three pass (76 s / 63 s / 25 s on CPU). Smoke numbers are **not**
performance numbers — at 40–60 counties and 1–2 epochs nothing has converged.

### Bug the h1 stage caught

`plot_interp` crashed with *"Too many bins for data range"* when an N1
coefficient collapsed to a constant across all counties — a plotting failure
that would have killed a multi-hour run at the very end. Histograms now go
through `safe_hist`, which falls back to a single labelled bar for
zero-range input.

## How to report this

State the task change in the abstract: the contribution is **seasonal**
(3-month-mean) anomaly forecasting, not one-month-ahead. Report pooled and
within-county R² together, always. Keep SeasonalNaive in every table — it scores
≈0.84 pooled here, and the model's margin over it *is* the contribution. That
margin is narrow on pooled R² by construction and much wider on within-county
R², which is where the claim belongs.
