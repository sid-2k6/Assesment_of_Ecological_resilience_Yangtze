# Phase 5 v2 — PERSIST revised

**`PERSIST_Phase5_v2.ipynb`** — single Colab/L4 notebook. Supersedes
`PERSIST_Phase5_Proposed_Model.ipynb` (v1) but does not delete it; v1 is kept so
the v1→v2 comparison in the paper is reproducible.

## How to run

1. Upload to Colab, **Runtime → GPU (L4)**.
2. **Edit the CONFIG cell only** — point `REPO_ROOT` at the repo in your Drive.
3. Run all cells.

Writes to `outputs_v2/`, so it will not overwrite the v1 run. Reads
`outputs/baseline_comparison.csv` from Phase 4 for the baseline comparison.

## Why v2 exists

The v1 full run **lost to the TFT baseline on every metric**, and 5 of 7
ablations showed the novelties making things *worse*:

| Metric | PERSIST v1 | TFT | Verdict |
|---|---|---|---|
| RMSE | 0.6355 | **0.6204** | lose |
| R² | 0.6593 | **0.6752** | lose |
| within-county R² | 0.4515 | **0.5349** | lose |
| residual Moran's I | 0.7519 | **0.6790** | lose |
| shock RMSE | 0.7292 | **0.6508** | lose |

### Root cause: effective sample size, not capacity

The graph formulation batches **all 1,068 counties into one snapshot**, so an
epoch contains only **168 training samples** — one per target month. The
per-county baselines (DRSEI, TFT) see **179,292** samples, because each
(county, month) pair is its own sample.

v1 therefore ran 21 optimiser steps per epoch against TFT's ~350, while fitting
381k parameters to 168 effective samples. It overfit hard: train R² 0.844 vs
val R² 0.674, best validation at epoch 13 of 120.

## The four fixes

| # | Fix | Addresses |
|---|---|---|
| **1** | **Node-chunked graph batching** — each step uses a spatially coherent county subset, with occasional full-graph steps. Multiplies gradient steps *and* acts as edge dropout. | tiny update budget + overfitting |
| **2** | **Repaired novelties** — every novelty now starts as a no-op (below) | 5/7 novelties hurting |
| **3** | **EMA weights + cosine schedule with warmup + warm-start at the nested baseline** | optimisation instability |
| **4** | **Multi-seed ensembling with error bars** (`N_SEEDS=3`) | v1's ablation deltas were all within ±0.013 — inside run-to-run noise, so "5 novelties hurt" may have been an artefact |

### Fix 2 in detail — every novelty starts as a no-op

The governing principle: a novelty's contribution is added as a **residual
scaled by a learnable parameter initialised at ~0**. The optimiser has to
switch each novelty on. A novelty that cannot earn its place stays off instead
of injecting noise.

| Novelty | v1 problem | v2 repair |
|---|---|---|
| **N1** | random init, so training starts far from the nested baseline | **warm-start**: σ bias initialised high, so the model *begins* near SeasonalNaive and improves from there |
| **N2** | graph message added raw to `h`, injecting noise | **gated residual, learnable scale init ~0** |
| **N4** | weights too strong; Laplacian over-smoothed the predictions | weights cut 10×; Laplacian moved onto **residuals** — the training-time analogue of residual Moran's I, which is the metric it is supposed to improve |
| **N5** | annual GRU fed the *same* features averaged (redundant), then *gated* against the monthly branch, diluting it | fed **year-over-year deltas**; combined by scaled residual, not a gate |
| **N6** | attribution head computed but never used and never supervised — untrained noise added to the output | attribution weights now **functionally gate the four information streams**, so the head is used and trainable |

N3 (lithology-adaptive MoE) was already a no-op-safe soft gate; only its
load-balancing weight changed. Its v1 expert entropy of 0.998 means the gate
was uniform — worth re-checking at full scale.

## What is unchanged from Phase 4

Targets, train-only climatology, feature scaling, temporal splits and the whole
`evaluate()` function are **copied byte-for-byte** from the baseline notebook.
Same metrics in the same three spaces (primary / raw / strict), same
diagnostics (`within_county_R2`, residual Moran's I, shock RMSE, per-reach).
No new raw data. This is what makes the comparison legitimate.

## Honest expectation

Target: **beat TFT on every metric**, with the largest margins on residual
Moran's I and shock RMSE — the two diagnostics the novelties should genuinely
own. Realistic range: **RMSE ≈ 0.57–0.60, R² ≈ 0.70–0.73**.

**R² of 0.82 on this target is not attainable without leakage.** The
deseasonalised monthly anomaly has a measured lag-1 autocorrelation of 0.155,
and TFT reaches 0.675 with 657k parameters. An 0.82 here would require target
leakage, a non-temporal split, or headlining raw-space R² (which still contains
the seasonal cycle — TFT already scores 0.9376 there, so it would prove
nothing). If v2 still loses, the defensible move is to change the *task* to
seasonal or multi-month aggregates, which are genuinely more predictable and
arguably a better match for resilience, not to inflate the metric.

## Ablations

Unchanged set — each variant switches **one** novelty off through the same code
path, so implementation drift is impossible. Now run over `N_SEEDS` seeds so
`ablation_comparison.csv` carries `all_RMSE_seed_std` alongside
`dRMSE_vs_PERSIST`, and the summary flags any delta smaller than the seed-noise
band as **not significant**. This is the check v1 was missing.

## Outputs

```
outputs_v2/
├── PERSIST/
│   ├── history.csv                epoch-wise train+val loss/RMSE/MAE/R²/lr
│   ├── per_seed_metrics.csv       one row per seed
│   ├── metrics.json / metrics.csv  ensemble + seed mean±std
│   ├── model_best.pt
│   ├── predictions/test_predictions.csv
│   ├── plots/                     24 figures
│   └── interpretability/          n1_coefficients.csv, n3_expert_usage.csv
├── ablations/<variant>/           history.csv, per_seed_metrics.csv, plots
├── ablations/ablation_comparison.csv
├── comparison_proposed_vs_baselines/   9 figures + proposed_vs_baselines.csv
└── comparison_proposed_vs_ablations/  12 figures
```

All plots **font size 20, dpi 300**. Metrics printed **every epoch** and
written to `history.csv` every epoch, for PERSIST *and* every ablation.

## Smoke test

`smoke_v2.py` patches CONFIG to a tiny local budget and executes all 18 code
cells. Two stages, because the full ablation sweep exceeds a single command
timeout:

```bash
python smoke_v2.py            # PERSIST only, 60 counties, 3 epochs, 2 seeds
python smoke_v2.py ablations  # + all 7 ablations, 40 counties, 2 epochs, 1 seed
```

Both stages pass. Stage B produces 164 PNGs and 33 CSVs across all four output
subtrees.

**Smoke numbers are not performance numbers.** At 40–60 counties and 2–3
epochs nothing has converged; the run only proves every code path executes and
every artefact is written. Real figures require the full 1,068-county,
150-epoch, 3-seed run on an L4.

### Harness bug worth recording

The v1 harness patched CONFIG with exact-string keys like `"SMOKE_TEST = False"`,
but the notebook writes column-aligned assignments (`SMOKE_TEST    = False`).
`str.replace` silently matched nothing, so the "smoke test" launched the full
1,068-county / 150-epoch / 3-seed / 8-config job and died on the command
timeout with a single `history.csv` to show for it. `smoke_v2.py` now patches by
regex and **aborts if any patch matches zero lines** — a silent no-op patch is
worse than a crash.
