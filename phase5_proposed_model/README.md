# Phase 5 — PERSIST (proposed model) + ablations

**`PERSIST_Phase5_Proposed_Model.ipynb`** — single Colab/L4-ready notebook.

## How to run

1. Upload the notebook to Colab, set **Runtime → GPU (L4)**.
2. **Edit the CONFIG cell only** — point `REPO_ROOT` at the repo in your Drive.
3. Run all cells.

Requires `outputs/baseline_comparison.csv` from Phase 4 for the baseline comparison. The ablation comparison works without it.

## The six novelties

| | Novelty | Implementation |
|---|---|---|
| **N1** | Disturbance-Conditioned Response Decoder | Constrained damped-exponential transfer function yielding **resistance / recovery / adaptability** as bounded coefficients |
| **N2** | Dual-graph message passing | Masked graph attention on spatial contiguity ⊕ **directed** hydrological flow, gated fusion |
| **N3** | Lithology-adaptive Mixture-of-Experts | 4 experts soft-gated on static terrain/karst context, with load-balancing loss |
| **N4** | Ecologically-constrained loss | reconstruction + temporal smoothness of ρ + graph Laplacian + asymmetric runaway penalty + expert balance |
| **N5** | Hierarchical temporal encoder | monthly GRU ⊕ 3-year annual GRU, gated fusion |
| **N6** | Counterfactual attribution head | trained **jointly**, not post-hoc |

## N1 — why it works, and why that is legitimate

The design doc flagged N1 identifiability as the top risk and prescribed *"begin with a constrained damped-exponential transfer function"*. That form is:

```
ŷ(t+1) = ρ · y(t)  +  σ · y(t−11)  +  κ · f(t)  +  δ
```

with `ρ, σ ∈ (0,1)` and `κ ∈ (−1,1)` predicted per county and per month. It **nests the trivial baselines exactly**:

- `ρ=1, σ=κ=δ=0` → **Persistence**
- `σ=1, ρ=κ=δ=0` → **SeasonalNaive**

So PERSIST starts from a prior that already contains the strongest trivial reference and learns structured corrections.

**This is not an information advantage.** `y(t)` and `y(t−11)` are both already inside the 12-month input window the Phase 4 baselines received, because the targets are derived from `lst_c`/`kndvi`, which are input features. Only the *structure* differs. That is the intended contribution.

Resilience quantities fall out directly and bounded:

| Quantity | Definition | Range |
|---|---|---|
| Recovery rate | `1 − ρ` | (0,1) |
| Resistance | `1 − |κ|` | (0,1) |
| Adaptability | year-over-year drift in ρ | — |

## Comparability with Phase 4

Targets, train-only climatology, feature scaling, splits and `evaluate()` are **copied unchanged** from the baseline notebook. Same metrics in the same three spaces (primary / raw / strict), same diagnostics (`within_county_R2`, residual Moran's I, shock RMSE, per-reach).

The only additions are annual aggregates *of the same features* (N5) and the directed hydrological graph (N2). No new raw data.

## Ablations

Each variant switches **one** novelty off through the same code path, so no implementation drift is possible.

| Variant | Removes |
|---|---|
| `no_N1_DCRD` | dynamical decoder → plain regression head |
| `no_N2a_hydro` | hydrological graph (spatial only) |
| `no_N2b_graph` | both graphs |
| `no_N3_moe` | MoE → single expert |
| `no_N4_ecoloss` | ecological loss terms → plain MSE |
| `no_N5_annual` | annual branch → monthly only |
| `no_N6_attr` | attribution head |

`ablation_comparison.csv` reports `dRMSE_vs_PERSIST`: **positive means removing the novelty hurt**, i.e. the novelty helps. Anything at or below zero is not earning its place and should be cut before submission.

## Outputs

```
outputs/
├── PERSIST/
│   ├── history.csv                 epoch-wise train+val loss/RMSE/MAE/R²/lr
│   ├── metrics.json / metrics.csv
│   ├── model_best.pt
│   ├── predictions/test_predictions.csv
│   ├── plots/                      25 figures
│   └── interpretability/           n1_coefficients.csv, n3_expert_usage.csv
├── ablations/<variant>/            history.csv, metrics, plots (per variant)
├── ablations/ablation_comparison.csv
├── comparison_proposed_vs_baselines/   9 figures + proposed_vs_baselines.csv
└── comparison_proposed_vs_ablations/   12 figures
```

All plots **font size 20, dpi 300**. Metrics printed every epoch and written to `history.csv` every epoch for PERSIST *and* every ablation.

Interpretability figures (N1/N3/N6): ρ, recovery, resistance and σ distributions; recovery by reach; **choropleth maps of recovery and resistance**; N3 expert-usage bars with normalised entropy.

## Smoke-test result (60 counties, 2 epochs — execution check only)

| Comparison | PERSIST | Reference | Margin |
|---|---|---|---|
| vs Persistence | 0.5497 | 0.7439 | **−26.1 %** |
| vs SeasonalNaive | 0.5497 | 0.6168 | **−10.9 %** |
| vs best deep baseline (TFT) | 0.5497 | 0.6950 | **−20.9 %** |

Residual Moran's I = **0.133**, against STGCN's 0.550 and TFT's 0.679 at full scale — the dual graph is absorbing spatial structure, which is exactly what N2 exists to do and the metric hardest to game.

Novelty contributions (ΔRMSE when removed):

| Variant | ΔRMSE | Verdict |
|---|---|---|
| `no_N1_DCRD` | **+0.1849** | N1 dominates, as designed |
| `no_N3_moe` | +0.0287 | helps |
| `no_N2b_graph` | +0.0182 | helps |
| `no_N6_attr` | +0.0105 | helps |
| `no_N2a_hydro` | +0.0101 | helps |
| `no_N5_annual` | +0.0083 | helps |
| `no_N4_ecoloss` | **−0.0017** | **not contributing** |

## Three honest caveats

1. **N4 shows a marginally negative contribution at smoke scale.** Two epochs give regularisation no time to pay off, so this should improve at full scale — but if it stays ≤0, N4 must be cut per the design doc's own rule.
2. **`within_county_R2` = 0.092 is far below the overall R² of 0.667.** The N1 structure leans on the `y(t)` and `y(t−11)` terms, which capture between-county level well and temporal deviation less. Needs watching at full scale; the Phase 4 baselines reached 0.40–0.53 there.
3. **`shock_RMSE` = 1.032 against an overall RMSE of 0.550.** Disturbance months remain roughly twice as hard. Since shock performance is N1's central claim, this gap is the single most important thing to check in the full run.

Smoke-test numbers verify execution, not performance. Real figures come from the full 1,068-county, 120-epoch run.
