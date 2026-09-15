# Phase 2 — Research Gap Analysis and Proposed Model

## PERSIST

**P**erturbation-response **S**patiotemporal **I**nference over **S**easonal and **T**opological structure

*A dynamical-systems deep learning framework for county-level ecological resilience assessment in the Yangtze River Economic Belt.*

**Status: design finalised.** Nothing implemented yet. Data acquisition resumes from the requirements in §8.

---

## 0. Finalised decisions

| Decision | Resolution |
|---|---|
| Novelties | **All six retained** (N1–N6) |
| Name | **PERSIST** |
| Study period | **2000–2020** (literature-comparable; matches Li et al. 2026) |
| Spatial unit | **County**, 1,068 units, official GB/T 2260 adcodes |
| Socioeconomic scope | **Two-tier.** RS proxies mandatory and unblocked; yearbook panel optional enhancement (§5) |
| Training framing | **Three-signal**: self-supervised reconstruction + held-out forecasting + external disturbance validation. **No hand-built index target** (§6) |

---

## 1. Gap analysis

Two independent evidence sources: the 42-paper survey (Phase 1) and measurements from our own extraction (Phase 3).

### 1.1 From the literature

| ID | Gap | Evidence |
|---|---|---|
| **G1** | **The 8/8 cell is empty.** No study is simultaneously YREB-scale, resilience-targeted, RS-driven and DL-based. Max observed 6/8. | Belt-scale resilience work [1,2,4,6] uses no DL; DL work [24,25,30,34–39,41] targets quality / land use / NDVI, never resilience. |
| **G2** | **Spatial autocorrelation confirmed but never modelled.** | [15] statistically confirms ERI autocorrelation, then fits a per-unit tree ensemble. [35],[38] show the means but for land use. |
| **G3** | **Indicator weighting analyst-specified throughout** (entropy, AHP, PCA, TOPSIS). | Only [22] stress-tests weighting; only [29] reduces objectively, and as decoupled preprocessing. |
| **G4** | **Uniform index applied across heterogeneous lithology.** | [28] establishes karst requires its own formulation; YREB upstream is karstic, downstream alluvial. |
| **G5** | **No ecologically-constrained DL losses.** | Only [37] (asymmetric ECOLOSS), mangrove extent, one coastal site. |
| **G6** | **Terrestrial and aquatic resilience modelled separately; gradients disagree.** | [1] east-over-west terrestrial; [3] downstream–upstream–midstream for water. Same study area. |
| **G7** | **Interpretability always post-hoc.** | [7,12,15,26,27] all apply SHAP over tree ensembles after the fact. |

### 1.2 From our own data — stronger, because measured

| ID | Finding | Consequence for design |
|---|---|---|
| **F1** | **Resilience is not vegetation-driven.** kNDVI, NPP, ET and NDVI all run upstream > downstream — the *inverse* of the published resilience gradient. | A vegetation-dominated index cannot reproduce the published pattern. The human/adaptive-capacity stream is load-bearing, not decorative. |
| **F2** | **Annual aggregation destroys the recovery signal.** 2013 was a record Yangtze heatwave: LST captured it (23.18 °C, highest of 2010–2016) but kNDVI showed **no dip**. | Recovery *is* resilience. Any annual-resolution model is structurally incapable of estimating it. Seasonal resolution is mandatory. |
| **F3** | **Missingness concentrates in the most urbanised counties.** MOD16 ET is undefined over impervious surface; 黄浦区 has 4 valid pixels. | Missingness must be modelled, not deleted — dropping rows biases the east–west comparison being tested. |

### 1.3 The central methodological problem

Every paper in the corpus computes a composite index via fixed weighting, then analyses that index. Computing ER by entropy weighting and training a network to predict it **from the same indicators** means the network learns the weighting formula: R² ≈ 0.99, scientific content zero.

**PERSIST's foundational commitment is to never regress on a hand-built index.**

---

## 2. Design principle

> Ecological resilience is a **dynamical property**, not a state variable. Holling (1973) defines it as the capacity to absorb disturbance and recover. It can therefore be estimated only from observed **perturbation–response behaviour over time** — never from a weighted sum of static indicators.

Everything below follows from this single reframing.

---

## 3. Architecture

### Inputs per county *i*, year *t*

| Stream | Contents | Cadence |
|---|---|---|
| **State** | kNDVI, EVI, NPP, ET, LST day/night, tasseled-cap wetness | seasonal (4/yr) |
| **Forcing** | precipitation anomaly, temperature anomaly, SPEI drought index, flood indicator | seasonal |
| **Static context** | elevation, relief, slope, roughness, **karst fraction**, land-cover composition | fixed |
| **Human (Tier 1)** | nighttime lights, impervious fraction, population density, gridded GDP | annual |
| **Human (Tier 2, optional)** | industrial structure, environmental investment, fiscal capacity | annual |
| **Relational** | queen contiguity graph; **directed** hydrological flow graph | fixed |

### Pipeline

```
        seasonal STATE + seasonal FORCING
                      │
   [N5] Hierarchical Temporal Encoder
        intra-annual seasonal transformer  ──►  inter-annual encoder (21 yr)
                      │
   [N1] Disturbance-Conditioned Response Decoder          ◄── the core
        forcing-stream ─┐
                        ├─► learned transfer function ─►  RESISTANCE
        state-stream ───┘                                  RECOVERY
                                                           ADAPTABILITY
                      │
   [N3] Lithology-Adaptive Gating  (Mixture-of-Experts)
        karst │ alluvial │ mountain │ urban
                      │
   [N2] Dual-Graph Message Passing
        G_spatial (undirected, 3,032 edges)  ⊕  G_hydro (directed, upstream→down)
                      │
   [N6] Counterfactual Attribution Head   (trained jointly, not post-hoc)
                      │
        RESILIENCE  +  driver attributions  +  regime map

   all trained under  [N4] Ecologically-Constrained Composite Loss
```

---

## 4. The six novelties

### N1 · Disturbance-Conditioned Response Decoder — *core contribution*

Two encoders: **forcing** (climate anomalies) and **state** (ecological indicators). A learned transfer function maps forcing to state response. The three resilience components are **derived quantities**, never assigned:

| Component | Model definition |
|---|---|
| **Resistance** | inverse sensitivity `(∂response/∂forcing)⁻¹` — how little the system deviates under a given shock |
| **Recovery** | learned decay rate returning the response to baseline (bounded autoregressive coefficient) |
| **Adaptability** | drift in the transfer-function parameters across years — does the response function itself improve? |

**Closes:** G3, G1, and the circularity trap.
**Defence:** first operationalisation of Holling's definition as a learned dynamical system rather than an indicator composite. Resistance and recovery become *measured*, not stipulated. In systems-theory terms the transfer function is an impulse-response function, which gives the construct a principled basis.
**Risk:** highest of the six — see §7.

### N2 · Dual-Graph Message Passing (terrestrial ⊕ hydrological)

Separate attention heads over two topologies, then learned fusion:
- **G_spatial** — queen contiguity, undirected. **Already built: 1,068 nodes, 3,032 edges, mean degree 5.68, fully connected.**
- **G_hydro** — directed upstream→downstream river-network edges. Asymmetric by construction: upstream conditions propagate downstream, not the reverse.

**Closes:** G2, G6.
**Defence:** provides a *mechanistic explanation* for the [1]-vs-[3] gradient contradiction — terrestrial and riverine resilience propagate through **different topologies**, so no single spatial structure can reproduce both. No prior study models either graph. Resolving a published contradiction is a stronger contribution than merely adding a module.

### N3 · Lithology-Adaptive Indicator Gating (Mixture-of-Experts)

A gating network conditioned on static context (karst fraction, elevation, slope, land-cover mix) produces soft weights over regime experts — **karst / alluvial / mountain / urban** — each learning its own indicator response.

**Closes:** G4.
**Defence:** [28] proves karst needs distinct index formulation; the belt spans karst upstream and alluvium downstream yet is universally assessed with one index. Soft gating degrades gracefully and produces an interpretable **regime map** as a secondary deliverable.

### N4 · Ecologically-Constrained Composite Loss

| Term | Purpose |
|---|---|
| `L_recon` | indicator reconstruction — **self-supervised; removes the need for any hand-built label** |
| `L_forecast` | predict next season / next year — genuine held-out signal, source of reportable metrics |
| `L_bound` | resistance ≥ 0; recovery rate ∈ (0,1) for dynamical stability |
| `L_smooth` | temporal smoothness — resilience cannot jump discontinuously year to year |
| `L_asym` | asymmetric penalty on ecologically implausible runaway trends |
| `L_lap` | graph-Laplacian regularisation with **learned** strength, not fixed |

**Closes:** G5.
**Defence:** generalises [37]'s ECOLOSS from mangrove extent at one site to a composite resilience target at regional scale, and adds dynamical-stability bounds absent there.

### N5 · Hierarchical Seasonal→Annual Temporal Encoder

Level 1 encodes within-year seasonal dynamics (where disturbance and recovery actually occur); Level 2 encodes the 21-year trajectory.

**Closes:** F2.
**Defence — the strongest empirical justification in the proposal:** necessity is *demonstrated from our own data*, not argued. 2013's record heatwave is invisible in annual kNDVI yet plain in LST. A concrete, reproducible motivating figure for the paper.

### N6 · Counterfactual Attribution Head (intrinsic, not post-hoc)

Attribution trained **jointly** with the model, yielding per-county per-year driver contributions consistent with the predictor **by construction**, and supporting counterfactual queries ("had construction land not expanded, what resilience?") without retraining.

**Closes:** G7.
**Defence:** differentiates against the entire XGBoost-SHAP strand [7,12,15,26,27], where attribution is a separate post-hoc approximation of a black box.

---

## 5. Human dimension — two-tier strategy

F1 makes the human stream load-bearing. But yearbook data requires institutional access and its missingness is systematically western and poor, which would bias the east–west finding under test.

| Tier | Contents | Availability | Role |
|---|---|---|---|
| **1 — RS proxies** | nighttime lights (harmonised DMSP/VIIRS), impervious surface, population grids, gridded GDP | **fully open** | **Mandatory.** Carries the human stream. |
| **2 — Yearbook panel** | industrial structure, environmental investment, fiscal capacity | institutional access | **Optional enhancement.** |

Tier 1 works because nighttime lights and impervious surface are established proxies for development and urbanisation, are gridded (clean county aggregation), and have **no missingness problem**.

**Consequence: the socioeconomic dimension is off the critical path.** PERSIST is fully buildable from open data. Tier 2 strengthens N1's adaptability component if obtained; otherwise it is a stated limitation.

---

## 6. Training framing — three signals, no index target

| Signal | Provides |
|---|---|
| **Self-supervised reconstruction** | Latent ecological state with **no arbitrary weighting** — escapes circularity |
| **Held-out forecasting** | **Legitimate RMSE / MAE / R²**, directly comparable to DRSEI, XGBoost-SHAP and Bayesian baselines. Not circular: the target is genuinely unseen future data |
| **External disturbance validation** | AUC / F1 against 2006, 2011, 2013 droughts and 2016, 2020 floods |

Used for **validation only, never as a training target**: concordance with published resilience values from [1].

This yields publishable hard numbers *and* an index nobody hand-weighted. The forecasting task is what permits an honest claim of performance superiority.

---

## 7. Honest risk assessment

Reviewers penalise architectures that stack modules to inflate novelty count. **Every component must earn its place in the ablation table or be removed before submission.**

| Risk | Severity | Mitigation |
|---|---|---|
| **Sample size.** 1,068 × 21 = 22,428 county-years is small for DL; XGBoost may win outright | **High** | Seasonal resolution → 89,712 county-seasons; graph exploits relational structure not sample count; pixel-level self-supervised pretraining then county fine-tuning; heavy regularisation |
| **N1 identifiability.** Separating forcing from response may be ill-posed | **High** | Begin with a constrained damped-exponential transfer function before permitting free-form; validate against known drought years |
| **N3 expert collapse** to a single expert | Medium | Load-balancing loss; report expert-utilisation entropy |
| **GPU availability unverified** in sandbox | Medium | Model is small and CPU-feasible at county scale. Must confirm before any patch-based variant |
| **Metric inflation.** [34] reports 99.32 % on a class-imbalanced task | Medium | Never report aggregate accuracy. Change-sensitive, class-balanced metrics; split by **time and space**, never randomly |
| **Novelty dilution** — six components | Medium | Ablations decide; ship only demonstrated contributors |
| **Nighttime light discontinuity** | Medium | Harmonised DMSP/VIIRS product; verify no false 2013 break before use |

---

## 8. Data requirements implied by this design

This is why the model was designed first: it dictates the data.

### Already held
1,068-county boundaries · queen contiguity graph (3,032 edges) · annual kNDVI/EVI/NDVI, LST day/night, NPP, ET · terrain (elevation, relief, slope, roughness) — 2000–2020, externally validated.

### Newly required, in priority order

| # | Need | Driven by | Source | Blocker |
|---|---|---|---|---|
| 1 | **Seasonal composites** — kNDVI, EVI, LST, wetness | **N5, N1** | MPC (proven) | none |
| 2 | **Climate forcing + SPEI** | **N1** — forcing stream is mandatory | MPC `terraclimate`, `era5-pds` | none |
| 3 | **Karst extent** | **N3** | WOKAM / carbonate outcrop maps | none expected |
| 4 | **Directed river network** | **N2** | HydroSHEDS flow direction | `hydrosheds.org` returns **403** — need alternative host |
| 5 | **Harmonised nighttime lights** | Tier 1 human | harmonised DMSP/VIIRS | must avoid false 2013 break |
| 6 | **Impervious surface + population + GDP grids** | Tier 1 human | GAIA/GISA, WorldPop/GHS-POP | none expected |
| 7 | **Land cover (CLCD)** | **N3** regimes + fragmentation | Zenodo `18180184` | large download |
| 8 | **Disturbance event catalogue** | validation | literature + records | manual compilation |
| 9 | Yearbook socioeconomic panel | Tier 2 | County yearbooks | institutional access — **optional** |

---

## 9. Evaluation plan

### Baselines (Phase 4)

| Baseline | Source | Question it settles |
|---|---|---|
| **DRSEI** (autoencoder + LSTM) | [24] Gong et al. 2025, 25 citations | Does spatial structure add anything beyond temporal? |
| **XGBoost-SHAP** | [7,12,15,26,27] | Does DL beat tree ensembles at this sample size? **The honest hurdle.** |
| **Bayesian spatiotemporal (INLA) / GWR** | [13],[8] | Does a learned model beat rigorous classical spatial statistics? |

### Metrics

- **Forecasting** — RMSE / MAE / R² on held-out years 2018–2020
- **Spatial transfer** — train downstream + midstream, test upstream (directly tests N3)
- **Disturbance detection** — AUC / F1 against 2006, 2011, 2013, 2016, 2020
- **Residual Moran's I** → should approach 0 if N2 genuinely captures spatial structure. *Cleanest single test of the graph, and harder to game than R².*
- **Concordance with [1]** — external validity
- **Expert-utilisation entropy** — confirms N3 has not collapsed

### Ablations (Phase 5)

| Variant | Isolates |
|---|---|
| Full PERSIST | — |
| − DCRD (regress an index directly) | **N1** |
| − hydrological graph | **N2a** |
| − spatial graph (MLP only) | **N2b** |
| − MoE gating (single expert) | **N3** |
| − ecological loss terms (plain MSE) | **N4** |
| − seasonal level (annual only) | **N5** |
| − attribution head | **N6** |

---

## 10. Minimum viable version

If scope must be cut, the defensible core is **N1 + N2 + N5** — a disturbance-response decoder on a dual graph at seasonal resolution. That alone closes G1, G2, G3, G6 and F2 and is novel against the entire corpus. N3, N4 and N6 are severable enhancements.

---

## 11. Immediate next steps (Phase 3 resumes here)

1. Seasonal compositing of kNDVI, EVI, LST, wetness — 2000–2020
2. Climate forcing + SPEI from TerraClimate / ERA5
3. Resolve the HydroSHEDS 403 and build the directed hydrological graph
4. Karst extent layer
5. Harmonised nighttime lights, with an explicit 2013 discontinuity check
6. CLCD land cover → regime definitions + fragmentation metrics
7. EDA: Moran's I, VIF, changepoint test on ET, minimum-pixel QC thresholds
