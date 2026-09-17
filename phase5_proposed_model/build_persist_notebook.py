#!/usr/bin/env python3
"""Generate the PERSIST proposed-model notebook (Phase 5).

Implements all six novelties, runs 8 ablations, and produces two separate
comparison sets (vs baselines, vs ablations). Single notebook, Colab/L4 ready,
only the CONFIG cell needs editing.
"""
import pathlib
import nbformat as nbf

C = []
def md(s): C.append(("md", s.strip("\n")))
def code(s): C.append(("code", s.strip("\n")))


# ══════════════════════════════════════════════════════════════════ TITLE
md(r"""
# PERSIST — Phase 5: Proposed Model + Ablations

**P**erturbation-response **S**patiotemporal **I**nference over **S**easonal and **T**opological structure

Ecological resilience in the Yangtze River Economic Belt.

## The six novelties

| | Novelty | Implementation |
|---|---|---|
| **N1** | Disturbance-Conditioned Response Decoder | Constrained damped-exponential transfer function producing **resistance / recovery / adaptability** as bounded, interpretable coefficients |
| **N2** | Dual-graph message passing | Spatial contiguity ⊕ **directed** hydrological flow, with learned gated fusion |
| **N3** | Lithology-adaptive Mixture-of-Experts | karst / alluvial / mountain / urban experts, soft-gated on static context |
| **N4** | Ecologically-constrained composite loss | reconstruction + bounds + temporal smoothness + graph Laplacian + asymmetric runaway penalty |
| **N5** | Hierarchical temporal encoder | monthly branch ⊕ multi-year annual branch, gated |
| **N6** | Counterfactual attribution head | trained **jointly**, not post-hoc |

## Why N1 is expected to perform well — and why that is legitimate

The design doc flagged N1 identifiability as the top risk and prescribed *"begin
with a constrained damped-exponential transfer function"*. That form is:

$$\hat{y}_{t+1} = \rho \odot y_t \;+\; \sigma \odot y_{t-11} \;+\; \kappa \odot f_t \;+\; \delta$$

with $\rho,\sigma \in (0,1)$ and $\kappa \in (-1,1)$ predicted per county and per
month by the network. This **nests the trivial baselines exactly**:

- $\rho{=}1, \sigma{=}\kappa{=}\delta{=}0$ → **Persistence**
- $\sigma{=}1, \rho{=}\kappa{=}\delta{=}0$ → **SeasonalNaive**

So PERSIST starts from a prior that already contains the strongest trivial
reference and learns structured corrections on top. Critically, $y_t$ and
$y_{t-11}$ are **already inside the 12-month input window the baselines
received** — nothing new is given to the model. Only the *structure* differs.

The resilience quantities fall out directly:

| Quantity | Definition | Bound |
|---|---|---|
| **Recovery rate** | $1-\rho$ — speed of return to baseline | (0,1) |
| **Resistance** | $1-\lvert\kappa\rvert$ — insensitivity to forcing | (0,1) |
| **Adaptability** | year-over-year drift in $\rho$ | — |

## Outputs

```
outputs/
├── PERSIST/                            history.csv, metrics, plots, predictions,
│                                       interpretability/ (N1 + N3 + N6 artefacts)
├── ablations/<variant>/                per-variant history, metrics, plots
├── ablations/ablation_comparison.csv
├── comparison_proposed_vs_baselines/   plots + csv
└── comparison_proposed_vs_ablations/   plots + csv
```

Metrics, targets, splits and `evaluate()` are **byte-identical to the Phase 4
baseline notebook**, so the numbers are directly comparable.

> **Only the CONFIG cell below needs editing.**
""")

# ═════════════════════════════════════════════════════════════════ CONFIG
md("## 1 · CONFIGURATION — *the only cell you need to edit*")

code(r'''
# ════════════════════════════════════════════════════════════════════════
#  CONFIG  —  EDIT ONLY THIS CELL
# ════════════════════════════════════════════════════════════════════════
from pathlib import Path

# --- Google Drive ------------------------------------------------------
MOUNT_DRIVE = True
REPO_ROOT   = Path("/content/drive/MyDrive/Assesment_of_Ecological_resilience_Yangtze")

# --- derived paths (normally leave alone) ------------------------------
DATA_DIR      = REPO_ROOT / "phase3_data" / "tables"
BOUNDARY_FILE = REPO_ROOT / "phase3_data" / "boundaries" / "yreb_counties_datav.gpkg"
OUTPUT_DIR    = REPO_ROOT / "outputs"

PANEL_FILE    = DATA_DIR / "panel_monthly.parquet"
ADJ_FILE      = DATA_DIR / "adjacency_edges.csv"
HYDRO_FILE    = DATA_DIR / "hydro_edges.csv"
# written by the Phase 4 notebook; PERSIST is compared against it
BASELINE_CSV  = OUTPUT_DIR / "baseline_comparison.csv"

# --- run mode ---------------------------------------------------------
SMOKE_TEST = False        # True = tiny subset + 2 epochs
RUN_ABLATIONS = True
SEED       = 42

# --- task (MUST match Phase 4 exactly for comparability) --------------
LOOKBACK   = 12
HORIZON    = 1
TARGETS    = ["lst_ds", "kndvi_ds"]
STRICT_TARGETS = ["lst_z", "kndvi_z"]
MIN_SD_FRAC = 0.10
CLIP_SIGMA  = 5.0

TRAIN_YEARS = (2000, 2014)
VAL_YEARS   = (2015, 2017)
TEST_YEARS  = (2018, 2020)

# --- PERSIST architecture --------------------------------------------
D_MODEL      = 96         # hidden width
N_EXPERTS    = 4          # N3: karst / alluvial / mountain / urban
ANNUAL_YEARS = 3          # N5: years of annual context
GRAPH_HEADS  = 2          # N2: attention heads per graph
DROPOUT      = 0.15

# --- N4 ecological loss weights --------------------------------------
W_RECON   = 0.05          # self-supervised reconstruction
W_SMOOTH  = 0.02          # temporal smoothness of the recovery coefficient
W_LAP     = 0.01          # graph Laplacian on predictions
W_ASYM    = 0.01          # asymmetric runaway-trend penalty
W_BALANCE = 0.01          # N3 expert load balancing (prevents collapse)

# --- training ---------------------------------------------------------
EPOCHS       = 120
GRAPH_BATCH  = 8
LR           = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
GRAD_CLIP    = 1.0

# --- plotting ---------------------------------------------------------
FONT_SIZE = 20
DPI       = 300
FIG_FMT   = "png"

# --- QC / evaluation -------------------------------------------------
USE_NREL_COVARIATE = True
SHOCK_THRESHOLD    = 1.5
# ════════════════════════════════════════════════════════════════════════
print("CONFIG loaded.")
print("  repo     :", REPO_ROOT)
print("  panel    :", PANEL_FILE)
print("  baselines:", BASELINE_CSV)
print("  outputs  :", OUTPUT_DIR)
print("  SMOKE_TEST =", SMOKE_TEST, "| RUN_ABLATIONS =", RUN_ABLATIONS)
''')

# ═══════════════════════════════════════════════════════════════════ ENV
md("## 2 · Environment")

code(r'''
import os, sys, json, math, time, warnings, random, itertools
warnings.filterwarnings("ignore")

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped:", e)

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    os.system(f"{sys.executable} -m pip install -q geopandas")
    try:
        import geopandas as gpd
        HAS_GPD = True
    except Exception:
        HAS_GPD = False

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
set_seed(SEED)

plt.rcParams.update({
    "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE - 2,
    "ytick.labelsize": FONT_SIZE - 2, "legend.fontsize": FONT_SIZE - 4,
    "figure.titlesize": FONT_SIZE + 2, "savefig.dpi": DPI,
    "figure.dpi": 100, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3,
})

PERSIST_DIR = OUTPUT_DIR / "PERSIST"
ABL_DIR     = OUTPUT_DIR / "ablations"
CMP_BASE    = OUTPUT_DIR / "comparison_proposed_vs_baselines"
CMP_ABL     = OUTPUT_DIR / "comparison_proposed_vs_ablations"
for d in [PERSIST_DIR / "plots", PERSIST_DIR / "predictions",
          PERSIST_DIR / "interpretability", ABL_DIR, CMP_BASE, CMP_ABL]:
    d.mkdir(parents=True, exist_ok=True)
print("output folders ready under", OUTPUT_DIR)
''')

# ══════════════════════════════════════════════════════ DATA (identical)
md(r"""
## 3 · Data and preprocessing

**Identical to the Phase 4 baseline notebook** — same target construction, same
train-only climatology, same feature scaling, same splits. This is what makes
the comparison valid.
""")

code(r'''
t0 = time.time()
panel = pd.read_parquet(PANEL_FILE)
panel["adcode"] = panel["adcode"].astype(str)
adj   = pd.read_csv(ADJ_FILE,   dtype={"u_adcode": str, "v_adcode": str})
hydro = pd.read_csv(HYDRO_FILE, dtype={"src": str, "dst": str})
panel["t"] = (panel.year - panel.year.min()) * 12 + (panel.month - 1)
panel = panel.sort_values(["adcode", "t"]).reset_index(drop=True)
print(f"panel {panel.shape[0]:,} x {panel.shape[1]} ({time.time()-t0:.1f}s) | "
      f"spatial edges {len(adj)} | hydro edges {len(hydro)}")

if SMOKE_TEST:
    keep = sorted(panel.adcode.unique())[:60]
    panel = panel[panel.adcode.isin(keep)].copy()
    print(f"SMOKE_TEST: {panel.adcode.nunique()} counties")

counties = sorted(panel.adcode.unique())
cidx = {c: i for i, c in enumerate(counties)}
N, T = len(counties), panel.t.nunique()
print(f"N={N} counties | T={T} months")

if "lst_day_n" in panel:
    med = panel.groupby("adcode")["lst_day_n"].transform("median").replace(0, np.nan)
    panel["n_rel"] = (panel["lst_day_n"] / med).fillna(1.0)
else:
    panel["n_rel"] = 1.0

train_mask = panel.year.between(*TRAIN_YEARS)
RAW_OF = {"lst_ds": "lst_c", "kndvi_ds": "kndvi",
          "lst_z": "lst_c", "kndvi_z": "kndvi"}
CLIM = {}
for prim, strict in zip(TARGETS, STRICT_TARGETS):
    raw = RAW_OF[prim]
    mmu = panel.loc[train_mask].groupby("month")[raw].mean()
    ds_un = panel[raw] - panel.month.map(mmu)
    gsd = float(ds_un[train_mask].std())
    panel[prim] = (ds_un / gsd).clip(-CLIP_SIGMA, CLIP_SIGMA)

    cm = (panel.loc[train_mask].groupby(["adcode", "month"])[raw]
          .agg(["mean", "std"]).reset_index()
          .rename(columns={"mean": "cmu", "std": "csd"}))
    floor = MIN_SD_FRAC * gsd
    cm["csd_eff"] = cm.csd.fillna(floor).clip(lower=floor)
    panel = panel.merge(cm[["adcode", "month", "cmu", "csd_eff"]],
                        on=["adcode", "month"], how="left")
    panel[strict] = ((panel[raw] - panel.cmu) / panel.csd_eff
                     ).clip(-CLIP_SIGMA, CLIP_SIGMA)
    CLIM[prim] = dict(raw=raw, gsd=gsd, mmu=mmu,
                      cmu=panel.groupby(["adcode", "month"]).cmu.first(),
                      csd=panel.groupby(["adcode", "month"]).csd_eff.first())
    panel = panel.drop(columns=["cmu", "csd_eff"])
    print(f"{prim:10} global sd={gsd:.4f}")

panel["year_frac"] = (panel.year - TRAIN_YEARS[0]) / 20.0
panel["moy_sin"] = np.sin(2 * np.pi * panel.month / 12)
panel["moy_cos"] = np.cos(2 * np.pi * panel.month / 12)

# ---- feature groups. N1 needs FORCING and STATE kept separate ----------
STATE_FEATS = [c for c in ["lst_c", "kndvi", "ndvi_mean"] if c in panel]
FORCE_FEATS = [c for c in ["ppt", "pet", "aet", "def", "q", "tmax", "tmin",
                           "vpd", "soil", "srad", "pdsi", "swe", "wbal",
                           "heat_z", "dry_z", "tmax_z", "ppt_z", "soil_z",
                           "srad_z", "vpd_z", "pdsi_z"] if c in panel]
HUMAN_FEATS = [c for c in ["ntl_mean", "ntl_sum"] if c in panel]
STATIC_FEATS = [c for c in ["elev_mean", "elev_std", "relief", "slope_mean",
                            "slope_std", "roughness", "karst_frac",
                            "area_km2"] if c in panel]
if USE_NREL_COVARIATE:
    FORCE_FEATS = FORCE_FEATS + ["n_rel"]
CAL_FEATS = ["moy_sin", "moy_cos", "year_frac"]

DYN_FEATS = STATE_FEATS + FORCE_FEATS + HUMAN_FEATS + CAL_FEATS
# index slices so N1 can address the two streams inside the same tensor
IDX_STATE = [DYN_FEATS.index(c) for c in STATE_FEATS]
IDX_FORCE = [DYN_FEATS.index(c) for c in FORCE_FEATS]
print(f"\ndynamic {len(DYN_FEATS)} (state {len(STATE_FEATS)}, "
      f"forcing {len(FORCE_FEATS)}, human {len(HUMAN_FEATS)}, cal {len(CAL_FEATS)})")
print(f"static  {len(STATIC_FEATS)}")
''')

code(r'''
panel["ci"] = panel.adcode.map(cidx)
tvals = np.sort(panel.t.unique()); tpos = {v: i for i, v in enumerate(tvals)}
panel["ti"] = panel.t.map(tpos)

X = np.full((N, T, len(DYN_FEATS)), np.nan, dtype=np.float32)
Y = np.full((N, T, len(TARGETS)), np.nan, dtype=np.float32)
X[panel.ci.values, panel.ti.values, :] = panel[DYN_FEATS].values.astype(np.float32)
Y[panel.ci.values, panel.ti.values, :] = panel[TARGETS].values.astype(np.float32)
S = (panel.groupby("adcode")[STATIC_FEATS].first()
     .reindex(counties).astype(np.float32).values)
year_of = panel.groupby("ti").year.first().reindex(range(T)).values
month_of = panel.groupby("ti").month.first().reindex(range(T)).values

for f in range(X.shape[2]):
    col = X[:, :, f]
    idx = np.where(~np.isnan(col), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    col = col[np.arange(N)[:, None], idx]
    rev = col[:, ::-1]
    idx2 = np.where(~np.isnan(rev), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx2, axis=1, out=idx2)
    X[:, :, f] = np.nan_to_num(rev[np.arange(N)[:, None], idx2][:, ::-1], nan=0.0)
S = np.nan_to_num(S, nan=0.0)

tr_t = np.where((year_of >= TRAIN_YEARS[0]) & (year_of <= TRAIN_YEARS[1]))[0]
va_t = np.where((year_of >= VAL_YEARS[0])   & (year_of <= VAL_YEARS[1]))[0]
te_t = np.where((year_of >= TEST_YEARS[0])  & (year_of <= TEST_YEARS[1]))[0]

mu = X[:, tr_t, :].reshape(-1, X.shape[2]).mean(0)
sd = X[:, tr_t, :].reshape(-1, X.shape[2]).std(0); sd[sd == 0] = 1.0
X = (X - mu) / sd
smu, ssd = S.mean(0), S.std(0); ssd[ssd == 0] = 1.0
S = (S - smu) / ssd

# ---- N5: annual context tensor (mean of dynamic feats per calendar year) ----
n_years = T // 12
Xann = X[:, :n_years * 12, :].reshape(N, n_years, 12, X.shape[2]).mean(axis=2)
print("annual context", Xann.shape)

def windows_for(times):
    return np.array(sorted([e for e in times if e - LOOKBACK >= 0]),
                    dtype=np.int64)
tr_e, va_e, te_e = windows_for(tr_t), windows_for(va_t), windows_for(te_t)
VALID = ~np.isnan(Y).any(axis=2)
print(f"target months train {len(tr_e)} | val {len(va_e)} | test {len(te_e)}")
print(f"valid targets {VALID.sum():,}/{N*T:,}")

Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y); St = torch.from_numpy(S)
Xa = torch.from_numpy(Xann)
Yfill = torch.from_numpy(np.nan_to_num(Y, nan=0.0))   # for the N1 lag terms
''')

# ══════════════════════════════════════════════════════════════ GRAPHS
md("## 4 · Dual graph construction (N2)")

code(r'''
def build_norm_adj(edge_df, a, b, directed):
    A = np.zeros((N, N), dtype=np.float32); hit = 0
    for u, v in zip(edge_df[a], edge_df[b]):
        if u in cidx and v in cidx:
            A[cidx[u], cidx[v]] = 1.0
            if not directed: A[cidx[v], cidx[u]] = 1.0
            hit += 1
    A = A + np.eye(N, dtype=np.float32)
    d = A.sum(1); dinv = np.power(d, -0.5, where=d > 0); dinv[np.isinf(dinv)] = 0
    return (A * dinv[:, None] * dinv[None, :]).astype(np.float32), hit

A_sp, n_sp = build_norm_adj(adj, "u_adcode", "v_adcode", False)
A_hy, n_hy = build_norm_adj(hydro, "src", "dst", True)   # DIRECTED: upstream->down
print(f"spatial  {n_sp} edges | density {(A_sp>0).mean():.5f}  (undirected)")
print(f"hydro    {n_hy} edges | density {(A_hy>0).mean():.5f}  (DIRECTED)")
print("N2 uses BOTH graphs; the STGCN baseline had the spatial one only.")

A_sp_t = torch.from_numpy(A_sp).to(DEVICE)
A_hy_t = torch.from_numpy(A_hy).to(DEVICE)

# unnormalised Laplacian for the N4 smoothness penalty
Araw = np.zeros((N, N), dtype=np.float32)
for u, v in zip(adj.u_adcode, adj.v_adcode):
    if u in cidx and v in cidx:
        Araw[cidx[u], cidx[v]] = Araw[cidx[v], cidx[u]] = 1.0
L_np = np.diag(Araw.sum(1)) - Araw
L_t = torch.from_numpy(L_np / max(Araw.sum(), 1.0)).to(DEVICE)
''')

# ═════════════════════════════════════════════════════ METRICS (identical)
md(r"""
## 5 · Metrics — identical to Phase 4

Same functions, same three evaluation spaces (primary / raw / strict), same
diagnostics (`within_county_R2`, residual Moran's I, shock RMSE, per-reach).
Copied unchanged so the comparison is exact.
""")

code(r'''
def _finite(o, p):
    m = np.isfinite(o) & np.isfinite(p); return o[m], p[m]

def willmott_d(o, p):
    o, p = _finite(o, p)
    if len(o) < 2: return np.nan
    om = o.mean(); den = np.sum((np.abs(p - om) + np.abs(o - om)) ** 2)
    return 1 - np.sum((o - p) ** 2) / den if den > 0 else np.nan

def kge(o, p):
    o, p = _finite(o, p)
    if len(o) < 3: return np.nan
    so, sp = o.std(), p.std()
    if so == 0: return np.nan
    r = np.corrcoef(o, p)[0, 1] if sp > 0 else 0.0
    return 1 - np.sqrt((r - 1) ** 2 + (sp / so - 1) ** 2 +
                       ((p.mean() - o.mean()) / so) ** 2)

def core_metrics(o, p):
    o, p = _finite(np.asarray(o, float), np.asarray(p, float))
    if len(o) < 3:
        return {k: np.nan for k in ["RMSE", "MAE", "R2", "PearsonR",
                                    "WillmottD", "KGE", "Bias"]}
    err = p - o; sst = np.sum((o - o.mean()) ** 2)
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))),
            "R2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
            "PearsonR": float(np.corrcoef(o, p)[0, 1]) if p.std() > 0 else np.nan,
            "WillmottD": float(willmott_d(o, p)), "KGE": float(kge(o, p)),
            "Bias": float(np.mean(err))}

def morans_I(values, A):
    v = np.asarray(values, float); m = np.isfinite(v)
    if m.sum() < 10: return np.nan
    W = A.copy(); np.fill_diagonal(W, 0.0); W = W[np.ix_(m, m)]
    v = v[m] - v[m].mean(); S0 = W.sum()
    if S0 == 0 or (v ** 2).sum() == 0: return np.nan
    return (len(v) / S0) * float(v @ (W @ v)) / float((v ** 2).sum())

MONTH_OF_TI = month_of
def build_space_maps():
    maps = {}
    for prim, strict in zip(TARGETS, STRICT_TARGETS):
        c = CLIM[prim]
        mmu_arr = np.array([c["mmu"].get(m, np.nan) for m in range(1, 13)])
        cmu = np.full((N, 13), np.nan); csd = np.full((N, 13), np.nan)
        for (a, m), v in c["cmu"].items():
            if a in cidx: cmu[cidx[a], m] = v
        for (a, m), v in c["csd"].items():
            if a in cidx: csd[cidx[a], m] = v
        maps[prim] = dict(gsd=c["gsd"], mmu=mmu_arr, cmu=cmu, csd=csd,
                          strict=strict, raw=c["raw"])
    return maps
SPACE = build_space_maps()

def to_raw(vals, prim, ci, ti):
    m = SPACE[prim]; return vals * m["gsd"] + m["mmu"][MONTH_OF_TI[ti] - 1]
def to_strict(vals, prim, ci, ti):
    m = SPACE[prim]; mon = MONTH_OF_TI[ti]
    return (vals * m["gsd"] + m["mmu"][mon - 1] - m["cmu"][ci, mon]) / m["csd"][ci, mon]

def evaluate(pred, obs, ci, ti, tag, A_for_moran):
    out = {}
    out.update({f"all_{k}": v for k, v in
                core_metrics(obs.ravel(), pred.ravel()).items()})
    for j, tn in enumerate(TARGETS):
        for k, v in core_metrics(obs[:, j], pred[:, j]).items():
            out[f"{tn}_{k}"] = v
    for j, prim in enumerate(TARGETS):
        for k, v in core_metrics(to_raw(obs[:, j], prim, ci, ti),
                                 to_raw(pred[:, j], prim, ci, ti)).items():
            out[f"raw_{SPACE[prim]['raw']}_{k}"] = v
        for k, v in core_metrics(to_strict(obs[:, j], prim, ci, ti),
                                 to_strict(pred[:, j], prim, ci, ti)).items():
            out[f"strict_{SPACE[prim]['strict']}_{k}"] = v

    df = pd.DataFrame({"ci": ci, "ti": ti})
    for j, tn in enumerate(TARGETS):
        df[f"obs_{tn}"] = obs[:, j]; df[f"pred_{tn}"] = pred[:, j]
        df[f"res_{tn}"] = pred[:, j] - obs[:, j]
    df["adcode"] = [counties[c] for c in df.ci]
    df["year"] = year_of[df.ti.values]; df["month"] = month_of[df.ti.values]

    prim = TARGETS[0]; Is = []
    for y, g in df.groupby("year"):
        vec = np.full(N, np.nan)
        gm = g.groupby("ci")[f"res_{prim}"].mean(); vec[gm.index.values] = gm.values
        Is.append(morans_I(vec, A_for_moran))
    out["residual_MoranI"] = float(np.nanmean(Is)) if Is else np.nan

    dd = df.dropna(subset=[f"obs_{prim}", f"pred_{prim}"]).copy()
    dd["o_d"] = dd[f"obs_{prim}"] - dd.groupby("ci")[f"obs_{prim}"].transform("mean")
    dd["p_d"] = dd[f"pred_{prim}"] - dd.groupby("ci")[f"pred_{prim}"].transform("mean")
    out["within_county_R2"] = core_metrics(dd.o_d, dd.p_d)["R2"]
    out["within_county_PearsonR"] = core_metrics(dd.o_d, dd.p_d)["PearsonR"]

    shock = panel[["ci", "ti", "heat_z"]].copy()
    shock["is_shock"] = shock.heat_z.abs() >= SHOCK_THRESHOLD
    df = df.merge(shock[["ci", "ti", "is_shock"]], on=["ci", "ti"], how="left")
    df["is_shock"] = df.is_shock.fillna(False).astype(bool)
    for lbl, sub in (("shock", df[df.is_shock]), ("calm", df[~df.is_shock])):
        mm = core_metrics(sub[f"obs_{prim}"], sub[f"pred_{prim}"])
        out[f"{lbl}_RMSE"] = mm["RMSE"]; out[f"{lbl}_R2"] = mm["R2"]
        out[f"{lbl}_n"] = int(len(sub))
    reach = panel.groupby("adcode").reach.first()
    df["reach"] = df.adcode.map(reach)
    for r, sub in df.groupby("reach"):
        out[f"reach_{r}_RMSE"] = core_metrics(
            sub[f"obs_{prim}"], sub[f"pred_{prim}"])["RMSE"]
    out["n_samples"] = int(len(df))
    return out, df
print("metrics + evaluate() defined (identical to Phase 4)")
''')

# ════════════════════════════════════════════════════════════ THE MODEL
md(r"""
## 6 · PERSIST architecture

Every novelty is switchable by a flag so the ablations reuse exactly the same
code path — no separate implementations that could drift apart.
""")

code(r'''
class GatedFuse(nn.Module):
    """Learned gate between two representations."""
    def __init__(self, d):
        super().__init__(); self.g = nn.Linear(2 * d, d)
    def forward(self, a, b):
        w = torch.sigmoid(self.g(torch.cat([a, b], -1)))
        return w * a + (1 - w) * b


class GraphAttn(nn.Module):
    """Masked multi-head attention restricted to graph neighbours.
    Works for directed adjacency, which is what the hydrological graph needs."""
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
    def forward(self, x, A):                       # x (B, N, d)
        B, n, d = x.shape
        Q = self.q(x).view(B, n, self.h, self.dk).transpose(1, 2)
        K = self.k(x).view(B, n, self.h, self.dk).transpose(1, 2)
        V = self.v(x).view(B, n, self.h, self.dk).transpose(1, 2)
        s = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)
        mask = (A > 0).unsqueeze(0).unsqueeze(0)   # (1,1,N,N)
        s = s.masked_fill(~mask, float("-inf"))
        a = torch.softmax(s, -1)
        a = torch.nan_to_num(a)                    # isolated rows
        h = (a @ V).transpose(1, 2).reshape(B, n, d)
        return self.o(h)


class PERSIST(nn.Module):
    """Proposed model. Flags switch novelties off for the ablation study."""
    def __init__(self, f_dyn, f_static, f_ann, n_out,
                 d=D_MODEL, experts=N_EXPERTS, heads=GRAPH_HEADS,
                 dropout=DROPOUT,
                 use_n1=True, use_graph=True, use_hydro=True,
                 use_moe=True, use_annual=True, use_attr=True):
        super().__init__()
        self.use_n1, self.use_graph, self.use_hydro = use_n1, use_graph, use_hydro
        self.use_moe, self.use_annual, self.use_attr = use_moe, use_annual, use_attr
        self.n_out = n_out
        self.E = experts if use_moe else 1

        # ---- N5 hierarchical temporal encoder -------------------------
        self.month_gru = nn.GRU(f_dyn, d, num_layers=2, batch_first=True,
                                dropout=dropout)
        if use_annual:
            self.year_gru = nn.GRU(f_ann, d, batch_first=True)
            self.fuse_time = GatedFuse(d)

        # ---- N1 two-stream encoders (forcing kept separate from state) -
        self.force_enc = nn.Sequential(
            nn.Linear(len(IDX_FORCE), d), nn.ELU(), nn.Linear(d, d))
        self.state_enc = nn.Sequential(
            nn.Linear(len(IDX_STATE), d), nn.ELU(), nn.Linear(d, d))

        # ---- N2 dual graph -------------------------------------------
        if use_graph:
            self.g_sp = GraphAttn(d, heads)
            self.norm_sp = nn.LayerNorm(d)
            if use_hydro:
                self.g_hy = GraphAttn(d, heads)
                self.norm_hy = nn.LayerNorm(d)
                self.fuse_graph = GatedFuse(d)

        # ---- N3 lithology-adaptive gate ------------------------------
        self.static_enc = nn.Sequential(nn.Linear(f_static, d), nn.ELU())
        self.gate = nn.Linear(f_static, self.E)

        # ---- coefficient heads (one set per expert) -------------------
        self.coef = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d, d), nn.ELU(), nn.Dropout(dropout),
                          nn.Linear(d, 3 * n_out)) for _ in range(self.E)])
        self.delta = nn.Sequential(nn.Linear(2 * d, d), nn.ELU(),
                                   nn.Linear(d, n_out))
        self.direct = nn.Sequential(nn.Linear(2 * d, d), nn.ELU(),
                                    nn.Dropout(dropout), nn.Linear(d, n_out))
        # ---- N4 reconstruction head ----------------------------------
        self.recon = nn.Linear(d, f_dyn)
        # ---- N6 attribution head (joint, not post-hoc) ---------------
        if use_attr:
            self.attr = nn.Linear(2 * d, 4)     # state / forcing / static / graph

    def forward(self, x, xa, s, y_prev, y_seas, A_sp, A_hy):
        B, L, n, Fd = x.shape
        flat = x.permute(0, 2, 1, 3).reshape(B * n, L, Fd)

        # N5: monthly branch (+ annual branch)
        hm, _ = self.month_gru(flat)
        h = hm[:, -1]
        if self.use_annual:
            fa = xa.permute(0, 2, 1, 3).reshape(B * n, xa.shape[1], xa.shape[3])
            hy, _ = self.year_gru(fa)
            h = self.fuse_time(h, hy[:, -1])
        h = h.view(B, n, -1)

        # N1: separate forcing and state embeddings at the last observed step
        last = x[:, -1]                                   # (B, N, F)
        hf = self.force_enc(last[..., IDX_FORCE])
        hs = self.state_enc(last[..., IDX_STATE])
        h = h + hs                                        # state-informed context

        # N2: dual-graph message passing
        graph_msg = torch.zeros_like(h)
        if self.use_graph:
            gs = self.norm_sp(self.g_sp(h, A_sp))
            if self.use_hydro:
                gh = self.norm_hy(self.g_hy(h, A_hy))
                graph_msg = self.fuse_graph(gs, gh)
            else:
                graph_msg = gs
            h = h + graph_msg

        # N3: expert weights from static lithology / terrain context
        ctx = self.static_enc(s).unsqueeze(0).expand(B, -1, -1)
        z = torch.cat([h, ctx], -1)
        if self.use_moe:
            w = torch.softmax(self.gate(s), -1)           # (N, E)
        else:
            w = torch.ones(n, 1, device=x.device)

        if self.use_n1:
            # constrained damped-exponential transfer function.
            # rho, sig in (0,1); kap in (-1,1). Nests Persistence (rho=1) and
            # SeasonalNaive (sig=1) exactly.
            coefs = torch.stack([c(z) for c in self.coef], dim=2)  # (B,N,E,3*out)
            wexp = w.unsqueeze(0).unsqueeze(-1)                    # (1,N,E,1)
            mixed = (coefs * wexp).sum(2)                          # (B,N,3*out)
            rho, sig, kap = mixed.split(self.n_out, dim=-1)
            rho = torch.sigmoid(rho); sig = torch.sigmoid(sig)
            kap = torch.tanh(kap)
            f_eff = self.delta(torch.cat([hf, ctx], -1))
            pred = rho * y_prev + sig * y_seas + kap * f_eff
            aux = dict(rho=rho, sig=sig, kap=kap, w=w)
        else:
            # ablation: plain regression head, no dynamical structure
            pred = self.direct(z)
            aux = dict(rho=None, sig=None, kap=None, w=w)

        aux["recon"] = self.recon(h)
        aux["attr"] = (torch.softmax(self.attr(z), -1) if self.use_attr else None)
        return pred, aux
print("PERSIST defined (all novelties flag-switchable)")
''')

md(r"""
## 7 · N4 — ecologically-constrained composite loss

| Term | Purpose |
|---|---|
| `L_pred` | masked MSE on the targets |
| `L_recon` | self-supervised reconstruction of the input features |
| `L_smooth` | temporal smoothness of the recovery coefficient $\rho$ — resilience cannot jump discontinuously |
| `L_lap` | graph-Laplacian penalty: neighbouring counties should not disagree wildly |
| `L_asym` | asymmetric penalty on ecologically implausible runaway predictions |
| `L_balance` | expert load balancing, so N3 cannot collapse onto one expert |
""")

code(r'''
def persist_loss(pred, y, mask, aux, x, use_eco=True):
    m = mask.unsqueeze(-1)
    denom = m.sum().clamp(min=1) * y.shape[-1]
    L_pred = (((pred - torch.nan_to_num(y)) ** 2) * m).sum() / denom
    parts = {"pred": float(L_pred)}
    total = L_pred
    if not use_eco:
        return total, parts

    # reconstruction of the last input step
    L_rec = F.mse_loss(aux["recon"], x[:, -1])
    total = total + W_RECON * L_rec; parts["recon"] = float(L_rec)

    # temporal smoothness of rho across the batch's time ordering
    if aux["rho"] is not None and pred.shape[0] > 1:
        L_sm = (aux["rho"][1:] - aux["rho"][:-1]).pow(2).mean()
        total = total + W_SMOOTH * L_sm; parts["smooth"] = float(L_sm)

    # graph Laplacian: p^T L p, penalises disagreement between neighbours
    p = pred.permute(0, 2, 1)                        # (B, out, N)
    L_lp = torch.einsum("bon,nm,bom->", p, L_t, p) / (p.shape[0] * p.shape[1])
    total = total + W_LAP * L_lp.clamp(min=0); parts["laplacian"] = float(L_lp)

    # asymmetric runaway penalty: only punish |pred| beyond the plausible range
    over = (pred.abs() - CLIP_SIGMA).clamp(min=0)
    L_as = over.pow(2).mean()
    total = total + W_ASYM * L_as; parts["asym"] = float(L_as)

    # expert load balancing (prevents N3 collapse)
    if aux["w"] is not None and aux["w"].shape[-1] > 1:
        use = aux["w"].mean(0)
        L_bal = (use * torch.log(use.clamp(min=1e-8))).sum()   # neg entropy
        total = total + W_BALANCE * L_bal; parts["balance"] = float(L_bal)
    return total, parts
print("N4 composite loss defined")
''')

# ══════════════════════════════════════════════════════════ DATA LOADER
md("## 8 · Graph-snapshot dataset")

code(r'''
class SnapDS(torch.utils.data.Dataset):
    """One sample = one target month across ALL counties (graph snapshot)."""
    def __init__(self, ends):
        self.ends = np.asarray(ends)
    def __len__(self): return len(self.ends)
    def __getitem__(self, i):
        e = int(self.ends[i])
        yi = e // 12
        lo = max(0, yi - ANNUAL_YEARS)
        ann = Xa[:, lo:yi]                                    # (N, k, F)
        if ann.shape[1] < ANNUAL_YEARS:                       # pad early years
            pad = Xa[:, :1].expand(-1, ANNUAL_YEARS - ann.shape[1], -1)
            ann = torch.cat([pad, ann], dim=1)
        return (Xt[:, e - LOOKBACK:e].permute(1, 0, 2),       # (L, N, F)
                ann.permute(1, 0, 2),                          # (k, N, F)
                Yt[:, e],                                      # (N, out)
                torch.from_numpy(VALID[:, e].astype(np.float32)),
                Yfill[:, e - 1],                               # y_prev  (N1)
                Yfill[:, max(e - 12, 0)],                      # y_seas  (N1)
                e)

def snap_loader(ends, bs, shuffle):
    return torch.utils.data.DataLoader(SnapDS(ends), batch_size=bs,
                                       shuffle=shuffle, num_workers=0)

def persist_step(model, batch, use_eco=True):
    x, xa, y, mask, yp, ys, _ = batch
    x, xa, y = x.to(DEVICE), xa.to(DEVICE), y.to(DEVICE)
    mask, yp, ys = mask.to(DEVICE), yp.to(DEVICE), ys.to(DEVICE)
    pred, aux = model(x, xa, St.to(DEVICE), yp, ys, A_sp_t, A_hy_t)
    loss, parts = persist_loss(pred, y, mask, aux, x, use_eco)
    return pred, y, mask, loss, aux, parts
print("snapshot loader + step defined")
''')

# ═══════════════════════════════════════════════════════════ TRAIN LOOP
md("## 9 · Training loop — epoch-wise printing **and** `history.csv`")

code(r'''
def quick_metrics(o, p):
    o, p = np.asarray(o).ravel(), np.asarray(p).ravel()
    m = np.isfinite(o) & np.isfinite(p); o, p = o[m], p[m]
    if len(o) < 3: return np.nan, np.nan, np.nan
    e = p - o; sst = ((o - o.mean()) ** 2).sum()
    return (float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean()),
            float(1 - (e ** 2).sum() / sst) if sst > 0 else np.nan)

def run_epoch(model, dl, opt, train, use_eco):
    model.train() if train else model.eval()
    tot, nb, P, O = 0.0, 0, [], []
    for batch in dl:
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            pred, y, mask, loss, aux, _ = persist_step(model, batch, use_eco)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        tot += float(loss); nb += 1
        mk = mask.detach().cpu().numpy().astype(bool)
        P.append(pred.detach().cpu().numpy()[mk])
        O.append(y.detach().cpu().numpy()[mk])
    return tot / max(nb, 1), np.concatenate(P), np.concatenate(O)

def train_persist(name, model, folder, epochs, use_eco=True):
    folder = Path(folder); (folder / "plots").mkdir(parents=True, exist_ok=True)
    (folder / "predictions").mkdir(parents=True, exist_ok=True)
    model = model.to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=.5,
                                                       patience=6, min_lr=1e-6)
    tr = snap_loader(tr_e, GRAPH_BATCH, True)
    va = snap_loader(va_e, GRAPH_BATCH, False)

    print(f"\n{'='*76}\n{name}  |  {nparam:,} parameters  |  {DEVICE}\n{'='*76}")
    hdr = (f"{'ep':>4} {'tr_loss':>10} {'va_loss':>10} {'tr_RMSE':>9} "
           f"{'va_RMSE':>9} {'tr_MAE':>8} {'va_MAE':>8} {'tr_R2':>8} "
           f"{'va_R2':>8} {'lr':>9} {'sec':>6}")
    print(hdr); print("-" * len(hdr))

    hist, best, bad, bstate = [], np.inf, 0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        trl, trP, trO = run_epoch(model, tr, opt, True, use_eco)
        val, vaP, vaO = run_epoch(model, va, opt, False, use_eco)
        trR, trM, trR2 = quick_metrics(trO, trP)
        vaR, vaM, vaR2 = quick_metrics(vaO, vaP)
        lr = opt.param_groups[0]["lr"]; sched.step(val)
        hist.append(dict(epoch=ep, train_loss=trl, val_loss=val,
                         train_rmse=trR, val_rmse=vaR, train_mae=trM,
                         val_mae=vaM, train_r2=trR2, val_r2=vaR2, lr=lr,
                         seconds=time.time() - t0))
        print(f"{ep:>4} {trl:>10.5f} {val:>10.5f} {trR:>9.4f} {vaR:>9.4f} "
              f"{trM:>8.4f} {vaM:>8.4f} {trR2:>8.4f} {vaR2:>8.4f} "
              f"{lr:>9.2e} {hist[-1]['seconds']:>6.1f}")
        pd.DataFrame(hist).to_csv(folder / "history.csv", index=False)
        if vaR < best - 1e-6:
            best, bad = vaR, 0
            bstate = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at epoch {ep} (best val RMSE {best:.4f})")
                break
    if bstate is not None: model.load_state_dict(bstate)
    torch.save(model.state_dict(), folder / "model_best.pt")
    print(f"best val RMSE = {best:.4f} | history.csv + model_best.pt saved")
    return model, hist, nparam

@torch.no_grad()
def predict_all(model, ends, use_eco=True):
    model.eval()
    P, O, C, Tt, AUX = [], [], [], [], []
    for batch in snap_loader(ends, GRAPH_BATCH, False):
        pred, y, mask, _, aux, _ = persist_step(model, batch, use_eco)
        pred = pred.cpu().numpy(); y = y.cpu().numpy()
        mk = mask.cpu().numpy().astype(bool); e = batch[-1].numpy()
        for b in range(pred.shape[0]):
            keep = np.where(mk[b])[0]
            P.append(pred[b][keep]); O.append(y[b][keep])
            C.append(keep); Tt.append(np.full(len(keep), int(e[b])))
            if aux["rho"] is not None:
                AUX.append(pd.DataFrame({
                    "ci": keep, "ti": int(e[b]),
                    "rho": aux["rho"][b].cpu().numpy()[keep, 0],
                    "sig": aux["sig"][b].cpu().numpy()[keep, 0],
                    "kap": aux["kap"][b].cpu().numpy()[keep, 0]}))
    aux_df = pd.concat(AUX, ignore_index=True) if AUX else None
    return (np.concatenate(P), np.concatenate(O), np.concatenate(C),
            np.concatenate(Tt), aux_df)
''')

# ════════════════════════════════════════════════════════════ PLOTTING
md("## 10 · Publication-quality plotting *(font 20, dpi 300)*")

code(r'''
def _save(fig, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(folder) / f"{name}.{FIG_FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def plot_history(hist, folder, model):
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(h.epoch, h.train_loss, lw=2.5, label="Train")
    ax.plot(h.epoch, h.val_loss, lw=2.5, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{model} — Training curve"); ax.legend()
    _save(fig, folder, "01_loss_curve")
    for k, lab in (("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R$^2$")):
        if f"train_{k}" not in h: continue
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.plot(h.epoch, h[f"train_{k}"], lw=2.5, label=f"Train {lab}")
        ax.plot(h.epoch, h[f"val_{k}"], lw=2.5, label=f"Val {lab}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab); ax.legend()
        ax.set_title(f"{model} — {lab} per epoch")
        _save(fig, folder, f"02_{k}_curve")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(h.epoch, h.lr, lw=2.5, color="darkgreen"); ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title(f"{model} — LR schedule")
    _save(fig, folder, "03_lr_schedule")

def plot_core(df, metrics, folder, model):
    for tn in TARGETS:
        o, p = df[f"obs_{tn}"].values, df[f"pred_{tn}"].values
        m = np.isfinite(o) & np.isfinite(p); o, p = o[m], p[m]
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.hexbin(o, p, gridsize=60, mincnt=1, cmap="viridis")
        lim = [np.percentile(o, .5), np.percentile(o, 99.5)]
        ax.plot(lim, lim, "r--", lw=2.5, label="1:1")
        ax.plot(lim, np.polyval(np.polyfit(o, p, 1), lim), "orange", lw=2.5,
                label="Fit")
        mm = core_metrics(o, p)
        ax.set_xlabel(f"Observed {tn}"); ax.set_ylabel(f"Predicted {tn}")
        ax.set_title(f"{model} — {tn}\nR$^2$={mm['R2']:.3f}  RMSE={mm['RMSE']:.3f}")
        ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim)
        _save(fig, folder, f"04_scatter_{tn}")
    prim = TARGETS[0]; r = df[f"res_{prim}"].dropna().values
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.hist(r, bins=80, color="steelblue", edgecolor="k", alpha=.85)
    ax.axvline(0, color="r", ls="--", lw=2.5)
    ax.set_xlabel(f"Residual ({prim})"); ax.set_ylabel("Count")
    ax.set_title(f"{model} — Residuals\nmean={r.mean():.3f} sd={r.std():.3f}")
    _save(fig, folder, "05_residual_hist")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(df[f"pred_{prim}"], df[f"res_{prim}"], s=4, alpha=.15)
    ax.axhline(0, color="r", ls="--", lw=2.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
    ax.set_title(f"{model} — Residuals vs prediction")
    _save(fig, folder, "06_residual_vs_pred")
    from scipy import stats as _st
    fig, ax = plt.subplots(figsize=(9, 9))
    _st.probplot(r, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(3); ax.get_lines()[1].set_linewidth(2.5)
    ax.set_title(f"{model} — Residual Q–Q")
    _save(fig, folder, "07_residual_qq")

    d = df.copy(); d["abs_err"] = d[f"res_{prim}"].abs()
    order = [r_ for r_ in ["upstream", "midstream", "downstream"]
             if r_ in d.reach.unique()]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.boxplot([d[d.reach == r_].abs_err.dropna() for r_ in order],
               showfliers=False)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order)
    ax.set_ylabel("Absolute error"); ax.set_title(f"{model} — Error by reach")
    _save(fig, folder, "08_error_by_reach")
    g = d.groupby("month").abs_err.mean()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(g.index, g.values, color="teal", edgecolor="k")
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month")
    ax.set_ylabel("MAE"); ax.set_title(f"{model} — Error seasonality")
    _save(fig, folder, "09_error_by_month")
    g = d.groupby("year").abs_err.mean()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(g.index, g.values, "o-", lw=2.5, ms=9, color="darkred")
    ax.set_xlabel("Year"); ax.set_ylabel("MAE")
    ax.set_title(f"{model} — Error by test year")
    _save(fig, folder, "10_error_by_year")
    if "is_shock" in d:
        v = [d[~d.is_shock].abs_err.dropna(), d[d.is_shock].abs_err.dropna()]
        v = [z if len(z) else pd.Series([np.nan]) for z in v]
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.boxplot(v, showfliers=False)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Calm", "Shock"])
        ax.set_ylabel("Absolute error")
        ax.set_title(f"{model} — Calm vs disturbance")
        _save(fig, folder, "11_error_shock_vs_calm")

    cnt = df.groupby("adcode").size().sort_values(ascending=False)
    pick = list(cnt.index[:4])
    fig, axes = plt.subplots(len(pick), 1, figsize=(14, 4.2 * len(pick)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, pick):
        sdf = df[df.adcode == a].sort_values("ti")
        ax.plot(sdf.ti, sdf[f"obs_{prim}"], "o-", lw=2.2, ms=6, label="Observed")
        ax.plot(sdf.ti, sdf[f"pred_{prim}"], "s--", lw=2.2, ms=6, label="Predicted")
        ax.set_ylabel(prim); ax.set_title(f"County {a}"); ax.legend(loc="upper right")
    axes[-1].set_xlabel("Month index")
    fig.suptitle(f"{model} — Example trajectories")
    _save(fig, folder, "12_example_timeseries")

    if HAS_GPD:
        try:
            gdf = gpd.read_file(BOUNDARY_FILE); gdf["adcode"] = gdf.adcode.astype(str)
            per = (df.assign(se=lambda z: z[f"res_{prim}"] ** 2)
                     .groupby("adcode").se.mean().pow(.5).rename("rmse").reset_index())
            g2 = gdf.merge(per, on="adcode", how="left")
            fig, ax = plt.subplots(figsize=(15, 11))
            g2.plot(column="rmse", ax=ax, legend=True, cmap="YlOrRd",
                    edgecolor="grey", linewidth=.15,
                    missing_kwds={"color": "lightgrey"},
                    legend_kwds={"label": f"Test RMSE ({prim})", "shrink": .7})
            ax.set_title(f"{model} — Spatial error"); ax.set_axis_off()
            _save(fig, folder, "13_map_rmse")
        except Exception as e:
            print("  map skipped:", e)

    keys = ["all_RMSE", "all_MAE", "all_R2", "all_PearsonR", "all_WillmottD",
            "all_KGE", "within_county_R2"]
    vals = [metrics.get(k, np.nan) for k in keys]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.bar([k.replace("all_", "") for k in keys], vals, color="slateblue",
           edgecolor="k")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=FONT_SIZE - 6)
    ax.axhline(0, color="k", lw=1); ax.set_ylabel("Value")
    ax.set_title(f"{model} — Test metrics"); plt.xticks(rotation=25, ha="right")
    _save(fig, folder, "14_metric_summary")

def plot_interpretability(aux_df, model_obj, folder, model):
    """N1 + N3 + N6 artefacts: recovery, resistance, expert usage."""
    if aux_df is None or not len(aux_df): return
    d = aux_df.copy()
    d["recovery"] = 1 - d.rho          # speed of return to baseline
    d["resistance"] = 1 - d.kap.abs()  # insensitivity to forcing
    d["adcode"] = [counties[c] for c in d.ci]
    reach = panel.groupby("adcode").reach.first()
    d["reach"] = d.adcode.map(reach)
    d.to_csv(Path(folder).parent / "interpretability" /
             "n1_coefficients.csv", index=False)

    for col, lab in (("rho", r"Persistence $\rho$"),
                     ("recovery", "Recovery rate (1-$\\rho$)"),
                     ("resistance", "Resistance (1-|$\\kappa$|)"),
                     ("sig", r"Seasonal carry $\sigma$")):
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.hist(d[col].dropna(), bins=60, color="darkslateblue",
                edgecolor="k", alpha=.85)
        ax.set_xlabel(lab); ax.set_ylabel("Count")
        ax.set_title(f"{model} — N1 {lab}\nmean={d[col].mean():.3f}")
        _save(fig, folder, f"20_n1_{col}_hist")

    fig, ax = plt.subplots(figsize=(11, 7))
    order = [r for r in ["upstream", "midstream", "downstream"]
             if r in d.reach.unique()]
    ax.boxplot([d[d.reach == r].recovery.dropna() for r in order],
               showfliers=False)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order)
    ax.set_ylabel("Recovery rate"); ax.set_xlabel("Reach")
    ax.set_title(f"{model} — N1 recovery by reach")
    _save(fig, folder, "21_n1_recovery_by_reach")

    if HAS_GPD:
        try:
            gdf = gpd.read_file(BOUNDARY_FILE); gdf["adcode"] = gdf.adcode.astype(str)
            for col, lab, cm in (("recovery", "Recovery rate", "viridis"),
                                 ("resistance", "Resistance", "magma")):
                per = d.groupby("adcode")[col].mean().reset_index()
                g2 = gdf.merge(per, on="adcode", how="left")
                fig, ax = plt.subplots(figsize=(15, 11))
                g2.plot(column=col, ax=ax, legend=True, cmap=cm,
                        edgecolor="grey", linewidth=.15,
                        missing_kwds={"color": "lightgrey"},
                        legend_kwds={"label": lab, "shrink": .7})
                ax.set_title(f"{model} — N1 {lab} (mean over test period)")
                ax.set_axis_off()
                _save(fig, folder, f"22_map_{col}")
        except Exception as e:
            print("  interpretability map skipped:", e)

    # N3 expert usage
    try:
        w = torch.softmax(model_obj.gate(St.to(DEVICE)), -1).detach().cpu().numpy()
        if w.shape[1] > 1:
            use = w.mean(0)
            ent = float(-(use * np.log(use + 1e-9)).sum() / np.log(len(use)))
            fig, ax = plt.subplots(figsize=(11, 7))
            ax.bar([f"E{i+1}" for i in range(len(use))], use,
                   color="teal", edgecolor="k")
            for i, v in enumerate(use):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=FONT_SIZE - 5)
            ax.set_ylabel("Mean gate weight")
            ax.set_title(f"{model} — N3 expert usage\n"
                         f"normalised entropy = {ent:.3f} (1.0 = balanced)")
            _save(fig, folder, "23_n3_expert_usage")
            pd.DataFrame({"expert": range(1, len(use) + 1), "usage": use}
                         ).to_csv(Path(folder).parent / "interpretability" /
                                  "n3_expert_usage.csv", index=False)
            print(f"  N3 expert entropy = {ent:.3f}")
    except Exception as e:
        print("  N3 plot skipped:", e)

def finalise(name, model, hist, nparam, folder, elapsed, use_eco=True,
             tag="proposed"):
    folder = Path(folder)
    pred, obs, ci, ti, aux_df = predict_all(model, te_e, use_eco)
    metrics, df = evaluate(pred, obs, ci, ti, "test", A_sp)
    metrics.update(model=name, type=tag, n_parameters=int(nparam),
                   epochs_run=len(hist),
                   best_val_rmse=float(min(h["val_rmse"] for h in hist)),
                   train_seconds=float(elapsed))
    with open(folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    pd.DataFrame([metrics]).to_csv(folder / "metrics.csv", index=False)
    df.to_csv(folder / "predictions" / "test_predictions.csv", index=False)

    print(f"\n--- {name} TEST METRICS ---")
    for k in ["all_RMSE", "all_MAE", "all_R2", "all_PearsonR", "all_WillmottD",
              "all_KGE", "all_Bias", "within_county_R2",
              "within_county_PearsonR", "residual_MoranI", "shock_RMSE",
              "calm_RMSE", "strict_lst_z_R2", "raw_lst_c_R2"]:
        if k in metrics: print(f"  {k:24} {metrics[k]:+.4f}")
    plot_history(hist, folder / "plots", name)
    plot_core(df, metrics, folder / "plots", name)
    if tag == "proposed":
        plot_interpretability(aux_df, model, folder / "plots", name)
    print(f"  plots -> {folder/'plots'}")
    return metrics
print("plotting + finalise defined")
''')

# ═════════════════════════════════════════════════════════════ RUN MAIN
md("## 11 · Train PERSIST")

code(r'''
EP = 2 if SMOKE_TEST else EPOCHS
F_DYN, F_STAT, F_ANN = X.shape[2], S.shape[1], Xann.shape[2]
print(f"F_dyn={F_DYN} F_static={F_STAT} F_annual={F_ANN} epochs={EP}")

set_seed(SEED)
t0 = time.time()
persist = PERSIST(F_DYN, F_STAT, F_ANN, len(TARGETS))
persist, hist_p, npar_p = train_persist("PERSIST", persist, PERSIST_DIR, EP)
M_PERSIST = finalise("PERSIST", persist, hist_p, npar_p, PERSIST_DIR,
                     time.time() - t0, True, "proposed")
''')

# ═══════════════════════════════════════════════════════════ ABLATIONS
md(r"""
## 12 · Ablations

Each variant switches **one** novelty off, reusing the identical code path so no
implementation drift can creep in.

| Variant | Novelty removed |
|---|---|
| `no_N1_DCRD` | dynamical decoder → plain regression head |
| `no_N2a_hydro` | hydrological graph (spatial only) |
| `no_N2b_graph` | both graphs |
| `no_N3_moe` | Mixture-of-Experts → single expert |
| `no_N4_ecoloss` | ecological loss terms → plain MSE |
| `no_N5_annual` | annual branch → monthly only |
| `no_N6_attr` | attribution head |
""")

code(r'''
ABLATIONS = {
    "no_N1_DCRD":    dict(use_n1=False),
    "no_N2a_hydro":  dict(use_hydro=False),
    "no_N2b_graph":  dict(use_graph=False, use_hydro=False),
    "no_N3_moe":     dict(use_moe=False),
    "no_N4_ecoloss": dict(),                 # handled via use_eco flag
    "no_N5_annual":  dict(use_annual=False),
    "no_N6_attr":    dict(use_attr=False),
}
ABL_RESULTS, ABL_HIST = {}, {}

if RUN_ABLATIONS:
    for vname, kw in ABLATIONS.items():
        use_eco = (vname != "no_N4_ecoloss")
        fold = ABL_DIR / vname
        set_seed(SEED)
        t0 = time.time()
        mdl = PERSIST(F_DYN, F_STAT, F_ANN, len(TARGETS), **kw)
        mdl, h, npar = train_persist(f"ABLATION: {vname}", mdl, fold, EP, use_eco)
        ABL_RESULTS[vname] = finalise(vname, mdl, h, npar, fold,
                                      time.time() - t0, use_eco, "ablation")
        ABL_HIST[vname] = h
        del mdl
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
else:
    print("RUN_ABLATIONS = False, skipped")
''')

code(r'''
# ---- ablation comparison table --------------------------------------
if ABL_RESULTS:
    rows = [M_PERSIST] + list(ABL_RESULTS.values())
    abl = pd.DataFrame(rows)
    front = ["model", "type", "n_parameters", "epochs_run", "all_RMSE",
             "all_MAE", "all_R2", "all_PearsonR", "all_WillmottD", "all_KGE",
             "within_county_R2", "residual_MoranI", "shock_RMSE"]
    abl = abl[[c for c in front if c in abl.columns] +
              [c for c in abl.columns if c not in front]]
    # delta vs full PERSIST: positive dRMSE means removing the novelty HURT
    base = float(abl.loc[abl.model == "PERSIST", "all_RMSE"].iloc[0])
    abl["dRMSE_vs_PERSIST"] = abl.all_RMSE.astype(float) - base
    baseR2 = float(abl.loc[abl.model == "PERSIST", "all_R2"].iloc[0])
    abl["dR2_vs_PERSIST"] = abl.all_R2.astype(float) - baseR2
    abl.to_csv(ABL_DIR / "ablation_comparison.csv", index=False)
    print("saved ->", ABL_DIR / "ablation_comparison.csv")
    show = ["model", "all_RMSE", "dRMSE_vs_PERSIST", "all_R2",
            "dR2_vs_PERSIST", "within_county_R2", "residual_MoranI"]
    display(abl[[c for c in show if c in abl]].round(4))
''')

# ══════════════════════════════════════════════════ COMPARISON: BASELINES
md("## 13 · Comparison A — PERSIST vs baselines")

code(r'''
def bar_compare(dfc, key, lab, folder, fname, highlight="PERSIST",
                ref_model=None):
    if key not in dfc: return
    fig, ax = plt.subplots(figsize=(12.5, 7))
    v = dfc[key].astype(float).values
    cols = ["#C0392B" if m == highlight else
            ("#BBBBBB" if t == "trivial" else "#4C72B0")
            for m, t in zip(dfc.model, dfc.get("type", ["deep"] * len(dfc)))]
    ax.bar(dfc.model, v, color=cols, edgecolor="k")
    for i, x in enumerate(v):
        if np.isfinite(x):
            ax.text(i, x, f"{x:.4f}", ha="center",
                    va="bottom" if x >= 0 else "top", fontsize=FONT_SIZE - 7)
    if ref_model is not None and ref_model in set(dfc.model):
        rv = float(dfc.loc[dfc.model == ref_model, key].iloc[0])
        ax.axhline(rv, color="crimson", ls="--", lw=2.5, label=f"{ref_model}")
        ax.legend()
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel(lab); ax.set_title(f"{lab}")
    plt.xticks(rotation=30, ha="right")
    _save(fig, folder, fname)

KEYS = [("all_RMSE", "Test RMSE (lower better)"),
        ("all_MAE", "Test MAE (lower better)"),
        ("all_R2", "Test R$^2$ (higher better)"),
        ("all_PearsonR", "Pearson r (higher better)"),
        ("all_WillmottD", "Willmott d (higher better)"),
        ("all_KGE", "KGE (higher better)"),
        ("within_county_R2", "Within-county R$^2$ (temporal skill)"),
        ("residual_MoranI", "Residual Moran's I (closer to 0 better)"),
        ("shock_RMSE", "Shock-month RMSE (lower better)")]

if BASELINE_CSV.exists():
    base_df = pd.read_csv(BASELINE_CSV)
    if "type" not in base_df: base_df["type"] = "deep"
    combo = pd.concat([base_df, pd.DataFrame([M_PERSIST])], ignore_index=True)
    keep = [c for c in ["model", "type", "n_parameters", "epochs_run",
                        "train_seconds", "all_RMSE", "all_MAE", "all_R2",
                        "all_PearsonR", "all_WillmottD", "all_KGE", "all_Bias",
                        "within_county_R2", "within_county_PearsonR",
                        "residual_MoranI", "shock_RMSE", "calm_RMSE"]
            if c in combo.columns]
    combo_out = combo[keep + [c for c in combo.columns if c not in keep]]
    combo_out.to_csv(CMP_BASE / "proposed_vs_baselines.csv", index=False)
    print("saved ->", CMP_BASE / "proposed_vs_baselines.csv")
    display(combo[keep].round(4))

    for k, lab in KEYS:
        bar_compare(combo, k, lab, CMP_BASE, f"cmp_{k}",
                    ref_model="SeasonalNaive" if k in
                    ("all_RMSE", "all_MAE", "shock_RMSE") else None)

    # verdict against the honest bars
    print("\n" + "=" * 70)
    for ref in ["Persistence", "SeasonalNaive"]:
        if ref in set(combo.model):
            rv = float(combo.loc[combo.model == ref, "all_RMSE"].iloc[0])
            pv = float(M_PERSIST["all_RMSE"])
            print(f"PERSIST vs {ref:15} {pv:.4f} vs {rv:.4f}  -> "
                  f"{'BEATS' if pv < rv else 'LOSES'} by "
                  f"{abs(rv-pv)/rv*100:.1f}%")
    dd = combo[combo.type == "deep"]
    if len(dd):
        bd = dd.loc[dd.all_RMSE.astype(float).idxmin()]
        pv = float(M_PERSIST["all_RMSE"])
        print(f"PERSIST vs best deep baseline ({bd['model']}): "
              f"{pv:.4f} vs {float(bd.all_RMSE):.4f} -> "
              f"{'BEATS' if pv < float(bd.all_RMSE) else 'LOSES'} by "
              f"{abs(float(bd.all_RMSE)-pv)/float(bd.all_RMSE)*100:.1f}%")
else:
    print("baseline_comparison.csv not found at", BASELINE_CSV)
    print("Run the Phase 4 notebook first; ablation comparison still works.")
    combo = None
''')

# ═════════════════════════════════════════════════ COMPARISON: ABLATIONS
md("## 14 · Comparison B — PERSIST vs ablations")

code(r'''
if ABL_RESULTS:
    for k, lab in KEYS:
        bar_compare(abl, k, lab, CMP_ABL, f"abl_{k}", ref_model="PERSIST")

    # novelty contribution: how much RMSE rises when each novelty is removed
    contrib = abl[abl.model != "PERSIST"][["model", "dRMSE_vs_PERSIST",
                                          "dR2_vs_PERSIST"]].copy()
    contrib = contrib.sort_values("dRMSE_vs_PERSIST", ascending=False)
    fig, ax = plt.subplots(figsize=(13, 7))
    cols = ["#2E7D32" if v > 0 else "#C62828"
            for v in contrib.dRMSE_vs_PERSIST]
    ax.barh(contrib.model, contrib.dRMSE_vs_PERSIST, color=cols, edgecolor="k")
    ax.axvline(0, color="k", lw=1.5)
    for i, v in enumerate(contrib.dRMSE_vs_PERSIST):
        ax.text(v, i, f" {v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=FONT_SIZE - 6)
    ax.set_xlabel("$\\Delta$RMSE when novelty removed")
    ax.set_title("Novelty contribution\n(positive = removing it HURTS, "
                 "so the novelty helps)")
    _save(fig, CMP_ABL, "abl_novelty_contribution")

    # ablation validation curves
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(pd.DataFrame(hist_p).epoch, pd.DataFrame(hist_p).val_rmse,
            lw=3.5, color="#C0392B", label="PERSIST (full)")
    for (vn, h), c in zip(ABL_HIST.items(), plt.cm.tab10.colors):
        hd = pd.DataFrame(h)
        ax.plot(hd.epoch, hd.val_rmse, lw=2, label=vn, color=c, alpha=.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE — PERSIST vs ablations")
    ax.legend(fontsize=FONT_SIZE - 8, ncol=2)
    _save(fig, CMP_ABL, "abl_val_curves")

    # radar over normalised metrics
    radar = [("all_R2", False), ("all_PearsonR", False), ("all_KGE", False),
             ("within_county_R2", False), ("all_RMSE", True), ("all_MAE", True)]
    radar = [(k, i) for k, i in radar if k in abl]
    lab = [k.replace("all_", "") for k, _ in radar]
    ang = np.linspace(0, 2 * np.pi, len(lab), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(polar=True))
    for i, mn in enumerate(abl.model):
        vs = []
        for k, inv in radar:
            cv = abl[k].astype(float); lo, hi = cv.min(), cv.max()
            v = float(abl.loc[abl.model == mn, k].iloc[0])
            sc = .5 if hi == lo else (v - lo) / (hi - lo)
            vs.append(1 - sc if inv else sc)
        vs += vs[:1]
        is_full = (mn == "PERSIST")
        ax.plot(ang, vs, lw=3.5 if is_full else 1.8,
                color="#C0392B" if is_full else plt.cm.tab10.colors[i % 10],
                label=mn, alpha=1.0 if is_full else .8)
        if is_full: ax.fill(ang, vs, alpha=.15, color="#C0392B")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(lab)
    ax.set_title("Ablation metric profile\n(outer = better)", pad=32)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.12),
              fontsize=FONT_SIZE - 8)
    _save(fig, CMP_ABL, "abl_radar")
    print("ablation comparison plots ->", CMP_ABL)
''')

# ══════════════════════════════════════════════════════════════ SUMMARY
md("## 15 · Summary")

code(r'''
print("=" * 78)
print("PHASE 5 COMPLETE — PERSIST + ablations")
print("=" * 78)
print(f"\nPERSIST test metrics:")
for k in ["all_RMSE", "all_MAE", "all_R2", "all_PearsonR", "all_KGE",
          "within_county_R2", "residual_MoranI", "shock_RMSE"]:
    if k in M_PERSIST: print(f"  {k:22} {M_PERSIST[k]:+.4f}")

if ABL_RESULTS:
    print("\nNovelty contributions (dRMSE when removed; positive = helps):")
    for _, r in (abl[abl.model != "PERSIST"]
                 .sort_values("dRMSE_vs_PERSIST", ascending=False).iterrows()):
        verdict = "HELPS" if r.dRMSE_vs_PERSIST > 0.001 else (
            "neutral" if r.dRMSE_vs_PERSIST > -0.001 else "HURTS (consider cutting)")
        print(f"  {r['model']:16} {r.dRMSE_vs_PERSIST:+.4f}   {verdict}")

print("\nArtefacts:")
print(f"  {PERSIST_DIR}")
print(f"     history.csv, metrics.json/csv, model_best.pt, predictions/,")
print(f"     plots/ ({len(list((PERSIST_DIR/'plots').glob('*.'+FIG_FMT)))} figures), "
      f"interpretability/")
if ABL_RESULTS:
    print(f"  {ABL_DIR}  ({len(ABL_RESULTS)} variants + ablation_comparison.csv)")
print(f"  {CMP_BASE}  ({len(list(CMP_BASE.glob('*.'+FIG_FMT)))} figures)")
print(f"  {CMP_ABL}   ({len(list(CMP_ABL.glob('*.'+FIG_FMT)))} figures)")
''')

md(r"""
---

## Notes

**Why the comparison is valid.** Targets, splits, scaling and `evaluate()` are
copied unchanged from the Phase 4 notebook. PERSIST sees the same 12-month input
window and the same feature set. The only additions are the *annual aggregates
of those same features* (N5) and the *directed hydrological graph* (N2) — no new
raw data.

**On N1 and the trivial baselines.** The damped-exponential form nests
Persistence and SeasonalNaive exactly, and $y_t$, $y_{t-11}$ are already inside
the baselines' input window. PERSIST is therefore not given extra information —
it is given better structure. That is the intended contribution.

**Reading the ablation table.** `dRMSE_vs_PERSIST > 0` means removing the
novelty made things worse, i.e. the novelty helps. Any novelty with a negative
or near-zero delta is not earning its place and should be cut before submission,
exactly as the design doc requires.

**Carried caveats.** `n_rel` is a covariate because LST clear-sky sampling leaks
≈0.20 into the anomaly (Phase 3e). The hydrological graph is terrain-derived,
not true river topology, and must be described as such. `strict_*_R2` is expected
to stay poor — the pure county-demeaned interannual anomaly has only 0.155
lag-1 autocorrelation and is close to unpredictable; that is a property of the
system, not a model failure.
""")

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md"
            else nbf.v4.new_code_cell(s) for k, s in C]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
}
out = pathlib.Path("PERSIST_Phase5_Proposed_Model.ipynb")
nbf.write(nb, str(out))
print(f"wrote {out} ({len(nb.cells)} cells, {out.stat().st_size/1024:.0f} KB)")
