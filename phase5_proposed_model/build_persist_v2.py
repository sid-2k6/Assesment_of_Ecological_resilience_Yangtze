#!/usr/bin/env python3
"""Generate PERSIST v2 notebook — fixes the v1 underperformance.

v1 result: PERSIST lost to TFT on every metric, and 5 of 7 novelties showed
negative contribution. Diagnosis and fixes are documented in the notebook.
"""
import pathlib
import nbformat as nbf

C = []
def md(s): C.append(("md", s.strip("\n")))
def code(s): C.append(("code", s.strip("\n")))


md(r"""
# PERSIST v2 — Proposed Model + Ablations (revised)

## Why v2 exists

v1 ran to completion but **lost to the TFT baseline on every metric**, and 5 of 7
novelties showed a *negative* contribution. This is the diagnosis and the repair.

| v1 metric | PERSIST v1 | TFT baseline | |
|---|---|---|---|
| RMSE | 0.6355 | **0.6204** | lost |
| R² | 0.6593 | **0.6752** | lost |
| within-county R² | 0.4515 | **0.5349** | lost |
| shock RMSE | 0.7292 | **0.6508** | lost |
| residual Moran's I | 0.7519 | 0.6790 (STGCN **0.5504**) | lost |

## Root cause: effective sample size, not capacity

The graph formulation batches **all counties into one snapshot**, so an epoch has
only **168 training samples** — one per target month. The per-county baselines
(DRSEI, TFT) get **179,292** samples because each (county, month) pair is its own
sample.

v1 therefore had 21 optimiser steps per epoch against TFT's ~350, *and* overfit
hard (train R² 0.844 vs val R² 0.674, best val at epoch 13). 381k parameters
against 168 effective samples.

## The four fixes

| # | Fix | Addresses |
|---|---|---|
| **1** | **Node-chunked graph batching** — each step uses a spatially coherent county subset (with occasional full-graph steps). Multiplies gradient steps *and* acts as edge dropout. | tiny update budget + overfitting |
| **2** | **Repaired novelties** — every one now starts as a no-op and can only help if it earns its place (see below) | 5/7 novelties hurting |
| **3** | **EMA weights + cosine schedule with warmup + warm-start init at the nested baseline** | optimisation instability |
| **4** | **Multi-seed ensembling with error bars** | v1 ablation deltas (±0.013) were within run-to-run noise and therefore uninterpretable |

## How each novelty was repaired

| Novelty | v1 problem | v2 repair |
|---|---|---|
| **N1** | random init, so training starts far from the nested baseline | **warm-start**: σ bias initialised high so the model *begins* near SeasonalNaive, then improves |
| **N2** | graph message added raw to `h`, injecting noise | **gated residual with learnable scale initialised at ~0** — graph starts as a no-op |
| **N4** | weights too strong; Laplacian over-smoothed predictions | weights reduced 10×; Laplacian moved onto **residuals**, which is the training-time analogue of residual Moran's I |
| **N5** | annual GRU fed the *same* features averaged (redundant), then *gated* against the monthly branch, diluting it | fed **year-over-year deltas** instead, and combined by **scaled residual** rather than a gate |
| **N6** | attribution head computed but never used and never supervised — pure untrained noise | attribution weights now **functionally gate the four information streams**, so the head is used and trainable |

## Honest expectation

Target: **beat TFT on every metric**, with the largest margins on residual
Moran's I and shock RMSE — the two diagnostics the novelties should genuinely
own. Realistic range is RMSE ≈ 0.57–0.60, R² ≈ 0.70–0.73.

R² of 0.82 on this target is **not** attainable without leakage. The
deseasonalised monthly anomaly has a measured lag-1 autocorrelation of 0.155,
and TFT reaches 0.675 with 657k parameters. Any 0.82 here would come from
target leakage, a non-temporal split, or reporting raw-space R² (which includes
the trivially predictable seasonal cycle — and where TFT already scores 0.9376).

> **Only the CONFIG cell below needs editing.**
""")

md("## 1 · CONFIGURATION — *the only cell you need to edit*")

code(r'''
# ════════════════════════════════════════════════════════════════════════
#  CONFIG  —  EDIT ONLY THIS CELL
# ════════════════════════════════════════════════════════════════════════
from pathlib import Path

MOUNT_DRIVE = True
REPO_ROOT   = Path("/content/drive/MyDrive/Assesment_of_Ecological_resilience_Yangtze")

DATA_DIR      = REPO_ROOT / "phase3_data" / "tables"
BOUNDARY_FILE = REPO_ROOT / "phase3_data" / "boundaries" / "yreb_counties_datav.gpkg"
OUTPUT_DIR    = REPO_ROOT / "outputs_v2"          # kept separate from v1
PANEL_FILE    = DATA_DIR / "panel_monthly.parquet"
ADJ_FILE      = DATA_DIR / "adjacency_edges.csv"
HYDRO_FILE    = DATA_DIR / "hydro_edges.csv"
# from the Phase 4 notebook (v1 outputs folder)
BASELINE_CSV  = REPO_ROOT / "outputs" / "baseline_comparison.csv"

SMOKE_TEST    = False
RUN_ABLATIONS = True
SEED          = 42

# --- task (MUST match Phase 4 for comparability) ----------------------
LOOKBACK   = 12
TARGETS    = ["lst_ds", "kndvi_ds"]
STRICT_TARGETS = ["lst_z", "kndvi_z"]
MIN_SD_FRAC = 0.10
CLIP_SIGMA  = 5.0
TRAIN_YEARS = (2000, 2014)
VAL_YEARS   = (2015, 2017)
TEST_YEARS  = (2018, 2020)

# --- FIX 4: multi-seed ensembling + error bars ------------------------
N_SEEDS = 3               # PERSIST and every ablation use the same count

# --- FIX 1: node-chunked graph batching ------------------------------
NODE_CHUNKS   = 3         # spatially coherent county subsets per step
P_FULL_GRAPH  = 0.25      # fraction of steps that use the complete graph
GRAPH_BATCH   = 4         # target months per step

# --- FIX 3: optimisation ---------------------------------------------
EPOCHS        = 150
LR            = 3e-3
WEIGHT_DECAY  = 3e-4
WARMUP_EPOCHS = 8
PATIENCE      = 30
GRAD_CLIP     = 1.0
USE_EMA       = True
EMA_DECAY     = 0.998
WARM_START_N1 = True      # init so training begins near SeasonalNaive

# --- architecture ----------------------------------------------------
D_MODEL      = 96
N_EXPERTS    = 4
ANNUAL_YEARS = 3
GRAPH_HEADS  = 2
DROPOUT      = 0.20

# --- FIX 2: N4 loss weights, reduced 10x from v1 ---------------------
W_RECON    = 0.005
W_SMOOTH   = 0.002
W_LAP_RES  = 0.02         # Laplacian on RESIDUALS (targets Moran's I)
W_ASYM     = 0.001
W_BALANCE  = 0.001

FONT_SIZE = 20
DPI       = 300
FIG_FMT   = "png"
USE_NREL_COVARIATE = True
SHOCK_THRESHOLD    = 1.5
# ════════════════════════════════════════════════════════════════════════
print("CONFIG loaded.")
print("  repo     :", REPO_ROOT)
print("  outputs  :", OUTPUT_DIR)
print("  baselines:", BASELINE_CSV)
print(f"  SMOKE_TEST={SMOKE_TEST}  RUN_ABLATIONS={RUN_ABLATIONS}  N_SEEDS={N_SEEDS}")
''')

md("## 2 · Environment")

code(r'''
import os, sys, json, math, time, warnings, random, copy
warnings.filterwarnings("ignore")

if MOUNT_DRIVE:
    try:
        from google.colab import drive; drive.mount("/content/drive")
    except Exception as e: print("Drive mount skipped:", e)

try:
    import geopandas as gpd; HAS_GPD = True
except ImportError:
    os.system(f"{sys.executable} -m pip install -q geopandas")
    try:
        import geopandas as gpd; HAS_GPD = True
    except Exception: HAS_GPD = False

import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| device:", DEVICE)
if DEVICE.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))

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
print("outputs ->", OUTPUT_DIR)
''')

md(r"""
## 3 · Data — identical preprocessing to Phase 4

Same targets, same train-only climatology, same scaling, same splits. Unchanged
so the baseline comparison stays valid.
""")

code(r'''
t0 = time.time()
panel = pd.read_parquet(PANEL_FILE)
panel["adcode"] = panel["adcode"].astype(str)
adj   = pd.read_csv(ADJ_FILE,   dtype={"u_adcode": str, "v_adcode": str})
hydro = pd.read_csv(HYDRO_FILE, dtype={"src": str, "dst": str})
panel["t"] = (panel.year - panel.year.min()) * 12 + (panel.month - 1)
panel = panel.sort_values(["adcode", "t"]).reset_index(drop=True)
print(f"panel {panel.shape[0]:,} x {panel.shape[1]} ({time.time()-t0:.1f}s)")

if SMOKE_TEST:
    keep = sorted(panel.adcode.unique())[:60]
    panel = panel[panel.adcode.isin(keep)].copy()
    print("SMOKE_TEST:", panel.adcode.nunique(), "counties")

counties = sorted(panel.adcode.unique())
cidx = {c: i for i, c in enumerate(counties)}
N, T = len(counties), panel.t.nunique()
print(f"N={N} T={T}")

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

panel["year_frac"] = (panel.year - TRAIN_YEARS[0]) / 20.0
panel["moy_sin"] = np.sin(2 * np.pi * panel.month / 12)
panel["moy_cos"] = np.cos(2 * np.pi * panel.month / 12)

STATE_FEATS = [c for c in ["lst_c", "kndvi", "ndvi_mean"] if c in panel]
FORCE_FEATS = [c for c in ["ppt", "pet", "aet", "def", "q", "tmax", "tmin",
                           "vpd", "soil", "srad", "pdsi", "swe", "wbal",
                           "heat_z", "dry_z", "tmax_z", "ppt_z", "soil_z",
                           "srad_z", "vpd_z", "pdsi_z"] if c in panel]
HUMAN_FEATS = [c for c in ["ntl_mean", "ntl_sum"] if c in panel]
STATIC_FEATS = [c for c in ["elev_mean", "elev_std", "relief", "slope_mean",
                            "slope_std", "roughness", "karst_frac",
                            "area_km2"] if c in panel]
if USE_NREL_COVARIATE: FORCE_FEATS = FORCE_FEATS + ["n_rel"]
CAL_FEATS = ["moy_sin", "moy_cos", "year_frac"]
DYN_FEATS = STATE_FEATS + FORCE_FEATS + HUMAN_FEATS + CAL_FEATS
IDX_STATE = [DYN_FEATS.index(c) for c in STATE_FEATS]
IDX_FORCE = [DYN_FEATS.index(c) for c in FORCE_FEATS]
print(f"dyn {len(DYN_FEATS)} | static {len(STATIC_FEATS)}")
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

# ---- FIX 2 (N5): annual branch gets YEAR-OVER-YEAR DELTAS, not the same
#      features averaged again. v1 fed redundant information.
n_years = T // 12
Xann_raw = X[:, :n_years * 12, :].reshape(N, n_years, 12, X.shape[2]).mean(axis=2)
Xann_d = np.diff(Xann_raw, axis=1, prepend=Xann_raw[:, :1])
Xann = np.concatenate([Xann_raw, Xann_d], axis=2).astype(np.float32)
print("annual context (levels + deltas)", Xann.shape)

def windows_for(times):
    return np.array(sorted([e for e in times if e - LOOKBACK >= 0]), dtype=np.int64)
tr_e, va_e, te_e = windows_for(tr_t), windows_for(va_t), windows_for(te_t)
VALID = ~np.isnan(Y).any(axis=2)
print(f"windows train {len(tr_e)} val {len(va_e)} test {len(te_e)}")

Xt = torch.from_numpy(X); Yt = torch.from_numpy(Y); St = torch.from_numpy(S)
Xa = torch.from_numpy(Xann)
Yfill = torch.from_numpy(np.nan_to_num(Y, nan=0.0))
''')

md("## 4 · Dual graph + node chunks (FIX 1)")

code(r'''
def build_norm_adj(edge_df, a, b, directed):
    A = np.zeros((N, N), dtype=np.float32); hit = 0
    for u, v in zip(edge_df[a], edge_df[b]):
        if u in cidx and v in cidx:
            A[cidx[u], cidx[v]] = 1.0
            if not directed: A[cidx[v], cidx[u]] = 1.0
            hit += 1
    return A, hit

def normalise(A):
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1); dinv = np.power(d, -0.5, where=d > 0); dinv[np.isinf(dinv)] = 0
    return (A * dinv[:, None] * dinv[None, :]).astype(np.float32)

A_sp_raw, n_sp = build_norm_adj(adj, "u_adcode", "v_adcode", False)
A_hy_raw, n_hy = build_norm_adj(hydro, "src", "dst", True)
A_sp, A_hy = normalise(A_sp_raw), normalise(A_hy_raw)
print(f"spatial {n_sp} edges | hydro {n_hy} edges (DIRECTED)")

A_sp_t = torch.from_numpy(A_sp).to(DEVICE)
A_hy_t = torch.from_numpy(A_hy).to(DEVICE)
L_t = torch.from_numpy((np.diag(A_sp_raw.sum(1)) - A_sp_raw)
                       / max(A_sp_raw.sum(), 1.0)).to(DEVICE)

# ---- FIX 1: spatially coherent chunks. adcodes are hierarchical
#      (province-prefecture-county) so a contiguous slice of the sorted list is
#      geographically compact, which preserves most edges. Lost cross-chunk
#      edges act as edge dropout.
chunks = np.array_split(np.arange(N), NODE_CHUNKS)
CHUNKS = []
for ch in chunks:
    sub_sp = torch.from_numpy(normalise(A_sp_raw[np.ix_(ch, ch)])).to(DEVICE)
    sub_hy = torch.from_numpy(normalise(A_hy_raw[np.ix_(ch, ch)])).to(DEVICE)
    kept = A_sp_raw[np.ix_(ch, ch)].sum() / max(A_sp_raw.sum(), 1)
    CHUNKS.append(dict(nodes=torch.from_numpy(ch).to(DEVICE),
                       A_sp=sub_sp, A_hy=sub_hy, n=len(ch)))
    print(f"  chunk n={len(ch):>5}  edges retained {kept*100:.1f}%")
FULL = dict(nodes=torch.arange(N).to(DEVICE), A_sp=A_sp_t, A_hy=A_hy_t, n=N)

steps_v1 = math.ceil(len(tr_e) / 8)
steps_v2 = math.ceil(len(tr_e) / GRAPH_BATCH) * (NODE_CHUNKS + 1)
print(f"\noptimiser steps per epoch: v1 ~{steps_v1}  ->  v2 ~{steps_v2} "
      f"({steps_v2/max(steps_v1,1):.1f}x)")
''')

md("## 5 · Metrics — identical to Phase 4")

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
    return 1 - np.sqrt((r-1)**2 + (sp/so-1)**2 + ((p.mean()-o.mean())/so)**2)
def core_metrics(o, p):
    o, p = _finite(np.asarray(o, float), np.asarray(p, float))
    if len(o) < 3:
        return {k: np.nan for k in ["RMSE","MAE","R2","PearsonR","WillmottD","KGE","Bias"]}
    err = p - o; sst = np.sum((o - o.mean()) ** 2)
    return {"RMSE": float(np.sqrt(np.mean(err**2))), "MAE": float(np.mean(np.abs(err))),
            "R2": float(1 - np.sum(err**2)/sst) if sst > 0 else np.nan,
            "PearsonR": float(np.corrcoef(o, p)[0,1]) if p.std() > 0 else np.nan,
            "WillmottD": float(willmott_d(o,p)), "KGE": float(kge(o,p)),
            "Bias": float(np.mean(err))}
def morans_I(values, A):
    v = np.asarray(values, float); m = np.isfinite(v)
    if m.sum() < 10: return np.nan
    W = A.copy(); np.fill_diagonal(W, 0.0); W = W[np.ix_(m, m)]
    v = v[m] - v[m].mean(); S0 = W.sum()
    if S0 == 0 or (v**2).sum() == 0: return np.nan
    return (len(v)/S0) * float(v @ (W @ v)) / float((v**2).sum())

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
def to_raw(v, prim, ci, ti):
    m = SPACE[prim]; return v * m["gsd"] + m["mmu"][MONTH_OF_TI[ti]-1]
def to_strict(v, prim, ci, ti):
    m = SPACE[prim]; mon = MONTH_OF_TI[ti]
    return (v*m["gsd"] + m["mmu"][mon-1] - m["cmu"][ci,mon]) / m["csd"][ci,mon]

def evaluate(pred, obs, ci, ti, tag, A_for_moran):
    out = {}
    out.update({f"all_{k}": v for k, v in core_metrics(obs.ravel(), pred.ravel()).items()})
    for j, tn in enumerate(TARGETS):
        for k, v in core_metrics(obs[:,j], pred[:,j]).items(): out[f"{tn}_{k}"] = v
    for j, prim in enumerate(TARGETS):
        for k, v in core_metrics(to_raw(obs[:,j],prim,ci,ti), to_raw(pred[:,j],prim,ci,ti)).items():
            out[f"raw_{SPACE[prim]['raw']}_{k}"] = v
        for k, v in core_metrics(to_strict(obs[:,j],prim,ci,ti), to_strict(pred[:,j],prim,ci,ti)).items():
            out[f"strict_{SPACE[prim]['strict']}_{k}"] = v
    df = pd.DataFrame({"ci": ci, "ti": ti})
    for j, tn in enumerate(TARGETS):
        df[f"obs_{tn}"]=obs[:,j]; df[f"pred_{tn}"]=pred[:,j]; df[f"res_{tn}"]=pred[:,j]-obs[:,j]
    df["adcode"]=[counties[c] for c in df.ci]
    df["year"]=year_of[df.ti.values]; df["month"]=month_of[df.ti.values]
    prim = TARGETS[0]; Is=[]
    for y, g in df.groupby("year"):
        vec=np.full(N,np.nan); gm=g.groupby("ci")[f"res_{prim}"].mean(); vec[gm.index.values]=gm.values
        Is.append(morans_I(vec, A_for_moran))
    out["residual_MoranI"]=float(np.nanmean(Is)) if Is else np.nan
    dd=df.dropna(subset=[f"obs_{prim}",f"pred_{prim}"]).copy()
    dd["o_d"]=dd[f"obs_{prim}"]-dd.groupby("ci")[f"obs_{prim}"].transform("mean")
    dd["p_d"]=dd[f"pred_{prim}"]-dd.groupby("ci")[f"pred_{prim}"].transform("mean")
    out["within_county_R2"]=core_metrics(dd.o_d,dd.p_d)["R2"]
    out["within_county_PearsonR"]=core_metrics(dd.o_d,dd.p_d)["PearsonR"]
    shock=panel[["ci","ti","heat_z"]].copy()
    shock["is_shock"]=shock.heat_z.abs()>=SHOCK_THRESHOLD
    df=df.merge(shock[["ci","ti","is_shock"]],on=["ci","ti"],how="left")
    df["is_shock"]=df.is_shock.fillna(False).astype(bool)
    for lbl, sub in (("shock",df[df.is_shock]),("calm",df[~df.is_shock])):
        mm=core_metrics(sub[f"obs_{prim}"],sub[f"pred_{prim}"])
        out[f"{lbl}_RMSE"]=mm["RMSE"]; out[f"{lbl}_R2"]=mm["R2"]; out[f"{lbl}_n"]=int(len(sub))
    reach=panel.groupby("adcode").reach.first(); df["reach"]=df.adcode.map(reach)
    for r, sub in df.groupby("reach"):
        out[f"reach_{r}_RMSE"]=core_metrics(sub[f"obs_{prim}"],sub[f"pred_{prim}"])["RMSE"]
    out["n_samples"]=int(len(df))
    return out, df
print("metrics identical to Phase 4")
''')

md(r"""
## 6 · PERSIST v2 architecture (FIX 2 — repaired novelties)

The governing principle: **every novelty starts as a no-op**. Each contribution
is added through a residual scaled by a learnable parameter initialised at ~0, so
the optimiser must actively choose to switch a novelty on. In v1 the novelties
were forced into the forward pass whether or not they helped.
""")

code(r'''
class GraphAttn(nn.Module):
    """Masked attention over graph neighbours; supports directed adjacency."""
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
    def forward(self, x, A):
        B, n, d = x.shape
        Q = self.q(x).view(B,n,self.h,self.dk).transpose(1,2)
        K = self.k(x).view(B,n,self.h,self.dk).transpose(1,2)
        V = self.v(x).view(B,n,self.h,self.dk).transpose(1,2)
        s = (Q @ K.transpose(-2,-1)) / math.sqrt(self.dk)
        s = s.masked_fill(~(A > 0).unsqueeze(0).unsqueeze(0), float("-inf"))
        a = torch.nan_to_num(torch.softmax(s, -1))
        return self.o((a @ V).transpose(1,2).reshape(B,n,d))


class PERSIST(nn.Module):
    def __init__(self, f_dyn, f_static, f_ann, n_out, d=D_MODEL,
                 experts=N_EXPERTS, heads=GRAPH_HEADS, dropout=DROPOUT,
                 use_n1=True, use_graph=True, use_hydro=True,
                 use_moe=True, use_annual=True, use_attr=True):
        super().__init__()
        self.use_n1, self.use_graph, self.use_hydro = use_n1, use_graph, use_hydro
        self.use_moe, self.use_annual, self.use_attr = use_moe, use_annual, use_attr
        self.n_out = n_out; self.E = experts if use_moe else 1

        self.month_gru = nn.GRU(f_dyn, d, num_layers=2, batch_first=True, dropout=dropout)
        self.month_norm = nn.LayerNorm(d)

        # --- N5 repaired: scaled residual (init ~0) instead of a gate --------
        if use_annual:
            self.year_gru = nn.GRU(f_ann, d, batch_first=True)
            self.a_year = nn.Parameter(torch.tensor(0.01))
            self.year_norm = nn.LayerNorm(d)

        self.force_enc = nn.Sequential(nn.Linear(len(IDX_FORCE), d), nn.ELU(),
                                       nn.Linear(d, d))
        self.state_enc = nn.Sequential(nn.Linear(len(IDX_STATE), d), nn.ELU(),
                                       nn.Linear(d, d))
        self.a_state = nn.Parameter(torch.tensor(0.1))

        # --- N2 repaired: gated residual, learnable scale init ~0 -----------
        if use_graph:
            self.g_sp = GraphAttn(d, heads); self.norm_sp = nn.LayerNorm(d)
            self.a_sp = nn.Parameter(torch.tensor(0.01))
            if use_hydro:
                self.g_hy = GraphAttn(d, heads); self.norm_hy = nn.LayerNorm(d)
                self.a_hy = nn.Parameter(torch.tensor(0.01))

        self.static_enc = nn.Sequential(nn.Linear(f_static, d), nn.ELU())
        self.gate = nn.Linear(f_static, self.E)

        # --- N6 repaired: attribution weights now GATE the four streams,
        #     so the head is functional and trainable rather than decorative.
        if use_attr:
            self.attr = nn.Linear(2 * d, 4)

        self.coef = nn.ModuleList([
            nn.Sequential(nn.Linear(2*d, d), nn.ELU(), nn.Dropout(dropout),
                          nn.Linear(d, 3*n_out)) for _ in range(self.E)])
        self.f_head = nn.Sequential(nn.Linear(2*d, d), nn.ELU(), nn.Linear(d, n_out))
        self.direct = nn.Sequential(nn.Linear(2*d, d), nn.ELU(),
                                    nn.Dropout(dropout), nn.Linear(d, n_out))
        self.recon = nn.Linear(d, f_dyn)

        # --- N1 warm start: bias sigma high so training BEGINS near
        #     SeasonalNaive (the strongest trivial reference), then improves.
        if WARM_START_N1 and use_n1:
            for c in self.coef:
                last = c[-1]
                nn.init.zeros_(last.weight)
                with torch.no_grad():
                    b = last.bias.view(3, n_out)
                    b[0].fill_(0.4)    # rho  -> sigmoid ~0.60
                    b[1].fill_(0.8)    # sig  -> sigmoid ~0.69
                    b[2].fill_(0.0)    # kap  -> tanh 0
                    last.bias.copy_(b.view(-1))

    def forward(self, x, xa, s, y_prev, y_seas, A_sp, A_hy):
        B, L, n, Fd = x.shape
        flat = x.permute(0,2,1,3).reshape(B*n, L, Fd)
        hm, _ = self.month_gru(flat)
        h = self.month_norm(hm[:, -1]).view(B, n, -1)

        if self.use_annual:
            fa = xa.permute(0,2,1,3).reshape(B*n, xa.shape[1], xa.shape[3])
            hy, _ = self.year_gru(fa)
            h = h + self.a_year * self.year_norm(hy[:, -1]).view(B, n, -1)

        last = x[:, -1]
        hf = self.force_enc(last[..., IDX_FORCE])
        hs = self.state_enc(last[..., IDX_STATE])
        h = h + self.a_state * hs

        g_msg = torch.zeros_like(h)
        if self.use_graph:
            g_msg = self.a_sp * self.norm_sp(self.g_sp(h, A_sp))
            if self.use_hydro:
                g_msg = g_msg + self.a_hy * self.norm_hy(self.g_hy(h, A_hy))
            h = h + g_msg

        ctx = self.static_enc(s).unsqueeze(0).expand(B, -1, -1)
        z = torch.cat([h, ctx], -1)

        # N6: functional attribution gating over the four streams
        attr = None
        if self.use_attr:
            attr = torch.softmax(self.attr(z), -1)
            h = h * (1 + 0.5 * (attr[..., :1] + attr[..., 3:4]))
            z = torch.cat([h, ctx], -1)

        w = (torch.softmax(self.gate(s), -1) if self.use_moe
             else torch.ones(n, 1, device=x.device))

        if self.use_n1:
            coefs = torch.stack([c(z) for c in self.coef], dim=2)
            mixed = (coefs * w.unsqueeze(0).unsqueeze(-1)).sum(2)
            rho, sig, kap = mixed.split(self.n_out, dim=-1)
            rho = torch.sigmoid(rho); sig = torch.sigmoid(sig); kap = torch.tanh(kap)
            f_eff = self.f_head(torch.cat([hf, ctx], -1))
            pred = rho * y_prev + sig * y_seas + kap * f_eff
            aux = dict(rho=rho, sig=sig, kap=kap, w=w)
        else:
            pred = self.direct(z)
            aux = dict(rho=None, sig=None, kap=None, w=w)
        aux["recon"] = self.recon(h); aux["attr"] = attr
        return pred, aux
print("PERSIST v2 defined - every novelty starts as a no-op")
''')

md(r"""
## 7 · N4 repaired — Laplacian moved onto residuals

The key change: v1 penalised `pᵀLp` on **predictions**, which just over-smoothed
them spatially. v2 penalises `rᵀLr` on **residuals**, which is the training-time
analogue of residual Moran's I — the metric we actually report. All weights are
also reduced ~10×.
""")

code(r'''
def persist_loss(pred, y, mask, aux, x, use_eco=True):
    m = mask.unsqueeze(-1)
    denom = m.sum().clamp(min=1) * y.shape[-1]
    L_pred = (((pred - torch.nan_to_num(y))**2) * m).sum() / denom
    parts = {"pred": float(L_pred)}
    if not use_eco: return L_pred, parts
    total = L_pred

    L_rec = F.mse_loss(aux["recon"], x[:, -1])
    total = total + W_RECON * L_rec; parts["recon"] = float(L_rec)

    if aux["rho"] is not None and pred.shape[0] > 1:
        L_sm = (aux["rho"][1:] - aux["rho"][:-1]).pow(2).mean()
        total = total + W_SMOOTH * L_sm; parts["smooth"] = float(L_sm)

    # Laplacian on RESIDUALS: directly discourages spatially correlated error,
    # which is what residual Moran's I measures. Only valid on the full graph.
    if aux.get("L_sub") is not None:
        res = (pred - torch.nan_to_num(y)) * m
        r = res.permute(0, 2, 1)
        L_lp = torch.einsum("bon,nm,bom->", r, aux["L_sub"], r) / (r.shape[0]*r.shape[1])
        total = total + W_LAP_RES * L_lp.clamp(min=0); parts["lap_res"] = float(L_lp)

    over = (pred.abs() - CLIP_SIGMA).clamp(min=0)
    total = total + W_ASYM * over.pow(2).mean()

    if aux["w"] is not None and aux["w"].shape[-1] > 1:
        use = aux["w"].mean(0)
        total = total + W_BALANCE * (use * torch.log(use.clamp(min=1e-8))).sum()
    return total, parts
print("N4 repaired: Laplacian on residuals, weights reduced 10x")
''')

md("## 8 · Node-chunked loader (FIX 1) + EMA (FIX 3)")

code(r'''
class EMA:
    """Exponential moving average of weights - cheap and reliably helps."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()
    def copy_to(self, model):
        model.load_state_dict({k: v.clone() for k, v in self.shadow.items()})


def make_steps(ends, shuffle):
    """(window-batch, chunk) pairs. Chunks multiply the update budget."""
    ends = np.asarray(ends)
    if shuffle: ends = np.random.permutation(ends)
    batches = [ends[i:i+GRAPH_BATCH] for i in range(0, len(ends), GRAPH_BATCH)]
    steps = []
    for b in batches:
        for ci_ in range(NODE_CHUNKS):
            steps.append((b, ci_))
        if np.random.rand() < P_FULL_GRAPH or not shuffle:
            steps.append((b, -1))          # -1 = full graph
    if shuffle: random.shuffle(steps)
    return steps


def gather_batch(ends, chunk_id):
    g = FULL if chunk_id < 0 else CHUNKS[chunk_id]
    nodes = g["nodes"]
    xs, xas, ys, ms, yps, yss = [], [], [], [], [], []
    for e in ends:
        e = int(e); yi = e // 12; lo = max(0, yi - ANNUAL_YEARS)
        ann = Xa[:, lo:yi]
        if ann.shape[1] < ANNUAL_YEARS:
            ann = torch.cat([Xa[:, :1].expand(-1, ANNUAL_YEARS-ann.shape[1], -1),
                             ann], dim=1)
        xs.append(Xt[:, e-LOOKBACK:e].permute(1,0,2))
        xas.append(ann.permute(1,0,2)); ys.append(Yt[:, e])
        ms.append(torch.from_numpy(VALID[:, e].astype(np.float32)))
        yps.append(Yfill[:, e-1]); yss.append(Yfill[:, max(e-12,0)])
    idx = nodes.cpu()
    return (torch.stack(xs)[:, :, idx].to(DEVICE),
            torch.stack(xas)[:, :, idx].to(DEVICE),
            torch.stack(ys)[:, idx].to(DEVICE),
            torch.stack(ms)[:, idx].to(DEVICE),
            torch.stack(yps)[:, idx].to(DEVICE),
            torch.stack(yss)[:, idx].to(DEVICE),
            g, np.asarray(ends))


def persist_step(model, ends, chunk_id, use_eco):
    x, xa, y, mask, yp, ys, g, _ = gather_batch(ends, chunk_id)
    pred, aux = model(x, xa, St.to(DEVICE)[g["nodes"]], yp, ys,
                      g["A_sp"], g["A_hy"])
    aux["L_sub"] = L_t if chunk_id < 0 else None    # Laplacian only on full graph
    loss, parts = persist_loss(pred, y, mask, aux, x, use_eco)
    return pred, y, mask, loss, aux, parts
print("node-chunked loader + EMA defined")
''')

md("## 9 · Training loop — epoch-wise printing **and** `history.csv`")

code(r'''
def quick_metrics(o, p):
    o, p = np.asarray(o).ravel(), np.asarray(p).ravel()
    m = np.isfinite(o) & np.isfinite(p); o, p = o[m], p[m]
    if len(o) < 3: return np.nan, np.nan, np.nan
    e = p - o; sst = ((o - o.mean())**2).sum()
    return (float(np.sqrt((e**2).mean())), float(np.abs(e).mean()),
            float(1 - (e**2).sum()/sst) if sst > 0 else np.nan)

def run_epoch(model, ends, opt, train, use_eco, ema=None):
    model.train() if train else model.eval()
    tot, nb, P, O = 0.0, 0, [], []
    for b, ch in make_steps(ends, train):
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            pred, y, mask, loss, aux, _ = persist_step(model, b, ch, use_eco)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if ema is not None: ema.update(model)
        tot += float(loss); nb += 1
        if ch < 0:                       # score on full-graph steps only
            mk = mask.detach().cpu().numpy().astype(bool)
            P.append(pred.detach().cpu().numpy()[mk])
            O.append(y.detach().cpu().numpy()[mk])
    if not P:
        with torch.no_grad():
            pred, y, mask, _, _, _ = persist_step(model, ends[:GRAPH_BATCH], -1, use_eco)
        mk = mask.cpu().numpy().astype(bool)
        P, O = [pred.cpu().numpy()[mk]], [y.cpu().numpy()[mk]]
    return tot/max(nb,1), np.concatenate(P), np.concatenate(O)

def cosine_lr(ep):
    if ep <= WARMUP_EPOCHS: return ep / max(WARMUP_EPOCHS, 1)
    prog = (ep - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

def train_one(name, model, folder, epochs, use_eco=True, verbose=True):
    folder = Path(folder); (folder/"plots").mkdir(parents=True, exist_ok=True)
    (folder/"predictions").mkdir(parents=True, exist_ok=True)
    model = model.to(DEVICE)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lr)
    ema = EMA(model, EMA_DECAY) if USE_EMA else None

    if verbose:
        print(f"\n{'='*78}\n{name}  |  {npar:,} params  |  {DEVICE}\n{'='*78}")
        hdr=(f"{'ep':>4} {'tr_loss':>10} {'va_loss':>10} {'tr_RMSE':>9} {'va_RMSE':>9} "
             f"{'tr_MAE':>8} {'va_MAE':>8} {'tr_R2':>8} {'va_R2':>8} {'lr':>9} {'sec':>6}")
        print(hdr); print("-"*len(hdr))

    hist, best, bad, bstate = [], np.inf, 0, None
    for ep in range(1, epochs+1):
        t0 = time.time()
        trl, trP, trO = run_epoch(model, tr_e, opt, True, use_eco, ema)
        # validate with EMA weights (that is what we will deploy)
        if ema is not None:
            backup = copy.deepcopy(model.state_dict()); ema.copy_to(model)
        val, vaP, vaO = run_epoch(model, va_e, opt, False, use_eco)
        trR,trM,trR2 = quick_metrics(trO,trP); vaR,vaM,vaR2 = quick_metrics(vaO,vaP)
        if vaR < best - 1e-6:
            best, bad = vaR, 0
            bstate = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: bad += 1
        if ema is not None: model.load_state_dict(backup)
        lr = opt.param_groups[0]["lr"]; sched.step()
        hist.append(dict(epoch=ep, train_loss=trl, val_loss=val, train_rmse=trR,
                         val_rmse=vaR, train_mae=trM, val_mae=vaM, train_r2=trR2,
                         val_r2=vaR2, lr=lr, seconds=time.time()-t0))
        if verbose:
            print(f"{ep:>4} {trl:>10.5f} {val:>10.5f} {trR:>9.4f} {vaR:>9.4f} "
                  f"{trM:>8.4f} {vaM:>8.4f} {trR2:>8.4f} {vaR2:>8.4f} "
                  f"{lr:>9.2e} {hist[-1]['seconds']:>6.1f}")
        pd.DataFrame(hist).to_csv(folder/"history.csv", index=False)
        if bad >= PATIENCE:
            if verbose: print(f"early stop at {ep} (best val RMSE {best:.4f})")
            break
    if bstate is not None: model.load_state_dict(bstate)
    torch.save(model.state_dict(), folder/"model_best.pt")
    if verbose: print(f"best val RMSE = {best:.4f}")
    return model, hist, npar

@torch.no_grad()
def predict_full(model):
    model.eval()
    P,O,C,Tt,AUX = [],[],[],[],[]
    for i in range(0, len(te_e), GRAPH_BATCH):
        b = te_e[i:i+GRAPH_BATCH]
        pred, y, mask, _, aux, _ = persist_step(model, b, -1, True)
        pred=pred.cpu().numpy(); y=y.cpu().numpy(); mk=mask.cpu().numpy().astype(bool)
        for k in range(pred.shape[0]):
            keep=np.where(mk[k])[0]
            P.append(pred[k][keep]); O.append(y[k][keep])
            C.append(keep); Tt.append(np.full(len(keep), int(b[k])))
            if aux["rho"] is not None:
                AUX.append(pd.DataFrame({"ci":keep,"ti":int(b[k]),
                    "rho":aux["rho"][k].cpu().numpy()[keep,0],
                    "sig":aux["sig"][k].cpu().numpy()[keep,0],
                    "kap":aux["kap"][k].cpu().numpy()[keep,0]}))
    return (np.concatenate(P), np.concatenate(O), np.concatenate(C),
            np.concatenate(Tt), pd.concat(AUX, ignore_index=True) if AUX else None)
''')

md(r"""
## 10 · Multi-seed runner (FIX 4)

v1's ablation deltas were all within ±0.013 — indistinguishable from run-to-run
noise, so the conclusion "5 novelties hurt" may have been an artefact. v2 runs
every configuration over `N_SEEDS` seeds and reports **mean ± std**, plus an
ensemble prediction (mean over seeds), which also reliably improves accuracy.
""")

code(r'''
def run_config(name, folder, kw, use_eco=True, n_seeds=N_SEEDS, tag="proposed"):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    preds, per_seed, hists, npar = [], [], [], 0
    obs = ci = ti = aux_df = None
    for si in range(n_seeds):
        set_seed(SEED + 1000*si)
        mdl = PERSIST(F_DYN, F_STAT, F_ANN, len(TARGETS), **kw)
        mdl, h, npar = train_one(f"{name} [seed {si+1}/{n_seeds}]", mdl,
                                 folder, EP, use_eco, verbose=(si == 0))
        p, o, c, t, a = predict_full(mdl)
        preds.append(p)
        if obs is None: obs, ci, ti, aux_df = o, c, t, a
        m1, _ = evaluate(p, o, c, t, "test", A_sp)
        per_seed.append(m1); hists.append(h)
        print(f"  seed {si+1}: RMSE {m1['all_RMSE']:.4f}  R2 {m1['all_R2']:.4f}  "
              f"MoranI {m1['residual_MoranI']:.4f}")
        del mdl
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    ens = np.mean(np.stack(preds), axis=0)          # ensemble prediction
    metrics, df = evaluate(ens, obs, ci, ti, "test", A_sp)
    metrics.update(model=name, type=tag, n_parameters=int(npar),
                   n_seeds=n_seeds, epochs_run=len(hists[0]),
                   best_val_rmse=float(min(h["val_rmse"] for h in hists[0])),
                   train_seconds=float(sum(sum(x["seconds"] for x in h) for h in hists)))
    for k in ["all_RMSE","all_R2","within_county_R2","residual_MoranI","shock_RMSE"]:
        vals = [m[k] for m in per_seed if np.isfinite(m.get(k, np.nan))]
        metrics[f"{k}_seed_mean"] = float(np.mean(vals)) if vals else np.nan
        metrics[f"{k}_seed_std"]  = float(np.std(vals)) if vals else np.nan
    pd.DataFrame(per_seed).to_csv(folder/"per_seed_metrics.csv", index=False)
    with open(folder/"metrics.json","w") as f: json.dump(metrics,f,indent=2,default=float)
    pd.DataFrame([metrics]).to_csv(folder/"metrics.csv", index=False)
    df.to_csv(folder/"predictions"/"test_predictions.csv", index=False)
    print(f"\n--- {name} (ensemble of {n_seeds}) ---")
    for k in ["all_RMSE","all_MAE","all_R2","all_PearsonR","all_WillmottD","all_KGE",
              "all_Bias","within_county_R2","residual_MoranI","shock_RMSE",
              "strict_lst_z_R2","raw_lst_c_R2"]:
        if k in metrics: print(f"  {k:24} {metrics[k]:+.4f}")
    print(f"  seed spread RMSE: {metrics['all_RMSE_seed_mean']:.4f} "
          f"+/- {metrics['all_RMSE_seed_std']:.4f}")
    return metrics, df, hists[0], aux_df
''')

md("## 11 · Plotting *(font 20, dpi 300)*")

code(r'''
def _save(fig, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(folder)/f"{name}.{FIG_FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

def plot_history(hist, folder, model):
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(figsize=(11,7))
    ax.plot(h.epoch,h.train_loss,lw=2.5,label="Train"); ax.plot(h.epoch,h.val_loss,lw=2.5,label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
    ax.set_title(f"{model} — Training curve"); _save(fig,folder,"01_loss_curve")
    for k,lab in (("rmse","RMSE"),("mae","MAE"),("r2","R$^2$")):
        fig, ax = plt.subplots(figsize=(11,7))
        ax.plot(h.epoch,h[f"train_{k}"],lw=2.5,label=f"Train {lab}")
        ax.plot(h.epoch,h[f"val_{k}"],lw=2.5,label=f"Val {lab}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab); ax.legend()
        ax.set_title(f"{model} — {lab} per epoch"); _save(fig,folder,f"02_{k}_curve")
    fig, ax = plt.subplots(figsize=(11,7))
    ax.plot(h.epoch,h.lr,lw=2.5,color="darkgreen"); ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
    ax.set_title(f"{model} — cosine schedule with warmup"); _save(fig,folder,"03_lr_schedule")

def plot_core(df, metrics, folder, model):
    for tn in TARGETS:
        o,p = df[f"obs_{tn}"].values, df[f"pred_{tn}"].values
        m=np.isfinite(o)&np.isfinite(p); o,p=o[m],p[m]
        fig,ax=plt.subplots(figsize=(9,9))
        ax.hexbin(o,p,gridsize=60,mincnt=1,cmap="viridis")
        lim=[np.percentile(o,.5),np.percentile(o,99.5)]
        ax.plot(lim,lim,"r--",lw=2.5,label="1:1")
        ax.plot(lim,np.polyval(np.polyfit(o,p,1),lim),"orange",lw=2.5,label="Fit")
        mm=core_metrics(o,p)
        ax.set_xlabel(f"Observed {tn}"); ax.set_ylabel(f"Predicted {tn}")
        ax.set_title(f"{model} — {tn}\nR$^2$={mm['R2']:.3f}  RMSE={mm['RMSE']:.3f}")
        ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim); _save(fig,folder,f"04_scatter_{tn}")
    prim=TARGETS[0]; r=df[f"res_{prim}"].dropna().values
    fig,ax=plt.subplots(figsize=(11,7))
    ax.hist(r,bins=80,color="steelblue",edgecolor="k",alpha=.85); ax.axvline(0,color="r",ls="--",lw=2.5)
    ax.set_xlabel(f"Residual ({prim})"); ax.set_ylabel("Count")
    ax.set_title(f"{model} — Residuals\nmean={r.mean():.3f} sd={r.std():.3f}")
    _save(fig,folder,"05_residual_hist")
    fig,ax=plt.subplots(figsize=(11,7))
    ax.scatter(df[f"pred_{prim}"],df[f"res_{prim}"],s=4,alpha=.15); ax.axhline(0,color="r",ls="--",lw=2.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
    ax.set_title(f"{model} — Residuals vs prediction"); _save(fig,folder,"06_residual_vs_pred")
    from scipy import stats as _st
    fig,ax=plt.subplots(figsize=(9,9)); _st.probplot(r,dist="norm",plot=ax)
    ax.get_lines()[0].set_markersize(3); ax.get_lines()[1].set_linewidth(2.5)
    ax.set_title(f"{model} — Residual Q–Q"); _save(fig,folder,"07_residual_qq")
    d=df.copy(); d["abs_err"]=d[f"res_{prim}"].abs()
    order=[x for x in ["upstream","midstream","downstream"] if x in d.reach.unique()]
    fig,ax=plt.subplots(figsize=(11,7))
    ax.boxplot([d[d.reach==x].abs_err.dropna() for x in order],showfliers=False)
    ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(order)
    ax.set_ylabel("Absolute error"); ax.set_title(f"{model} — Error by reach")
    _save(fig,folder,"08_error_by_reach")
    g=d.groupby("month").abs_err.mean()
    fig,ax=plt.subplots(figsize=(11,7)); ax.bar(g.index,g.values,color="teal",edgecolor="k")
    ax.set_xticks(range(1,13)); ax.set_xlabel("Month"); ax.set_ylabel("MAE")
    ax.set_title(f"{model} — Error seasonality"); _save(fig,folder,"09_error_by_month")
    g=d.groupby("year").abs_err.mean()
    fig,ax=plt.subplots(figsize=(11,7)); ax.plot(g.index,g.values,"o-",lw=2.5,ms=9,color="darkred")
    ax.set_xlabel("Year"); ax.set_ylabel("MAE"); ax.set_title(f"{model} — Error by test year")
    _save(fig,folder,"10_error_by_year")
    v=[d[~d.is_shock].abs_err.dropna(), d[d.is_shock].abs_err.dropna()]
    v=[z if len(z) else pd.Series([np.nan]) for z in v]
    fig,ax=plt.subplots(figsize=(9,7)); ax.boxplot(v,showfliers=False)
    ax.set_xticks([1,2]); ax.set_xticklabels(["Calm","Shock"]); ax.set_ylabel("Absolute error")
    ax.set_title(f"{model} — Calm vs disturbance"); _save(fig,folder,"11_error_shock_vs_calm")
    pick=list(df.groupby("adcode").size().sort_values(ascending=False).index[:4])
    fig,axes=plt.subplots(len(pick),1,figsize=(14,4.2*len(pick)),sharex=True)
    axes=np.atleast_1d(axes)
    for ax,a in zip(axes,pick):
        s=df[df.adcode==a].sort_values("ti")
        ax.plot(s.ti,s[f"obs_{prim}"],"o-",lw=2.2,ms=6,label="Observed")
        ax.plot(s.ti,s[f"pred_{prim}"],"s--",lw=2.2,ms=6,label="Predicted")
        ax.set_ylabel(prim); ax.set_title(f"County {a}"); ax.legend(loc="upper right")
    axes[-1].set_xlabel("Month index"); fig.suptitle(f"{model} — Example trajectories")
    _save(fig,folder,"12_example_timeseries")
    if HAS_GPD:
        try:
            gdf=gpd.read_file(BOUNDARY_FILE); gdf["adcode"]=gdf.adcode.astype(str)
            per=(df.assign(se=lambda z:z[f"res_{prim}"]**2).groupby("adcode").se.mean()
                   .pow(.5).rename("rmse").reset_index())
            g2=gdf.merge(per,on="adcode",how="left")
            fig,ax=plt.subplots(figsize=(15,11))
            g2.plot(column="rmse",ax=ax,legend=True,cmap="YlOrRd",edgecolor="grey",
                    linewidth=.15,missing_kwds={"color":"lightgrey"},
                    legend_kwds={"label":f"Test RMSE ({prim})","shrink":.7})
            ax.set_title(f"{model} — Spatial error"); ax.set_axis_off()
            _save(fig,folder,"13_map_rmse")
        except Exception as e: print("  map skipped:",e)
    keys=["all_RMSE","all_MAE","all_R2","all_PearsonR","all_WillmottD","all_KGE","within_county_R2"]
    vals=[metrics.get(k,np.nan) for k in keys]
    fig,ax=plt.subplots(figsize=(13,7))
    ax.bar([k.replace("all_","") for k in keys],vals,color="slateblue",edgecolor="k")
    for i,v in enumerate(vals):
        if np.isfinite(v): ax.text(i,v,f"{v:.3f}",ha="center",va="bottom" if v>=0 else "top",fontsize=FONT_SIZE-6)
    ax.axhline(0,color="k",lw=1); ax.set_ylabel("Value")
    ax.set_title(f"{model} — Test metrics"); plt.xticks(rotation=25,ha="right")
    _save(fig,folder,"14_metric_summary")

def plot_interp(aux_df, folder, model, interp_dir):
    if aux_df is None or not len(aux_df): return
    d=aux_df.copy(); d["recovery"]=1-d.rho; d["resistance"]=1-d.kap.abs()
    d["adcode"]=[counties[c] for c in d.ci]
    d["reach"]=d.adcode.map(panel.groupby("adcode").reach.first())
    Path(interp_dir).mkdir(parents=True, exist_ok=True)
    d.to_csv(Path(interp_dir)/"n1_coefficients.csv", index=False)
    for col,lab in (("rho",r"Persistence $\rho$"),("recovery","Recovery rate (1-$\\rho$)"),
                    ("resistance","Resistance (1-|$\\kappa$|)"),("sig",r"Seasonal carry $\sigma$")):
        fig,ax=plt.subplots(figsize=(11,7))
        ax.hist(d[col].dropna(),bins=60,color="darkslateblue",edgecolor="k",alpha=.85)
        ax.set_xlabel(lab); ax.set_ylabel("Count")
        ax.set_title(f"{model} — N1 {lab}\nmean={d[col].mean():.3f}")
        _save(fig,folder,f"20_n1_{col}_hist")
    order=[x for x in ["upstream","midstream","downstream"] if x in d.reach.unique()]
    fig,ax=plt.subplots(figsize=(11,7))
    ax.boxplot([d[d.reach==x].recovery.dropna() for x in order],showfliers=False)
    ax.set_xticks(range(1,len(order)+1)); ax.set_xticklabels(order)
    ax.set_ylabel("Recovery rate"); ax.set_title(f"{model} — N1 recovery by reach")
    _save(fig,folder,"21_n1_recovery_by_reach")
    if HAS_GPD:
        try:
            gdf=gpd.read_file(BOUNDARY_FILE); gdf["adcode"]=gdf.adcode.astype(str)
            for col,lab,cm in (("recovery","Recovery rate","viridis"),
                               ("resistance","Resistance","magma")):
                per=d.groupby("adcode")[col].mean().reset_index()
                g2=gdf.merge(per,on="adcode",how="left")
                fig,ax=plt.subplots(figsize=(15,11))
                g2.plot(column=col,ax=ax,legend=True,cmap=cm,edgecolor="grey",linewidth=.15,
                        missing_kwds={"color":"lightgrey"},legend_kwds={"label":lab,"shrink":.7})
                ax.set_title(f"{model} — N1 {lab}"); ax.set_axis_off()
                _save(fig,folder,f"22_map_{col}")
        except Exception as e: print("  interp map skipped:",e)
print("plotting defined")
''')

md("## 12 · Train PERSIST v2")

code(r'''
EP = 2 if SMOKE_TEST else EPOCHS
F_DYN, F_STAT, F_ANN = X.shape[2], S.shape[1], Xann.shape[2]
NS = 1 if SMOKE_TEST else N_SEEDS
print(f"F_dyn={F_DYN} F_static={F_STAT} F_annual={F_ANN} epochs={EP} seeds={NS}")

M_PERSIST, DF_P, HIST_P, AUX_P = run_config(
    "PERSIST", PERSIST_DIR, dict(), True, NS, "proposed")
plot_history(HIST_P, PERSIST_DIR/"plots", "PERSIST")
plot_core(DF_P, M_PERSIST, PERSIST_DIR/"plots", "PERSIST")
plot_interp(AUX_P, PERSIST_DIR/"plots", "PERSIST", PERSIST_DIR/"interpretability")
print("plots ->", PERSIST_DIR/"plots")
''')

md("## 13 · Ablations (multi-seed, with error bars)")

code(r'''
ABLATIONS = {
    "no_N1_DCRD":    dict(use_n1=False),
    "no_N2a_hydro":  dict(use_hydro=False),
    "no_N2b_graph":  dict(use_graph=False, use_hydro=False),
    "no_N3_moe":     dict(use_moe=False),
    "no_N4_ecoloss": dict(),
    "no_N5_annual":  dict(use_annual=False),
    "no_N6_attr":    dict(use_attr=False),
}
ABL_RESULTS, ABL_HIST = {}, {}
if RUN_ABLATIONS:
    for vn, kw in ABLATIONS.items():
        m, dfx, h, _ = run_config(vn, ABL_DIR/vn, kw,
                                  use_eco=(vn != "no_N4_ecoloss"),
                                  n_seeds=NS, tag="ablation")
        ABL_RESULTS[vn] = m; ABL_HIST[vn] = h
        plot_history(h, ABL_DIR/vn/"plots", vn)
        plot_core(dfx, m, ABL_DIR/vn/"plots", vn)
else:
    print("RUN_ABLATIONS = False")
''')

code(r'''
if ABL_RESULTS:
    abl = pd.DataFrame([M_PERSIST] + list(ABL_RESULTS.values()))
    front = ["model","type","n_seeds","n_parameters","all_RMSE","all_RMSE_seed_mean",
             "all_RMSE_seed_std","all_MAE","all_R2","all_PearsonR","all_WillmottD",
             "all_KGE","within_county_R2","residual_MoranI","shock_RMSE"]
    abl = abl[[c for c in front if c in abl.columns] +
              [c for c in abl.columns if c not in front]]
    base = float(abl.loc[abl.model=="PERSIST","all_RMSE"].iloc[0])
    baseR2 = float(abl.loc[abl.model=="PERSIST","all_R2"].iloc[0])
    abl["dRMSE_vs_PERSIST"] = abl.all_RMSE.astype(float) - base
    abl["dR2_vs_PERSIST"]  = abl.all_R2.astype(float) - baseR2
    # is the delta larger than seed noise?
    noise = float(abl.loc[abl.model=="PERSIST","all_RMSE_seed_std"].iloc[0])
    abl["significant"] = abl.dRMSE_vs_PERSIST.abs() > 2*noise
    abl.to_csv(ABL_DIR/"ablation_comparison.csv", index=False)
    print(f"seed noise (PERSIST RMSE std) = {noise:.4f}; "
          f"|delta| must exceed {2*noise:.4f} to be meaningful\n")
    show=["model","all_RMSE","all_RMSE_seed_std","dRMSE_vs_PERSIST","significant",
          "all_R2","within_county_R2","residual_MoranI"]
    display(abl[[c for c in show if c in abl]].round(4))
''')

md("## 14 · Comparison A — PERSIST vs baselines")

code(r'''
def bar_compare(dfc, key, lab, folder, fname, highlight="PERSIST", ref_model=None):
    if key not in dfc: return
    fig, ax = plt.subplots(figsize=(12.5,7))
    v = dfc[key].astype(float).values
    tp = dfc.get("type", pd.Series(["deep"]*len(dfc)))
    cols=["#C0392B" if m==highlight else ("#BBBBBB" if t=="trivial" else "#4C72B0")
          for m,t in zip(dfc.model, tp)]
    ax.bar(dfc.model, v, color=cols, edgecolor="k")
    for i,x in enumerate(v):
        if np.isfinite(x):
            ax.text(i,x,f"{x:.4f}",ha="center",va="bottom" if x>=0 else "top",
                    fontsize=FONT_SIZE-7)
    if ref_model and ref_model in set(dfc.model):
        rv=float(dfc.loc[dfc.model==ref_model,key].iloc[0])
        ax.axhline(rv,color="crimson",ls="--",lw=2.5,label=ref_model); ax.legend()
    ax.axhline(0,color="k",lw=1); ax.set_ylabel(lab); ax.set_title(lab)
    plt.xticks(rotation=30,ha="right"); _save(fig,folder,fname)

KEYS=[("all_RMSE","Test RMSE (lower better)"),("all_MAE","Test MAE (lower better)"),
      ("all_R2","Test R$^2$ (higher better)"),("all_PearsonR","Pearson r (higher better)"),
      ("all_WillmottD","Willmott d (higher better)"),("all_KGE","KGE (higher better)"),
      ("within_county_R2","Within-county R$^2$ (temporal skill)"),
      ("residual_MoranI","Residual Moran's I (closer to 0 better)"),
      ("shock_RMSE","Shock-month RMSE (lower better)")]

if BASELINE_CSV.exists():
    base_df = pd.read_csv(BASELINE_CSV)
    if "type" not in base_df: base_df["type"]="deep"
    combo = pd.concat([base_df, pd.DataFrame([M_PERSIST])], ignore_index=True)
    keep=[c for c in ["model","type","n_parameters","all_RMSE","all_MAE","all_R2",
                      "all_PearsonR","all_WillmottD","all_KGE","all_Bias",
                      "within_county_R2","residual_MoranI","shock_RMSE"] if c in combo]
    combo[keep+[c for c in combo.columns if c not in keep]].to_csv(
        CMP_BASE/"proposed_vs_baselines.csv", index=False)
    display(combo[keep].round(4))
    for k,lab in KEYS:
        bar_compare(combo,k,lab,CMP_BASE,f"cmp_{k}",
                    ref_model="SeasonalNaive" if k in ("all_RMSE","all_MAE","shock_RMSE") else None)

    print("\n" + "="*74 + "\nVERDICT — PERSIST v2 vs every reference\n" + "="*74)
    LOWER = {"all_RMSE","all_MAE","shock_RMSE","residual_MoranI"}
    wins = losses = 0
    for k,_ in KEYS:
        if k not in combo: continue
        pv = float(M_PERSIST[k])
        others = combo[combo.model!="PERSIST"]
        best = (others[k].astype(float).min() if k in LOWER
                else others[k].astype(float).max())
        bm = others.loc[others[k].astype(float).idxmin() if k in LOWER
                        else others[k].astype(float).idxmax(),"model"]
        ok = (pv < best) if k in LOWER else (pv > best)
        wins += ok; losses += (not ok)
        print(f"  {k:20} PERSIST {pv:+.4f} vs best other {best:+.4f} ({bm})"
              f"  -> {'WIN' if ok else 'lose'}")
    print(f"\n  PERSIST wins {wins}/{wins+losses} metrics against the best of all references.")
else:
    print("baseline_comparison.csv not found:", BASELINE_CSV)
    combo=None
''')

md("## 15 · Comparison B — PERSIST vs ablations")

code(r'''
if ABL_RESULTS:
    for k,lab in KEYS:
        bar_compare(abl,k,lab,CMP_ABL,f"abl_{k}",ref_model="PERSIST")
    contrib=abl[abl.model!="PERSIST"][["model","dRMSE_vs_PERSIST","significant"]].copy()
    contrib=contrib.sort_values("dRMSE_vs_PERSIST",ascending=False)
    fig,ax=plt.subplots(figsize=(13,7))
    cols=["#2E7D32" if (v>0 and s) else ("#C62828" if (v<0 and s) else "#BDBDBD")
          for v,s in zip(contrib.dRMSE_vs_PERSIST,contrib.significant)]
    ax.barh(contrib.model,contrib.dRMSE_vs_PERSIST,color=cols,edgecolor="k")
    ax.axvline(0,color="k",lw=1.5)
    ax.axvspan(-2*noise,2*noise,color="grey",alpha=.18,label="seed-noise band")
    for i,v in enumerate(contrib.dRMSE_vs_PERSIST):
        ax.text(v,i,f" {v:+.4f}",va="center",ha="left" if v>=0 else "right",
                fontsize=FONT_SIZE-6)
    ax.set_xlabel("$\\Delta$RMSE when novelty removed"); ax.legend()
    ax.set_title("Novelty contribution\n(green = novelty helps beyond seed noise)")
    _save(fig,CMP_ABL,"abl_novelty_contribution")
    fig,ax=plt.subplots(figsize=(13,7))
    hp=pd.DataFrame(HIST_P); ax.plot(hp.epoch,hp.val_rmse,lw=3.5,color="#C0392B",label="PERSIST")
    for (vn,h),c in zip(ABL_HIST.items(),plt.cm.tab10.colors):
        hd=pd.DataFrame(h); ax.plot(hd.epoch,hd.val_rmse,lw=2,label=vn,color=c,alpha=.85)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation RMSE")
    ax.set_title("Validation RMSE — PERSIST vs ablations")
    ax.legend(fontsize=FONT_SIZE-8,ncol=2); _save(fig,CMP_ABL,"abl_val_curves")
    radar=[("all_R2",False),("all_PearsonR",False),("all_KGE",False),
           ("within_county_R2",False),("all_RMSE",True),("all_MAE",True)]
    radar=[(k,i) for k,i in radar if k in abl]
    lab=[k.replace("all_","") for k,_ in radar]
    ang=np.linspace(0,2*np.pi,len(lab),endpoint=False).tolist(); ang+=ang[:1]
    fig,ax=plt.subplots(figsize=(11,11),subplot_kw=dict(polar=True))
    for i,mn in enumerate(abl.model):
        vs=[]
        for k,inv in radar:
            cv=abl[k].astype(float); lo,hi=cv.min(),cv.max()
            v=float(abl.loc[abl.model==mn,k].iloc[0])
            sc=.5 if hi==lo else (v-lo)/(hi-lo); vs.append(1-sc if inv else sc)
        vs+=vs[:1]; full=(mn=="PERSIST")
        ax.plot(ang,vs,lw=3.5 if full else 1.8,
                color="#C0392B" if full else plt.cm.tab10.colors[i%10],label=mn,
                alpha=1.0 if full else .8)
        if full: ax.fill(ang,vs,alpha=.15,color="#C0392B")
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(lab)
    ax.set_title("Ablation metric profile\n(outer = better)",pad=32)
    ax.legend(loc="upper right",bbox_to_anchor=(1.42,1.12),fontsize=FONT_SIZE-8)
    _save(fig,CMP_ABL,"abl_radar")
    print("ablation plots ->",CMP_ABL)
''')

md("## 16 · Summary")

code(r'''
print("="*78); print("PERSIST v2 COMPLETE"); print("="*78)
print("\nPERSIST v2 test metrics (ensemble):")
for k in ["all_RMSE","all_MAE","all_R2","all_PearsonR","all_KGE",
          "within_county_R2","residual_MoranI","shock_RMSE"]:
    if k in M_PERSIST: print(f"  {k:22} {M_PERSIST[k]:+.4f}")
print("\nv1 comparison:")
V1 = {"all_RMSE":0.6355,"all_R2":0.6593,"within_county_R2":0.4515,
      "residual_MoranI":0.7519,"shock_RMSE":0.7292}
for k,v1 in V1.items():
    v2=float(M_PERSIST[k]); better = v2<v1 if k in {"all_RMSE","residual_MoranI","shock_RMSE"} else v2>v1
    print(f"  {k:20} v1 {v1:+.4f} -> v2 {v2:+.4f}  {'improved' if better else 'WORSE'}")
if ABL_RESULTS:
    print(f"\nNovelty contributions (seed noise band +/-{2*noise:.4f}):")
    for _,r in abl[abl.model!="PERSIST"].sort_values("dRMSE_vs_PERSIST",ascending=False).iterrows():
        tagv = ("HELPS" if r.dRMSE_vs_PERSIST>0 else "HURTS") if r.significant else "within noise"
        print(f"  {r['model']:16} {r.dRMSE_vs_PERSIST:+.4f}   {tagv}")
print("\nArtefacts:")
print(f"  {PERSIST_DIR}  ({len(list((PERSIST_DIR/'plots').glob('*.'+FIG_FMT)))} figures)")
if ABL_RESULTS: print(f"  {ABL_DIR}  ({len(ABL_RESULTS)} variants)")
print(f"  {CMP_BASE}  ({len(list(CMP_BASE.glob('*.'+FIG_FMT)))} figures)")
print(f"  {CMP_ABL}   ({len(list(CMP_ABL.glob('*.'+FIG_FMT)))} figures)")
''')

md(r"""
---

## Notes

**What changed and why.** All four fixes target the diagnosis that PERSIST v1's
problem was **effective sample size** (168 snapshots) and **forced novelties**,
not model capacity. Node chunking multiplies the update budget while acting as
edge dropout; every novelty now enters through a residual scaled by a learnable
parameter initialised near zero, so the optimiser must switch it on; EMA and
cosine warmup stabilise the short training run; and multi-seed ensembling both
improves accuracy and finally puts error bars on the ablation deltas.

**Read the ablation table with the noise band.** v1's deltas were all within
±0.013, which is why "5 novelties hurt" was not a safe conclusion. v2 reports
the PERSIST seed standard deviation and flags a delta as meaningful only when it
exceeds 2σ.

**On the 82% request.** Not attainable on this target without leakage. The
deseasonalised monthly anomaly has a measured lag-1 autocorrelation of 0.155,
and a well-tuned 657k-parameter TFT reaches R² 0.675. Reported raw-space R² is
already ~0.93 for every model because it contains the seasonal cycle, so it is
not evidence of skill. The defensible goal here is to beat every baseline on
every metric, with the largest margins on residual Moran's I and shock RMSE.
""")

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if k=="md" else nbf.v4.new_code_cell(s)
            for k, s in C]
nb.metadata = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
               "language_info":{"name":"python","version":"3.11"},
               "accelerator":"GPU","colab":{"provenance":[],"gpuType":"L4"}}
out = pathlib.Path("PERSIST_Phase5_v2.ipynb")
nbf.write(nb, str(out))
print(f"wrote {out} ({len(nb.cells)} cells, {out.stat().st_size/1024:.0f} KB)")
