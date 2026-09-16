#!/usr/bin/env python3
"""Generate the end-to-end baselines notebook for PERSIST (Phase 4).

Three deep-learning baselines: DRSEI (AE+LSTM), STGCN, TFT.
Single notebook, Colab/L4 ready, only the CONFIG cell needs editing.
"""
import pathlib

import nbformat as nbf

C = []          # (kind, source)


def md(s):
    C.append(("md", s.strip("\n")))


def code(s):
    C.append(("code", s.strip("\n")))


# ══════════════════════════════════════════════════════════════════ TITLE
md(r"""
# PERSIST — Phase 4: Deep Learning Baselines

**Ecological Resilience in the Yangtze River Economic Belt**

Three deep-learning baselines, each isolating a different PERSIST novelty:

| # | Baseline | Architecture | Isolates |
|---|---|---|---|
| 1 | **DRSEI** | Autoencoder + LSTM | N2 (dual graph), N3 (regime MoE) — temporal only, no spatial |
| 2 | **STGCN** | Graph conv + gated temporal conv | N1 (disturbance-response), N4 (ecological loss) — same info, generic objective |
| 3 | **TFT** | Temporal Fusion Transformer | N1, N5 (hierarchy), N6 (attribution) — attention + static covariates + interpretability |

**Task.** Forecast next-month ecological state anomalies (`LST_z`, `kNDVI_z`) from a lookback window of state + forcing + static context. Genuinely held-out in time, so metrics are non-circular and directly comparable to the proposed model.

**Split.** By **time**, never randomly — random splits leak spatially and temporally autocorrelated information.

**Outputs.** `outputs/<baseline>/` per model with `history.csv`, `metrics.json`, `metrics.csv`, predictions and all plots; plus a top-level `baseline_comparison.csv`.

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
MOUNT_DRIVE = True                       # False if not on Colab
REPO_ROOT   = Path("/content/drive/MyDrive/Assesment_of_Ecological_resilience_Yangtze")

# --- derived paths (normally leave alone) ------------------------------
DATA_DIR      = REPO_ROOT / "phase3_data" / "tables"
BOUNDARY_FILE = REPO_ROOT / "phase3_data" / "boundaries" / "yreb_counties_datav.gpkg"
OUTPUT_DIR    = REPO_ROOT / "outputs"

PANEL_FILE   = DATA_DIR / "panel_monthly.parquet"
ADJ_FILE     = DATA_DIR / "adjacency_edges.csv"
HYDRO_FILE   = DATA_DIR / "hydro_edges.csv"
KARST_FILE   = DATA_DIR / "karst_county.csv"
TERRAIN_FILE = DATA_DIR / "terrain_county.csv"

# --- run mode ---------------------------------------------------------
SMOKE_TEST = False        # True = tiny subset + 2 epochs, for a quick check
SEED       = 42

# --- task -------------------------------------------------------------
LOOKBACK   = 12           # months of history fed to the model
HORIZON    = 1            # forecast 1 month ahead
TARGETS    = ["lst_z", "kndvi_z"]      # response channels (LST is N1 primary)

# --- temporal split (by calendar year, inclusive) ---------------------
TRAIN_YEARS = (2000, 2014)
VAL_YEARS   = (2015, 2017)
TEST_YEARS  = (2018, 2020)

# --- training ---------------------------------------------------------
EPOCHS        = 60
BATCH_SIZE    = 512       # per-county models (DRSEI, TFT)
GRAPH_BATCH   = 8         # graph snapshots per batch (STGCN)
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 12        # early stopping on val RMSE
GRAD_CLIP     = 1.0

# --- model sizes ------------------------------------------------------
DRSEI_LATENT  = 32
DRSEI_HIDDEN  = 64
STGCN_CHANNELS = 48
TFT_HIDDEN    = 64
TFT_HEADS     = 4
DROPOUT       = 0.1

# --- plotting ---------------------------------------------------------
FONT_SIZE = 20
DPI       = 300
FIG_FMT   = "png"

# --- QC (from Phase 3e: LST clear-sky bias is MODERATE) ---------------
USE_NREL_COVARIATE = True    # include relative sampling density as a feature
SHOCK_THRESHOLD    = 1.5     # |heat_z| defining a disturbance month
# ════════════════════════════════════════════════════════════════════════
print("CONFIG loaded.")
print("  repo   :", REPO_ROOT)
print("  panel  :", PANEL_FILE)
print("  outputs:", OUTPUT_DIR)
print("  SMOKE_TEST =", SMOKE_TEST)
''')

# ═══════════════════════════════════════════════════════════════════ ENV
md("## 2 · Environment, dependencies, GPU")

code(r'''
import os, sys, json, math, time, warnings, random
warnings.filterwarnings("ignore")

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception as e:
        print("Drive mount skipped:", e)

# geopandas is needed only for the choropleth maps
try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    print("installing geopandas ...")
    os.system(f"{sys.executable} -m pip install -q geopandas")
    try:
        import geopandas as gpd
        HAS_GPD = True
    except Exception:
        HAS_GPD = False
        print("geopandas unavailable - maps will be skipped")

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

# global plot style
plt.rcParams.update({
    "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE - 2,
    "ytick.labelsize": FONT_SIZE - 2, "legend.fontsize": FONT_SIZE - 4,
    "figure.titlesize": FONT_SIZE + 2, "savefig.dpi": DPI,
    "figure.dpi": 100, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.3, "figure.autolayout": False,
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINES = ["DRSEI_AE_LSTM", "STGCN", "TFT"]
for b in BASELINES:
    (OUTPUT_DIR / b / "plots").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / b / "predictions").mkdir(parents=True, exist_ok=True)
print("output folders ready under", OUTPUT_DIR)
''')

# ══════════════════════════════════════════════════════════════════ LOAD
md("""
## 3 · Load data

`panel_monthly.parquet` — 269,136 rows × 58 cols (1,068 counties × 21 years × 12 months).
""")

code(r'''
t0 = time.time()
panel = pd.read_parquet(PANEL_FILE)
panel["adcode"] = panel["adcode"].astype(str)
print(f"panel: {panel.shape[0]:,} rows x {panel.shape[1]} cols  "
      f"({time.time()-t0:.1f}s)")
print(f"counties {panel.adcode.nunique()} | years "
      f"{panel.year.min()}-{panel.year.max()}")

adj   = pd.read_csv(ADJ_FILE,   dtype={"u_adcode": str, "v_adcode": str})
hydro = pd.read_csv(HYDRO_FILE, dtype={"src": str, "dst": str})
print(f"spatial edges {len(adj)} | hydro edges {len(hydro)}")

# a global monthly time index
panel["t"] = (panel.year - panel.year.min()) * 12 + (panel.month - 1)
panel = panel.sort_values(["adcode", "t"]).reset_index(drop=True)

display(panel.head(3))
print("\ncolumns:", list(panel.columns))
''')

# ═══════════════════════════════════════════════════════════ PREPROCESS
md(r"""
## 4 · Preprocessing

Three points that matter for correctness:

1. **Anomalies are recomputed using the *training period only*.** The panel's shipped `*_z` columns use full-period climatology, which leaks test information into the target definition. We rebuild them from raw values with train-only county×month means and standard deviations.
2. **Feature scaling is fitted on train only.**
3. **`n_rel`** (relative LST sampling density) is included as a covariate, per the Phase 3e finding that clear-sky sampling leaks ~0.20 correlation into the LST anomaly.
""")

code(r'''
# ---- optional smoke-test subset --------------------------------------
if SMOKE_TEST:
    keep = sorted(panel.adcode.unique())[:60]
    panel = panel[panel.adcode.isin(keep)].copy()
    print(f"SMOKE_TEST: reduced to {panel.adcode.nunique()} counties")

counties = sorted(panel.adcode.unique())
cidx = {c: i for i, c in enumerate(counties)}
N = len(counties)
T = panel.t.nunique()
print(f"N counties = {N} | T months = {T}")

# ---- relative LST sampling density (Phase 3e QC) ---------------------
if "lst_day_n" in panel:
    med = panel.groupby("adcode")["lst_day_n"].transform("median").replace(0, np.nan)
    panel["n_rel"] = (panel["lst_day_n"] / med).fillna(1.0)
else:
    panel["n_rel"] = 1.0

# ---- train-only climatology -> leakage-free anomalies ----------------
train_mask = panel.year.between(*TRAIN_YEARS)
RAW_TARGETS = {"lst_z": "lst_c", "kndvi_z": "kndvi"}

for zname, raw in RAW_TARGETS.items():
    clim = (panel[train_mask].groupby(["adcode", "month"])[raw]
            .agg(["mean", "std"]).reset_index()
            .rename(columns={"mean": f"{raw}_mu", "std": f"{raw}_sd"}))
    panel = panel.merge(clim, on=["adcode", "month"], how="left")
    panel[zname] = (panel[raw] - panel[f"{raw}_mu"]) / \
                   panel[f"{raw}_sd"].replace(0, np.nan)
    panel[zname] = panel[zname].replace([np.inf, -np.inf], np.nan)
print("targets rebuilt on train-only climatology:", TARGETS)
print(panel[TARGETS].describe().round(3).to_string())

# ---- feature groups --------------------------------------------------
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

DYN_FEATS = STATE_FEATS + FORCE_FEATS + HUMAN_FEATS
print(f"\ndynamic features ({len(DYN_FEATS)}): {DYN_FEATS}")
print(f"static  features ({len(STATIC_FEATS)}): {STATIC_FEATS}")

# month-of-year as cyclic encoding (seasonality, cheap and effective)
panel["moy_sin"] = np.sin(2 * np.pi * panel.month / 12)
panel["moy_cos"] = np.cos(2 * np.pi * panel.month / 12)
DYN_FEATS = DYN_FEATS + ["moy_sin", "moy_cos"]
''')

code(r'''
# ---- dense (N, T, F) tensors ----------------------------------------
panel["ci"] = panel.adcode.map(cidx)
tvals = np.sort(panel.t.unique())
tpos = {v: i for i, v in enumerate(tvals)}
panel["ti"] = panel.t.map(tpos)

X = np.full((N, T, len(DYN_FEATS)), np.nan, dtype=np.float32)
Y = np.full((N, T, len(TARGETS)), np.nan, dtype=np.float32)
X[panel.ci.values, panel.ti.values, :] = panel[DYN_FEATS].values.astype(np.float32)
Y[panel.ci.values, panel.ti.values, :] = panel[TARGETS].values.astype(np.float32)

stat_df = (panel.groupby("adcode")[STATIC_FEATS].first()
           .reindex(counties).astype(np.float32))
S = stat_df.values

year_of = panel.groupby("ti").year.first().reindex(range(T)).values
month_of = panel.groupby("ti").month.first().reindex(range(T)).values
print("X", X.shape, "| Y", Y.shape, "| S", S.shape)
print(f"NaN share  X={np.isnan(X).mean():.4f}  Y={np.isnan(Y).mean():.4f}")

# ---- impute dynamic NaNs by county-wise forward/backward fill --------
for f in range(X.shape[2]):
    col = X[:, :, f]
    idx = np.where(~np.isnan(col), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    col = col[np.arange(N)[:, None], idx]
    # backward pass for leading NaNs
    rev = col[:, ::-1]
    idx2 = np.where(~np.isnan(rev), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx2, axis=1, out=idx2)
    col = rev[np.arange(N)[:, None], idx2][:, ::-1]
    X[:, :, f] = np.nan_to_num(col, nan=0.0)
S = np.nan_to_num(S, nan=0.0)
print(f"after fill: NaN share X={np.isnan(X).mean():.4f}")

# ---- train-only scaling ---------------------------------------------
tr_t = np.where((year_of >= TRAIN_YEARS[0]) & (year_of <= TRAIN_YEARS[1]))[0]
va_t = np.where((year_of >= VAL_YEARS[0])   & (year_of <= VAL_YEARS[1]))[0]
te_t = np.where((year_of >= TEST_YEARS[0])  & (year_of <= TEST_YEARS[1]))[0]
print(f"months  train {len(tr_t)} | val {len(va_t)} | test {len(te_t)}")

mu = X[:, tr_t, :].reshape(-1, X.shape[2]).mean(0)
sd = X[:, tr_t, :].reshape(-1, X.shape[2]).std(0)
sd[sd == 0] = 1.0
X = (X - mu) / sd

smu, ssd = S.mean(0), S.std(0)
ssd[ssd == 0] = 1.0
S = (S - smu) / ssd
print("features scaled on train statistics only")
''')

code(r'''
# ---- window index construction --------------------------------------
# a sample ends at time e (target time); inputs span [e-LOOKBACK, e-1]
def windows_for(times):
    out = []
    tset = set(times.tolist())
    for e in times:
        if e - LOOKBACK < 0:
            continue
        if all((e - k) >= 0 for k in range(1, LOOKBACK + 1)):
            out.append(e)
    return np.array(sorted(set(out) & tset), dtype=np.int64)

tr_e, va_e, te_e = windows_for(tr_t), windows_for(va_t), windows_for(te_t)
print(f"target months  train {len(tr_e)} | val {len(va_e)} | test {len(te_e)}")

VALID = ~np.isnan(Y).any(axis=2)          # (N, T) target availability
print(f"valid (county, month) targets: {VALID.sum():,} / {N*T:,}")

def flat_samples(ends):
    """(county, end_time) pairs with a valid target."""
    cs, ts = [], []
    for e in ends:
        ok = np.where(VALID[:, e])[0]
        cs.append(ok); ts.append(np.full(len(ok), e))
    return np.concatenate(cs), np.concatenate(ts)

tr_ci, tr_ti = flat_samples(tr_e)
va_ci, va_ti = flat_samples(va_e)
te_ci, te_ti = flat_samples(te_e)
print(f"flat samples  train {len(tr_ci):,} | val {len(va_ci):,} | "
      f"test {len(te_ci):,}")

Xt = torch.from_numpy(X)
Yt = torch.from_numpy(Y)
St = torch.from_numpy(S)

class SeqDS(torch.utils.data.Dataset):
    """Per-county windows for DRSEI and TFT."""
    def __init__(self, ci, ti):
        self.ci = torch.from_numpy(ci); self.ti = torch.from_numpy(ti)
    def __len__(self): return len(self.ci)
    def __getitem__(self, i):
        c, e = int(self.ci[i]), int(self.ti[i])
        return (Xt[c, e - LOOKBACK:e],      # (L, F)
                St[c],                       # (Fs,)
                Yt[c, e],                    # (n_targets,)
                self.ci[i], self.ti[i])

def loader(ci, ti, bs, shuffle):
    return torch.utils.data.DataLoader(SeqDS(ci, ti), batch_size=bs,
                                       shuffle=shuffle, num_workers=0,
                                       drop_last=False)
''')

# ═══════════════════════════════════════════════════════════════ GRAPH
md("## 5 · Graph construction (for STGCN)")

code(r'''
def build_norm_adj(edge_df, a_col, b_col, directed=False):
    """Symmetric-normalised adjacency with self loops: D^-1/2 (A+I) D^-1/2."""
    A = np.zeros((N, N), dtype=np.float32)
    hit = 0
    for u, v in zip(edge_df[a_col], edge_df[b_col]):
        if u in cidx and v in cidx:
            A[cidx[u], cidx[v]] = 1.0
            if not directed:
                A[cidx[v], cidx[u]] = 1.0
            hit += 1
    A = A + np.eye(N, dtype=np.float32)
    d = A.sum(1)
    dinv = np.power(d, -0.5, where=d > 0)
    dinv[np.isinf(dinv)] = 0
    return (A * dinv[:, None] * dinv[None, :]).astype(np.float32), hit

A_sp, n_sp = build_norm_adj(adj, "u_adcode", "v_adcode", directed=False)
A_hy, n_hy = build_norm_adj(hydro, "src", "dst", directed=True)
print(f"spatial adjacency: {n_sp} edges mapped | density {(A_sp>0).mean():.5f}")
print(f"hydro   adjacency: {n_hy} edges mapped | density {(A_hy>0).mean():.5f}")

# STGCN baseline deliberately uses the SPATIAL graph only.
# The dual-graph fusion is a PERSIST novelty (N2) and must not be given away.
A_t = torch.from_numpy(A_sp).to(DEVICE)
print("STGCN will use the spatial graph only (dual-graph = PERSIST N2)")
''')

# ══════════════════════════════════════════════════════════════ METRICS
md(r"""
## 6 · Metrics

Chosen so the **same set applies unchanged to the proposed model**:

| Metric | Why |
|---|---|
| RMSE, MAE | standard error magnitude |
| R² (1 − SSE/SST) | variance explained |
| Pearson r | pattern agreement independent of bias |
| Willmott's d | index of agreement, standard in ecological modelling |
| KGE | Kling–Gupta, decomposes correlation / variability / bias |
| Bias (ME) | systematic over- or under-prediction |
| **Residual Moran's I** | **cleanest single test of the graph contribution (N2)** — should approach 0 |
| **Shock RMSE** | error on disturbance months only (N1) — resilience is about response to real shocks |
| Per-reach RMSE | upstream / midstream / downstream generalisation (N3) |

Note KGE's bias term is normalised by the observed standard deviation rather than the mean, because targets are standardised anomalies with means near zero.
""")

code(r'''
def _finite(o, p):
    m = np.isfinite(o) & np.isfinite(p)
    return o[m], p[m]

def willmott_d(o, p):
    o, p = _finite(o, p)
    if len(o) < 2: return np.nan
    om = o.mean()
    den = np.sum((np.abs(p - om) + np.abs(o - om)) ** 2)
    return 1 - np.sum((o - p) ** 2) / den if den > 0 else np.nan

def kge(o, p):
    """Kling-Gupta with sd-normalised bias (targets are z-anomalies)."""
    o, p = _finite(o, p)
    if len(o) < 3: return np.nan
    so, sp = o.std(), p.std()
    if so == 0: return np.nan
    r = np.corrcoef(o, p)[0, 1] if sp > 0 else 0.0
    alpha = sp / so
    beta = (p.mean() - o.mean()) / so
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + beta ** 2)

def core_metrics(o, p):
    o, p = _finite(np.asarray(o, float), np.asarray(p, float))
    if len(o) < 3:
        return {k: np.nan for k in
                ["RMSE", "MAE", "R2", "PearsonR", "WillmottD", "KGE", "Bias"]}
    err = p - o
    sst = np.sum((o - o.mean()) ** 2)
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
        "PearsonR": float(np.corrcoef(o, p)[0, 1]) if p.std() > 0 else np.nan,
        "WillmottD": float(willmott_d(o, p)),
        "KGE": float(kge(o, p)),
        "Bias": float(np.mean(err)),
    }

def morans_I(values, A):
    """Global Moran's I of residuals over the county graph."""
    v = np.asarray(values, float)
    m = np.isfinite(v)
    if m.sum() < 10: return np.nan
    W = A.copy()
    np.fill_diagonal(W, 0.0)
    W = W[np.ix_(m, m)]
    v = v[m] - v[m].mean()
    S0 = W.sum()
    if S0 == 0 or (v ** 2).sum() == 0: return np.nan
    num = float(v @ (W @ v))
    return (len(v) / S0) * num / float((v ** 2).sum())
print("metric functions defined")
''')

code(r'''
def evaluate(pred, obs, ci, ti, tag, A_for_moran):
    """Full metric bundle for one model on one split."""
    out = {}
    # overall + per target
    out.update({f"all_{k}": v for k, v in
                core_metrics(obs.ravel(), pred.ravel()).items()})
    for j, tname in enumerate(TARGETS):
        for k, v in core_metrics(obs[:, j], pred[:, j]).items():
            out[f"{tname}_{k}"] = v

    df = pd.DataFrame({"ci": ci, "ti": ti})
    for j, tname in enumerate(TARGETS):
        df[f"obs_{tname}"] = obs[:, j]
        df[f"pred_{tname}"] = pred[:, j]
        df[f"res_{tname}"] = pred[:, j] - obs[:, j]
    df["adcode"] = [counties[c] for c in df.ci]
    df["year"] = year_of[df.ti.values]
    df["month"] = month_of[df.ti.values]

    # residual Moran's I: county means of the primary target residual, per year
    prim = TARGETS[0]
    Is = []
    for y, g in df.groupby("year"):
        vec = np.full(N, np.nan)
        gm = g.groupby("ci")[f"res_{prim}"].mean()
        vec[gm.index.values] = gm.values
        Is.append(morans_I(vec, A_for_moran))
    out["residual_MoranI"] = float(np.nanmean(Is)) if Is else np.nan

    # shock months (disturbance response, N1). Keyed on (ci, ti) so the join
    # cannot silently mismatch on differing time conventions.
    shock = panel[["ci", "ti", "heat_z"]].copy()
    shock["is_shock"] = shock.heat_z.abs() >= SHOCK_THRESHOLD
    df = df.merge(shock[["ci", "ti", "is_shock"]], on=["ci", "ti"], how="left")
    df["is_shock"] = df.is_shock.fillna(False).astype(bool)
    for lbl, sub in (("shock", df[df.is_shock]), ("calm", df[~df.is_shock])):
        mm = core_metrics(sub[f"obs_{prim}"], sub[f"pred_{prim}"])
        out[f"{lbl}_RMSE"] = mm["RMSE"]
        out[f"{lbl}_R2"] = mm["R2"]
        out[f"{lbl}_n"] = int(len(sub))

    # per reach
    reach = panel.groupby("adcode").reach.first()
    df["reach"] = df.adcode.map(reach)
    for r, sub in df.groupby("reach"):
        out[f"reach_{r}_RMSE"] = core_metrics(
            sub[f"obs_{prim}"], sub[f"pred_{prim}"])["RMSE"]

    out["n_samples"] = int(len(df))
    return out, df
print("evaluate() defined")
''')

# ═════════════════════════════════════════════════════════════ PLOTTING
md("## 7 · Plotting utilities  *(font size 20, dpi 300)*")

code(r'''
def _save(fig, folder, name):
    p = Path(folder) / f"{name}.{FIG_FMT}"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p

def plot_history(hist, folder, model):
    h = pd.DataFrame(hist)
    # loss
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(h.epoch, h.train_loss, lw=2.5, label="Train loss")
    ax.plot(h.epoch, h.val_loss, lw=2.5, label="Validation loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)")
    ax.set_title(f"{model} — Training curve"); ax.legend()
    _save(fig, folder, "01_loss_curve")

    for metric, lab in (("rmse", "RMSE"), ("mae", "MAE"), ("r2", "R$^2$")):
        tr, va = f"train_{metric}", f"val_{metric}"
        if tr not in h: continue
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.plot(h.epoch, h[tr], lw=2.5, label=f"Train {lab}")
        ax.plot(h.epoch, h[va], lw=2.5, label=f"Val {lab}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(lab)
        ax.set_title(f"{model} — {lab} per epoch"); ax.legend()
        _save(fig, folder, f"02_{metric}_curve")

    if "lr" in h:
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.plot(h.epoch, h.lr, lw=2.5, color="darkgreen")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate")
        ax.set_yscale("log"); ax.set_title(f"{model} — LR schedule")
        _save(fig, folder, "03_lr_schedule")

def plot_scatter(df, folder, model):
    for tname in TARGETS:
        o, p = df[f"obs_{tname}"].values, df[f"pred_{tname}"].values
        m = np.isfinite(o) & np.isfinite(p)
        o, p = o[m], p[m]
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.hexbin(o, p, gridsize=60, mincnt=1, cmap="viridis")
        lim = [np.percentile(o, 0.5), np.percentile(o, 99.5)]
        ax.plot(lim, lim, "r--", lw=2.5, label="1:1")
        if len(o) > 2:
            b = np.polyfit(o, p, 1)
            ax.plot(lim, np.polyval(b, lim), "orange", lw=2.5, label="Fit")
        mm = core_metrics(o, p)
        ax.set_xlabel(f"Observed {tname}"); ax.set_ylabel(f"Predicted {tname}")
        ax.set_title(f"{model} — {tname}\nR$^2$={mm['R2']:.3f}  "
                     f"RMSE={mm['RMSE']:.3f}")
        ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim)
        _save(fig, folder, f"04_scatter_{tname}")

def plot_residuals(df, folder, model):
    prim = TARGETS[0]
    r = df[f"res_{prim}"].dropna().values
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.hist(r, bins=80, color="steelblue", edgecolor="k", alpha=.85)
    ax.axvline(0, color="r", ls="--", lw=2.5)
    ax.set_xlabel(f"Residual ({prim})"); ax.set_ylabel("Count")
    ax.set_title(f"{model} — Residual distribution\n"
                 f"mean={r.mean():.3f}  sd={r.std():.3f}")
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
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_linewidth(2.5)
    ax.set_title(f"{model} — Residual Q–Q plot")
    _save(fig, folder, "07_residual_qq")

def plot_error_breakdowns(df, folder, model):
    prim = TARGETS[0]
    df = df.copy()
    df["abs_err"] = df[f"res_{prim}"].abs()

    fig, ax = plt.subplots(figsize=(11, 7))
    order = [r for r in ["upstream", "midstream", "downstream"]
             if r in df.reach.unique()]
    # set tick labels manually: boxplot's `labels` kwarg was renamed to
    # `tick_labels` in matplotlib 3.9+, so neither name is portable
    ax.boxplot([df[df.reach == r].abs_err.dropna() for r in order],
               showfliers=False)
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order)
    ax.set_ylabel("Absolute error"); ax.set_xlabel("Reach")
    ax.set_title(f"{model} — Error by reach")
    _save(fig, folder, "08_error_by_reach")

    g = df.groupby("month").abs_err.mean()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.bar(g.index, g.values, color="teal", edgecolor="k")
    ax.set_xlabel("Month"); ax.set_ylabel("Mean absolute error")
    ax.set_title(f"{model} — Error seasonality"); ax.set_xticks(range(1, 13))
    _save(fig, folder, "09_error_by_month")

    g = df.groupby("year").abs_err.mean()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(g.index, g.values, "o-", lw=2.5, ms=9, color="darkred")
    ax.set_xlabel("Year"); ax.set_ylabel("Mean absolute error")
    ax.set_title(f"{model} — Error by test year")
    _save(fig, folder, "10_error_by_year")

    if "is_shock" in df:
        fig, ax = plt.subplots(figsize=(9, 7))
        vals = [df[~df.is_shock].abs_err.dropna(), df[df.is_shock].abs_err.dropna()]
        vals = [v if len(v) else pd.Series([np.nan]) for v in vals]
        ax.boxplot(vals, showfliers=False)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["Calm", "Shock"])
        ax.set_ylabel("Absolute error")
        ax.set_title(f"{model} — Calm vs disturbance months")
        _save(fig, folder, "11_error_shock_vs_calm")

def plot_examples(df, folder, model, k=4):
    prim = TARGETS[0]
    cnt = df.groupby("adcode").size().sort_values(ascending=False)
    pick = list(cnt.index[:k])
    fig, axes = plt.subplots(len(pick), 1, figsize=(14, 4.2 * len(pick)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, pick):
        s = df[df.adcode == a].sort_values("ti")
        ax.plot(s.ti, s[f"obs_{prim}"], "o-", lw=2.2, ms=6, label="Observed")
        ax.plot(s.ti, s[f"pred_{prim}"], "s--", lw=2.2, ms=6, label="Predicted")
        ax.set_ylabel(prim)
        ax.set_title(f"County {a}")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Month index")
    fig.suptitle(f"{model} — Example county trajectories (test period)")
    _save(fig, folder, "12_example_timeseries")

def plot_map(df, folder, model):
    if not HAS_GPD: return
    try:
        gdf = gpd.read_file(BOUNDARY_FILE)
        gdf["adcode"] = gdf.adcode.astype(str)
    except Exception as e:
        print("map skipped:", e); return
    prim = TARGETS[0]
    per = (df.assign(se=lambda d: d[f"res_{prim}"] ** 2)
             .groupby("adcode").se.mean().pow(.5).rename("rmse").reset_index())
    g = gdf.merge(per, on="adcode", how="left")
    fig, ax = plt.subplots(figsize=(15, 11))
    g.plot(column="rmse", ax=ax, legend=True, cmap="YlOrRd",
           edgecolor="grey", linewidth=.15, missing_kwds={"color": "lightgrey"},
           legend_kwds={"label": f"Test RMSE ({prim})", "shrink": .7})
    ax.set_title(f"{model} — Spatial distribution of test error")
    ax.set_axis_off()
    _save(fig, folder, "13_map_rmse")

def plot_metric_bars(metrics, folder, model):
    keys = ["all_RMSE", "all_MAE", "all_R2", "all_PearsonR",
            "all_WillmottD", "all_KGE"]
    vals = [metrics.get(k, np.nan) for k in keys]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar([k.replace("all_", "") for k in keys], vals,
           color="slateblue", edgecolor="k")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=FONT_SIZE - 4)
    ax.set_ylabel("Value"); ax.set_title(f"{model} — Test metrics")
    ax.axhline(0, color="k", lw=1)
    _save(fig, folder, "14_metric_summary")

def make_all_plots(hist, df, metrics, folder, model):
    plot_history(hist, folder, model)
    plot_scatter(df, folder, model)
    plot_residuals(df, folder, model)
    plot_error_breakdowns(df, folder, model)
    plot_examples(df, folder, model)
    plot_map(df, folder, model)
    plot_metric_bars(metrics, folder, model)
    print(f"  plots written to {folder}")
print("plotting utilities defined")
''')

# ═══════════════════════════════════════════════════════════════ MODELS
md(r"""
## 8 · Baseline 1 — DRSEI (Autoencoder + LSTM)

*Gong et al. 2025, Remote Sensing 17(3):558 — "Beyond the Remote Sensing Ecological Index"*

An autoencoder compresses the indicator vector at each step; an LSTM models the
compressed sequence. **Purely temporal and per-county — no spatial structure.**
It therefore isolates the contribution of PERSIST's dual graph (N2) and
lithology-adaptive gating (N3). Trained with a joint reconstruction + forecast
objective, mirroring the original design.
""")

code(r'''
class DRSEI(nn.Module):
    def __init__(self, f_dyn, f_static, latent=DRSEI_LATENT,
                 hidden=DRSEI_HIDDEN, n_out=len(TARGETS), dropout=DROPOUT):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(f_dyn, hidden), nn.ReLU(),
            nn.Linear(hidden, latent))
        self.dec = nn.Sequential(                 # reconstruction head
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, f_dyn))
        self.lstm = nn.LSTM(latent, hidden, num_layers=2, batch_first=True,
                            dropout=dropout)
        self.static = nn.Sequential(
            nn.Linear(f_static, hidden // 2), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden + hidden // 2, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x, s):
        B, L, Fd = x.shape
        z = self.enc(x.reshape(B * L, Fd))
        recon = self.dec(z).reshape(B, L, Fd)
        z = z.reshape(B, L, -1)
        out, _ = self.lstm(z)
        h = torch.cat([out[:, -1], self.static(s)], dim=1)
        return self.head(h), recon

def drsei_step(model, batch, recon_w=0.1):
    x, s, y, _, _ = batch
    x, s, y = x.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
    pred, recon = model(x, s)
    loss = F.mse_loss(pred, y) + recon_w * F.mse_loss(recon, x)
    return pred, y, loss
print("DRSEI defined")
''')

md(r"""
## 9 · Baseline 2 — STGCN (Spatio-Temporal Graph Convolutional Network)

Graph convolution over the county contiguity graph, interleaved with gated
temporal convolutions. It receives the **same information and the same spatial
graph** as PERSIST but optimises a generic regression objective — no
forcing/response separation (N1) and no ecological constraints (N4). Any PERSIST
gain over STGCN is therefore attributable to the dynamical framing rather than
to model capacity.

Deliberately uses the **spatial graph only**: dual-graph fusion is PERSIST's N2.
""")

code(r'''
class TemporalGated(nn.Module):
    """Gated 1-D conv along time (GLU)."""
    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(cin, 2 * cout, (1, k))
    def forward(self, x):                      # (B, C, N, T)
        h = self.conv(x)
        p, q = h.chunk(2, dim=1)
        return p * torch.sigmoid(q)

class GraphConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.lin = nn.Linear(cin, cout)
    def forward(self, x, A):                   # (B, C, N, T)
        h = torch.einsum("bcnt,nm->bcmt", x, A)
        h = h.permute(0, 2, 3, 1)              # (B, N, T, C)
        h = self.lin(h)
        return h.permute(0, 3, 1, 2)           # (B, C, N, T)

class STBlock(nn.Module):
    def __init__(self, cin, cmid, cout, dropout=DROPOUT):
        super().__init__()
        self.t1 = TemporalGated(cin, cmid)
        self.g = GraphConv(cmid, cmid)
        self.t2 = TemporalGated(cmid, cout)
        self.norm = nn.BatchNorm2d(cout)
        self.do = nn.Dropout(dropout)
    def forward(self, x, A):
        h = self.t1(x)
        h = F.relu(self.g(h, A))
        h = self.t2(h)
        return self.do(self.norm(h))

class STGCN(nn.Module):
    def __init__(self, f_dyn, f_static, ch=STGCN_CHANNELS,
                 n_out=len(TARGETS)):
        super().__init__()
        self.b1 = STBlock(f_dyn, ch, ch)
        self.b2 = STBlock(ch, ch, ch)
        self.static = nn.Sequential(nn.Linear(f_static, ch // 2), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(ch + ch // 2, ch), nn.ReLU(),
                                  nn.Linear(ch, n_out))
    def forward(self, x, s, A):
        # x (B, L, N, F) -> (B, F, N, L)
        h = x.permute(0, 3, 2, 1)
        h = self.b1(h, A)
        h = self.b2(h, A)
        h = h[:, :, :, -1].permute(0, 2, 1)     # (B, N, C) last time step
        sc = self.static(s).unsqueeze(0).expand(h.shape[0], -1, -1)
        return self.head(torch.cat([h, sc], dim=2))   # (B, N, n_out)

class GraphDS(torch.utils.data.Dataset):
    """One sample = one time window across ALL counties."""
    def __init__(self, ends):
        self.ends = np.asarray(ends)
    def __len__(self): return len(self.ends)
    def __getitem__(self, i):
        e = int(self.ends[i])
        return (Xt[:, e - LOOKBACK:e].permute(1, 0, 2),   # (L, N, F)
                Yt[:, e],                                  # (N, n_out)
                torch.from_numpy(VALID[:, e].astype(np.float32)),
                e)

def graph_loader(ends, bs, shuffle):
    return torch.utils.data.DataLoader(GraphDS(ends), batch_size=bs,
                                       shuffle=shuffle, num_workers=0)

def stgcn_step(model, batch):
    x, y, mask, _ = batch
    x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
    pred = model(x, St.to(DEVICE), A_t)
    m = mask.unsqueeze(-1)
    loss = ((pred - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum().clamp(min=1) \
           / y.shape[-1]
    return pred, y, mask, loss
print("STGCN defined")
''')

md(r"""
## 10 · Baseline 3 — Temporal Fusion Transformer (TFT)

*Lim et al., attention-based multi-horizon forecasting*

Purpose-built for exactly this data shape: **static covariates** (terrain, karst)
plus **time-varying observed inputs** (state, forcing), with variable-selection
networks, gated residual networks, and interpretable multi-head attention.

This is the **hardest** of the three baselines — it already offers
interpretability, so it competes directly with N6. Including it is deliberate: a
novelty that only wins against weak comparators is not a novelty.
""")

code(r'''
class GLU(nn.Module):
    def __init__(self, d):
        super().__init__(); self.fc = nn.Linear(d, 2 * d)
    def forward(self, x):
        a, b = self.fc(x).chunk(2, -1)
        return a * torch.sigmoid(b)

class GRN(nn.Module):
    """Gated Residual Network with optional static context."""
    def __init__(self, din, dh, dout=None, ctx=None, dropout=DROPOUT):
        super().__init__()
        dout = dout or din
        self.fc1 = nn.Linear(din, dh)
        self.ctx = nn.Linear(ctx, dh, bias=False) if ctx else None
        self.fc2 = nn.Linear(dh, dout)
        self.glu = GLU(dout)
        self.do = nn.Dropout(dropout)
        self.skip = nn.Linear(din, dout) if din != dout else nn.Identity()
        self.norm = nn.LayerNorm(dout)
    def forward(self, x, c=None):
        h = self.fc1(x)
        if self.ctx is not None and c is not None:
            h = h + self.ctx(c)
        h = self.do(self.fc2(F.elu(h)))
        return self.norm(self.glu(h) + self.skip(x))

class VSN(nn.Module):
    """Variable Selection Network over n scalar variables."""
    def __init__(self, n_vars, dh, ctx=None, dropout=DROPOUT):
        super().__init__()
        self.n = n_vars
        self.flat = GRN(n_vars, dh, n_vars, ctx, dropout)
        self.per = nn.ModuleList([GRN(1, dh, dh, None, dropout)
                                  for _ in range(n_vars)])
    def forward(self, x, c=None):               # x (..., n_vars)
        w = torch.softmax(self.flat(x, c), dim=-1).unsqueeze(-1)
        feats = torch.stack([m(x[..., i:i + 1]) for i, m in
                             enumerate(self.per)], dim=-2)
        return (w * feats).sum(-2), w.squeeze(-1)

class InterpretableMHA(nn.Module):
    def __init__(self, d, heads=TFT_HEADS, dropout=DROPOUT):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, self.dk)          # shared value across heads
        self.out = nn.Linear(self.dk, d)
        self.do = nn.Dropout(dropout)
    def forward(self, q, k, v):
        B, L, _ = q.shape
        Q = self.q(q).view(B, L, self.h, self.dk).transpose(1, 2)
        K = self.k(k).view(B, L, self.h, self.dk).transpose(1, 2)
        V = self.v(v).unsqueeze(1)
        att = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(self.dk), -1)
        h = (self.do(att) @ V).mean(1)
        return self.out(h), att.mean(1)

class TFT(nn.Module):
    def __init__(self, f_dyn, f_static, d=TFT_HIDDEN, n_out=len(TARGETS),
                 dropout=DROPOUT):
        super().__init__()
        self.stat_vsn = VSN(f_static, d, None, dropout)
        self.c_sel = GRN(d, d, d, None, dropout)
        self.c_enr = GRN(d, d, d, None, dropout)
        self.c_h = GRN(d, d, d, None, dropout)
        self.c_c = GRN(d, d, d, None, dropout)
        self.dyn_vsn = VSN(f_dyn, d, d, dropout)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.gate1 = GLU(d); self.norm1 = nn.LayerNorm(d)
        self.enrich = GRN(d, d, d, d, dropout)
        self.attn = InterpretableMHA(d, TFT_HEADS, dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ff = GRN(d, d, d, None, dropout)
        self.head = nn.Linear(d, n_out)

    def forward(self, x, s, return_attn=False):
        sv, sw = self.stat_vsn(s)
        c_sel, c_enr = self.c_sel(sv), self.c_enr(sv)
        h0 = self.c_h(sv).unsqueeze(0)
        c0 = self.c_c(sv).unsqueeze(0)
        dv, vw = self.dyn_vsn(x, c_sel.unsqueeze(1).expand(-1, x.shape[1], -1))
        out, _ = self.lstm(dv, (h0.contiguous(), c0.contiguous()))
        h = self.norm1(self.gate1(out) + dv)
        h = self.enrich(h, c_enr.unsqueeze(1).expand(-1, h.shape[1], -1))
        a, att = self.attn(h, h, h)            # a: (B, L, d)
        # single-horizon forecast -> use the final position of both streams
        h = self.norm2(a[:, -1] + h[:, -1])    # (B, d)
        y = self.head(self.ff(h))
        return (y, vw, sw, att) if return_attn else (y, None, None, None)

def tft_step(model, batch):
    x, s, y, _, _ = batch
    x, s, y = x.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
    pred, _, _, _ = model(x, s)
    return pred, y, F.mse_loss(pred, y)
print("TFT defined")
''')

# ══════════════════════════════════════════════════════════════ TRAINING
md("## 11 · Shared training loop — epoch-wise printing **and** `history.csv` logging")

code(r'''
def quick_metrics(o, p):
    o, p = np.asarray(o).ravel(), np.asarray(p).ravel()
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    if len(o) < 3: return np.nan, np.nan, np.nan
    e = p - o
    sst = ((o - o.mean()) ** 2).sum()
    return (float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean()),
            float(1 - (e ** 2).sum() / sst) if sst > 0 else np.nan)

def run_epoch_seq(model, dl, opt, step_fn, train):
    model.train() if train else model.eval()
    tot, n, P, O = 0.0, 0, [], []
    for batch in dl:
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            pred, y, loss = step_fn(model, batch)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        bs = y.shape[0]
        tot += float(loss) * bs; n += bs
        P.append(pred.detach().cpu().numpy()); O.append(y.detach().cpu().numpy())
    return tot / max(n, 1), np.concatenate(P), np.concatenate(O)

def run_epoch_graph(model, dl, opt, train):
    model.train() if train else model.eval()
    tot, n, P, O = 0.0, 0, [], []
    for batch in dl:
        if train: opt.zero_grad()
        with torch.set_grad_enabled(train):
            pred, y, mask, loss = stgcn_step(model, batch)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        tot += float(loss); n += 1
        mk = mask.detach().cpu().numpy().astype(bool)
        P.append(pred.detach().cpu().numpy()[mk])
        O.append(y.detach().cpu().numpy()[mk])
    return tot / max(n, 1), np.concatenate(P), np.concatenate(O)

def train_model(name, model, mode, loaders, epochs):
    folder = OUTPUT_DIR / name
    model = model.to(DEVICE)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=.5, patience=4, min_lr=1e-6)

    print(f"\n{'='*74}\n{name}  |  {nparam:,} parameters  |  device {DEVICE}\n{'='*74}")
    hdr = (f"{'ep':>4} {'tr_loss':>10} {'va_loss':>10} {'tr_RMSE':>9} "
           f"{'va_RMSE':>9} {'tr_MAE':>8} {'va_MAE':>8} {'tr_R2':>8} "
           f"{'va_R2':>8} {'lr':>9} {'sec':>6}")
    print(hdr); print("-" * len(hdr))

    hist, best, bad, best_state = [], np.inf, 0, None
    tr_dl, va_dl, _ = loaders
    for ep in range(1, epochs + 1):
        te = time.time()
        if mode == "graph":
            trl, trP, trO = run_epoch_graph(model, tr_dl, opt, True)
            val, vaP, vaO = run_epoch_graph(model, va_dl, opt, False)
        else:
            step = drsei_step if mode == "drsei" else tft_step
            trl, trP, trO = run_epoch_seq(model, tr_dl, opt, step, True)
            val, vaP, vaO = run_epoch_seq(model, va_dl, opt, step, False)

        trR, trM, trR2 = quick_metrics(trO, trP)
        vaR, vaM, vaR2 = quick_metrics(vaO, vaP)
        lr = opt.param_groups[0]["lr"]
        sched.step(val)
        row = dict(epoch=ep, train_loss=trl, val_loss=val, train_rmse=trR,
                   val_rmse=vaR, train_mae=trM, val_mae=vaM, train_r2=trR2,
                   val_r2=vaR2, lr=lr, seconds=time.time() - te)
        hist.append(row)
        print(f"{ep:>4} {trl:>10.5f} {val:>10.5f} {trR:>9.4f} {vaR:>9.4f} "
              f"{trM:>8.4f} {vaM:>8.4f} {trR2:>8.4f} {vaR2:>8.4f} "
              f"{lr:>9.2e} {row['seconds']:>6.1f}")
        # log every epoch so a crash still leaves a usable history
        pd.DataFrame(hist).to_csv(folder / "history.csv", index=False)

        if vaR < best - 1e-6:
            best, bad = vaR, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping at epoch {ep} (best val RMSE {best:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), folder / "model_best.pt")
    print(f"best val RMSE = {best:.4f} | history.csv + model_best.pt saved")
    return model, hist, nparam
''')

code(r'''
@torch.no_grad()
def predict_seq(model, ci, ti, mode):
    model.eval()
    dl = loader(ci, ti, BATCH_SIZE * 2, False)
    P, O, C, Tt = [], [], [], []
    for x, s, y, c, t in dl:
        x, s = x.to(DEVICE), s.to(DEVICE)
        pred = model(x, s)[0]
        P.append(pred.cpu().numpy()); O.append(y.numpy())
        C.append(c.numpy()); Tt.append(t.numpy())
    return (np.concatenate(P), np.concatenate(O),
            np.concatenate(C), np.concatenate(Tt))

@torch.no_grad()
def predict_graph(model, ends):
    model.eval()
    dl = graph_loader(ends, GRAPH_BATCH, False)
    P, O, C, Tt = [], [], [], []
    for x, y, mask, e in dl:
        pred = model(x.to(DEVICE), St.to(DEVICE), A_t).cpu().numpy()
        y = y.numpy(); mk = mask.numpy().astype(bool)
        for b in range(pred.shape[0]):
            keep = np.where(mk[b])[0]
            P.append(pred[b][keep]); O.append(y[b][keep])
            C.append(keep); Tt.append(np.full(len(keep), int(e[b])))
    return (np.concatenate(P), np.concatenate(O),
            np.concatenate(C), np.concatenate(Tt))

def finalise(name, model, hist, nparam, mode, elapsed):
    folder = OUTPUT_DIR / name
    if mode == "graph":
        pred, obs, ci, ti = predict_graph(model, te_e)
    else:
        pred, obs, ci, ti = predict_seq(model, te_ci, te_ti, mode)
    metrics, df = evaluate(pred, obs, ci, ti, "test", A_sp)
    metrics.update(model=name, n_parameters=int(nparam),
                   epochs_run=len(hist),
                   best_val_rmse=float(min(h["val_rmse"] for h in hist)),
                   train_seconds=float(elapsed))

    with open(folder / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    pd.DataFrame([metrics]).to_csv(folder / "metrics.csv", index=False)
    df.to_csv(folder / "predictions" / "test_predictions.csv", index=False)

    print(f"\n--- {name} TEST METRICS ---")
    for k in ["all_RMSE", "all_MAE", "all_R2", "all_PearsonR", "all_WillmottD",
              "all_KGE", "all_Bias", "residual_MoranI", "shock_RMSE",
              "calm_RMSE"]:
        if k in metrics:
            print(f"  {k:18} {metrics[k]:+.4f}")
    make_all_plots(hist, df, metrics, folder / "plots", name)
    return metrics
print("training / evaluation harness ready")
''')

# ══════════════════════════════════════════════════════════════════ RUN
md("## 12 · Run all three baselines")

code(r'''
EP = 2 if SMOKE_TEST else EPOCHS
F_DYN, F_STAT = X.shape[2], S.shape[1]
print(f"F_dyn={F_DYN}  F_static={F_STAT}  epochs={EP}")
ALL = {}
''')

code(r'''
# ---------------- Baseline 1 : DRSEI ---------------------------------
set_seed(SEED)
dl_tr = loader(tr_ci, tr_ti, BATCH_SIZE, True)
dl_va = loader(va_ci, va_ti, BATCH_SIZE, False)
t0 = time.time()
m1, h1, n1 = train_model("DRSEI_AE_LSTM", DRSEI(F_DYN, F_STAT), "drsei",
                         (dl_tr, dl_va, None), EP)
ALL["DRSEI_AE_LSTM"] = finalise("DRSEI_AE_LSTM", m1, h1, n1, "drsei",
                                time.time() - t0)
''')

code(r'''
# ---------------- Baseline 2 : STGCN ---------------------------------
set_seed(SEED)
gtr = graph_loader(tr_e, GRAPH_BATCH, True)
gva = graph_loader(va_e, GRAPH_BATCH, False)
t0 = time.time()
m2, h2, n2 = train_model("STGCN", STGCN(F_DYN, F_STAT), "graph",
                         (gtr, gva, None), EP)
ALL["STGCN"] = finalise("STGCN", m2, h2, n2, "graph", time.time() - t0)
''')

code(r'''
# ---------------- Baseline 3 : TFT -----------------------------------
set_seed(SEED)
t0 = time.time()
m3, h3, n3 = train_model("TFT", TFT(F_DYN, F_STAT), "tft",
                         (dl_tr, dl_va, None), EP)
ALL["TFT"] = finalise("TFT", m3, h3, n3, "tft", time.time() - t0)
''')

# ═══════════════════════════════════════════════════════════ COMPARISON
md("## 13 · Baseline comparison")

code(r'''
comp = pd.DataFrame(ALL).T.reset_index(drop=True)
front = ["model", "n_parameters", "epochs_run", "train_seconds",
         "best_val_rmse", "all_RMSE", "all_MAE", "all_R2", "all_PearsonR",
         "all_WillmottD", "all_KGE", "all_Bias", "residual_MoranI",
         "shock_RMSE", "calm_RMSE"]
cols = [c for c in front if c in comp.columns] + \
       [c for c in comp.columns if c not in front]
comp = comp[cols]
comp.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)
print("saved ->", OUTPUT_DIR / "baseline_comparison.csv")
display(comp[[c for c in front if c in comp.columns]].round(4))
''')

code(r'''
# ------- comparison plots -------------------------------------------
CMP = OUTPUT_DIR / "comparison_plots"
CMP.mkdir(exist_ok=True, parents=True)
models = comp.model.tolist()
palette = ["#4C72B0", "#DD8452", "#55A868"]

# 1. grouped metric bars
metric_sets = [("all_RMSE", "RMSE (lower better)"),
               ("all_MAE", "MAE (lower better)"),
               ("all_R2", "R$^2$ (higher better)"),
               ("all_PearsonR", "Pearson r (higher better)"),
               ("all_WillmottD", "Willmott d (higher better)"),
               ("all_KGE", "KGE (higher better)"),
               ("residual_MoranI", "Residual Moran's I (closer to 0 better)"),
               ("shock_RMSE", "Shock-month RMSE (lower better)")]
for key, lab in metric_sets:
    if key not in comp: continue
    fig, ax = plt.subplots(figsize=(11, 7))
    vals = comp[key].astype(float).values
    ax.bar(models, vals, color=palette[:len(models)], edgecolor="k")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.4f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=FONT_SIZE - 5)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel(lab); ax.set_title(f"Baseline comparison — {lab}")
    plt.xticks(rotation=15)
    fig.savefig(CMP / f"cmp_{key}.{FIG_FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# 2. combined validation curves
fig, ax = plt.subplots(figsize=(12, 7))
for (nm, h), col in zip([("DRSEI_AE_LSTM", h1), ("STGCN", h2), ("TFT", h3)],
                        palette):
    d = pd.DataFrame(h)
    ax.plot(d.epoch, d.val_rmse, lw=2.5, label=nm, color=col)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation RMSE")
ax.set_title("Validation RMSE — all baselines"); ax.legend()
fig.savefig(CMP / f"cmp_val_curves.{FIG_FMT}", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# 3. per-reach RMSE grouped bars
reach_cols = [c for c in comp.columns if c.startswith("reach_")]
if reach_cols:
    fig, ax = plt.subplots(figsize=(12, 7))
    w = .8 / len(models)
    xs = np.arange(len(reach_cols))
    for i, (mn, col) in enumerate(zip(models, palette)):
        v = comp.loc[comp.model == mn, reach_cols].astype(float).values.ravel()
        ax.bar(xs + i * w, v, w, label=mn, color=col, edgecolor="k")
    ax.set_xticks(xs + w * (len(models) - 1) / 2)
    ax.set_xticklabels([c.replace("reach_", "").replace("_RMSE", "")
                        for c in reach_cols])
    ax.set_ylabel("RMSE"); ax.set_title("Per-reach test RMSE")
    ax.legend()
    fig.savefig(CMP / f"cmp_reach_rmse.{FIG_FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# 4. radar of normalised metrics
radar = [("all_R2", False), ("all_PearsonR", False), ("all_WillmottD", False),
         ("all_KGE", False), ("all_RMSE", True), ("all_MAE", True)]
radar = [(k, inv) for k, inv in radar if k in comp]
if radar:
    lab = [k.replace("all_", "") for k, _ in radar]
    ang = np.linspace(0, 2 * np.pi, len(lab), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    for mn, col in zip(models, palette):
        vs = []
        for k, inv in radar:
            col_v = comp[k].astype(float)
            lo, hi = col_v.min(), col_v.max()
            v = comp.loc[comp.model == mn, k].astype(float).iloc[0]
            s = 0.5 if hi == lo else (v - lo) / (hi - lo)
            vs.append(1 - s if inv else s)
        vs += vs[:1]
        ax.plot(ang, vs, lw=2.5, label=mn, color=col)
        ax.fill(ang, vs, alpha=.12, color=col)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(lab)
    ax.set_title("Normalised metric profile\n(outer = better)", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.savefig(CMP / f"cmp_radar.{FIG_FMT}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

# 5. parameters vs accuracy
fig, ax = plt.subplots(figsize=(11, 7))
for mn, col in zip(models, palette):
    r = comp[comp.model == mn]
    ax.scatter(float(r.n_parameters.iloc[0]), float(r.all_RMSE.iloc[0]),
               s=420, color=col, edgecolor="k", label=mn, zorder=3)
ax.set_xscale("log"); ax.set_xlabel("Parameters")
ax.set_ylabel("Test RMSE"); ax.set_title("Model size vs accuracy")
ax.legend()
fig.savefig(CMP / f"cmp_params_vs_rmse.{FIG_FMT}", dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("comparison plots ->", CMP)
''')

code(r'''
# ------- final summary ------------------------------------------------
print("\n" + "=" * 74)
print("PHASE 4 COMPLETE — baseline results")
print("=" * 74)
key = ["model", "all_RMSE", "all_MAE", "all_R2", "all_KGE",
       "residual_MoranI", "shock_RMSE"]
print(comp[[c for c in key if c in comp]].round(4).to_string(index=False))

best = comp.loc[comp.all_RMSE.astype(float).idxmin(), "model"]
print(f"\nStrongest baseline by test RMSE: {best}")
print("This is the bar the proposed model (PERSIST) must clear.\n")
print("Artefacts written:")
for b in BASELINES:
    f = OUTPUT_DIR / b
    n_pl = len(list((f / 'plots').glob(f'*.{FIG_FMT}')))
    print(f"  {b:16} history.csv, metrics.json/csv, model_best.pt, "
          f"{n_pl} plots")
print(f"  baseline_comparison.csv")
print(f"  comparison_plots/  "
      f"{len(list(CMP.glob(f'*.{FIG_FMT}')))} plots")
''')

md(r"""
---

## Notes

**Metrics** are identical to those planned for PERSIST, so the comparison table
transfers directly. Two are diagnostic rather than merely descriptive:

- **Residual Moran's I** — if PERSIST's dual graph (N2) genuinely captures
  spatial structure, its residual Moran's I should fall closer to zero than the
  baselines'. Harder to game than R².
- **Shock-month RMSE** — resilience is defined by response to real disturbance,
  so error on `|heat_z| ≥ 1.5` months matters more than average error.

**Known caveats carried from Phase 3:**

1. `n_rel` is included as a covariate because LST clear-sky sampling leaks
   ≈ 0.20 correlation into the anomaly (Phase 3e). Without it, part of the
   apparent forcing response is a sampling artefact.
2. The hydrological graph is terrain-derived, not true river topology.
   STGCN here uses the spatial graph only, so this does not affect the baselines.
3. Anomalies are rebuilt on train-only climatology inside this notebook; the
   shipped `*_z` columns use full-period statistics and would leak.
""")

# ═════════════════════════════════════════════════════════════════ BUILD
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md"
            else nbf.v4.new_code_cell(s) for k, s in C]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
}
out = pathlib.Path("PERSIST_Phase4_Baselines.ipynb")
nbf.write(nb, str(out))
print(f"wrote {out}  ({len(nb.cells)} cells, "
      f"{out.stat().st_size/1024:.0f} KB)")
