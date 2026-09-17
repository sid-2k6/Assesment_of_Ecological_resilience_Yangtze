#!/usr/bin/env python3
"""Within-county (temporal) predictability ceiling as a function of horizon H.

The overall R2 in horizon_scan.csv is inflated by the between-county component:
counties differ in mean anomaly and that part is trivially predictable. This
script strips it out (county-demeaning using TRAIN means only) and asks how
much of the remaining purely-temporal variance a linear AR+climate model can
recover. That is the honest ceiling estimate for within_county_R2.

Also reports the between/within variance split of the aggregated target, and a
non-overlapping-stride sensitivity check.
"""
import numpy as np
import pandas as pd

PANEL = ("/projects/sandbox/Assesment_of_Ecological_resilience_Yangtze/"
         "phase3_data/tables/panel_monthly.parquet")
TRAIN_YEARS, VAL_YEARS, TEST_YEARS = (2000, 2014), (2015, 2017), (2018, 2020)
CLIP_SIGMA = 5.0
TARGETS = ["lst_ds", "kndvi_ds"]
RAW_OF = {"lst_ds": "lst_c", "kndvi_ds": "kndvi"}

panel = pd.read_parquet(PANEL)
panel["adcode"] = panel["adcode"].astype(str)
panel["t"] = (panel.year - panel.year.min()) * 12 + (panel.month - 1)
panel = panel.sort_values(["adcode", "t"]).reset_index(drop=True)
counties = sorted(panel.adcode.unique())
cidx = {c: i for i, c in enumerate(counties)}
panel["ci"] = panel.adcode.map(cidx)
tvals = np.sort(panel.t.unique()); tpos = {v: i for i, v in enumerate(tvals)}
panel["ti"] = panel.t.map(tpos)
N, T = len(counties), len(tvals)

train_mask = panel.year.between(*TRAIN_YEARS)
for prim in TARGETS:
    raw = RAW_OF[prim]
    mmu = panel.loc[train_mask].groupby("month")[raw].mean()
    ds = panel[raw] - panel.month.map(mmu)
    gsd = float(ds[train_mask].std())
    panel[prim] = (ds / gsd).clip(-CLIP_SIGMA, CLIP_SIGMA)

year_of = panel.groupby("ti").year.first().reindex(range(T)).values
month_of = panel.groupby("ti").month.first().reindex(range(T)).values
Y = np.full((N, T, 2), np.nan, np.float32)
Y[panel.ci.values, panel.ti.values, :] = panel[TARGETS].values
CLIM_F = [c for c in ["ppt", "pet", "def", "tmax", "tmin", "vpd", "soil",
                      "srad", "pdsi", "heat_z", "dry_z"] if c in panel]
Xc = np.full((N, T, len(CLIM_F)), np.nan, np.float32)
Xc[panel.ci.values, panel.ti.values, :] = panel[CLIM_F].values
for f in range(Xc.shape[2]):
    col = Xc[:, :, f]
    idx = np.where(~np.isnan(col), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    Xc[:, :, f] = np.nan_to_num(col[np.arange(N)[:, None], idx], nan=0.0)


def r2(o, p):
    m = np.isfinite(o) & np.isfinite(p); o, p = o[m], p[m]
    sst = ((o - o.mean()) ** 2).sum()
    return float(1 - ((p - o) ** 2).sum() / sst)


def fwd_mean(H):
    out = np.full_like(Y, np.nan)
    for e in range(T - H + 1):
        out[:, e] = Y[:, e:e + H].mean(axis=1)
    return out


def ends_for(years, H, L=12, stride=1):
    ok = np.where((year_of >= years[0]) & (year_of <= years[1]))[0]
    lo, hi = ok.min(), ok.max()
    e = np.array([x for x in range(lo, hi + 1)
                  if x - L >= 0 and x + H - 1 <= hi], dtype=int)
    return e[::stride]


def ridge_fit(Xtr, ytr, lam=1e2):
    k = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    Xtr, ytr = np.nan_to_num(Xtr[k]), ytr[k]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Z = (Xtr - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ ytr)
    return w, mu, sd


def design(Yh, ends, H):
    F = []
    for lag in range(1, 13):
        F.append(Y[:, ends - lag, 0]); F.append(Y[:, ends - lag, 1])
    F.append(Yh[:, ends - 12, 0])                 # seasonal-naive term
    F.append(Yh[:, ends - H, 0])                  # trailing-H term
    for f in range(Xc.shape[2]):
        F.append(Xc[:, ends - 1, f])
    mo = month_of[ends]
    F.append(np.tile(np.sin(2*np.pi*mo/12), (N, 1)))
    F.append(np.tile(np.cos(2*np.pi*mo/12), (N, 1)))
    F.append(np.tile(np.ones(len(ends)), (N, 1)))
    return np.stack([f.ravel() for f in F], axis=1).astype(np.float64)


print("=" * 100)
print(f"{'H':>3} | {'var_between':>11} {'var_within':>10} | "
      f"{'pooled R2':>9} {'within R2':>9} {'within-fit R2':>13} | "
      f"{'SN within':>9} | {'stride-H R2':>11}")
print("=" * 100)
rows = []
for H in [1, 2, 3, 4, 6]:
    Yh = fwd_mean(H)
    tr_e, te_e = ends_for(TRAIN_YEARS, H), ends_for(TEST_YEARS, H)

    # ---- variance decomposition of the aggregated target (test period)
    A = Yh[:, te_e, 0]
    cm = np.nanmean(A, axis=1, keepdims=True)
    vb = float(np.nanvar(np.repeat(cm, A.shape[1], 1)))
    vw = float(np.nanvar(A - cm))
    fb = vb / (vb + vw)

    Xtr, ytr = design(Yh, tr_e, H), Yh[:, tr_e, 0].ravel()
    Xte, yte = design(Yh, te_e, H), Yh[:, te_e, 0].ravel()
    w, mu, sd = ridge_fit(Xtr, ytr)
    pr = ((np.nan_to_num(Xte) - mu) / sd) @ w
    pooled = r2(yte, pr)

    # ---- within-county R2 of that pooled model
    ci_te = np.repeat(np.arange(N), len(te_e))
    d = pd.DataFrame({"ci": ci_te, "o": yte, "p": pr}).dropna()
    od = d.o - d.groupby("ci").o.transform("mean")
    pd_ = d.p - d.groupby("ci").p.transform("mean")
    within = r2(od.values, pd_.values)

    # ---- model fit DIRECTLY on the within-transformed problem (county means
    #      removed from features and target using TRAIN means only)
    ci_tr = np.repeat(np.arange(N), len(tr_e))
    Xtr_n = np.nan_to_num(Xtr); Xte_n = np.nan_to_num(Xte)
    fmu = np.zeros((N, Xtr.shape[1]))
    for j in range(Xtr.shape[1]):
        fmu[:, j] = pd.Series(Xtr_n[:, j]).groupby(ci_tr).mean().reindex(range(N)).values
    ymu = pd.Series(ytr).groupby(ci_tr).mean().reindex(range(N)).values
    Xtr_w = Xtr_n - fmu[ci_tr]; ytr_w = ytr - np.nan_to_num(ymu)[ci_tr]
    Xte_w = Xte_n - fmu[ci_te]; yte_w = yte - np.nan_to_num(ymu)[ci_te]
    w2, mu2, sd2 = ridge_fit(Xtr_w, ytr_w, lam=1e2)
    pr_w = ((Xte_w - mu2) / sd2) @ w2
    within_fit = r2(yte_w, pr_w)

    # ---- SeasonalNaive within-county skill (the bar the model must clear)
    sn = Yh[:, te_e - 12, 0].ravel()
    ds = pd.DataFrame({"ci": ci_te, "o": yte, "p": sn}).dropna()
    sn_w = r2((ds.o - ds.groupby("ci").o.transform("mean")).values,
              (ds.p - ds.groupby("ci").p.transform("mean")).values)

    # ---- non-overlapping stride sensitivity check
    te_s = ends_for(TEST_YEARS, H, stride=H)
    Xs = design(Yh, te_s, H); ys = Yh[:, te_s, 0].ravel()
    pr_s = ((np.nan_to_num(Xs) - mu) / sd) @ w
    stride_r2 = r2(ys, pr_s)

    print(f"{H:>3} | {fb:>11.3f} {1-fb:>10.3f} | {pooled:>9.4f} {within:>9.4f} "
          f"{within_fit:>13.4f} | {sn_w:>9.4f} | {stride_r2:>11.4f}")
    rows.append(dict(H=H, frac_between=fb, pooled_r2=pooled, within_r2=within,
                     within_fit_r2=within_fit, seasnaive_within_r2=sn_w,
                     stride_r2=stride_r2))
print("=" * 100)
pd.DataFrame(rows).to_csv("/projects/sandbox/yreb_resilience/within_scan.csv", index=False)
print("saved within_scan.csv")
