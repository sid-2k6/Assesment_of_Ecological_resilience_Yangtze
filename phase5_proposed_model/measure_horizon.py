#!/usr/bin/env python3
"""How predictable is the target as a function of the aggregation horizon H?

Target for horizon H at end-index e:  mean( y[e : e+H] )
i.e. the mean deseasonalised anomaly over the next H months. H=1 is the
current Phase 4 / Phase 5 task.

Reports, on the TEST period only (2018-2020), with a strict embargo so no
target window ever crosses a split boundary:
  - SeasonalNaive       mean of the same H months one year earlier
  - TrailingPersistence mean of the last H observed months
  - CountyClimatology   train-period county mean
  - Ridge(AR+climate)   closed-form ridge on y-lags 1..12 + climate + calendar
                        -> a cheap proxy for the skill ceiling a deep model
                           can plausibly reach
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
print(f"N={N} T={T}")

train_mask = panel.year.between(*TRAIN_YEARS)
for prim in TARGETS:
    raw = RAW_OF[prim]
    mmu = panel.loc[train_mask].groupby("month")[raw].mean()
    ds = panel[raw] - panel.month.map(mmu)
    gsd = float(ds[train_mask].std())
    panel[prim] = (ds / gsd).clip(-CLIP_SIGMA, CLIP_SIGMA)

year_of = panel.groupby("ti").year.first().reindex(range(T)).values
month_of = panel.groupby("ti").month.first().reindex(range(T)).values

# (N, T) arrays
Y = np.full((N, T, len(TARGETS)), np.nan, np.float32)
Y[panel.ci.values, panel.ti.values, :] = panel[TARGETS].values
CLIM_F = [c for c in ["ppt", "pet", "def", "tmax", "tmin", "vpd", "soil",
                      "srad", "pdsi", "heat_z", "dry_z"] if c in panel]
Xc = np.full((N, T, len(CLIM_F)), np.nan, np.float32)
Xc[panel.ci.values, panel.ti.values, :] = panel[CLIM_F].values
# forward-fill climate gaps
for f in range(Xc.shape[2]):
    col = Xc[:, :, f]
    idx = np.where(~np.isnan(col), np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    Xc[:, :, f] = np.nan_to_num(col[np.arange(N)[:, None], idx], nan=0.0)


def r2(o, p):
    m = np.isfinite(o) & np.isfinite(p)
    o, p = o[m], p[m]
    sst = ((o - o.mean()) ** 2).sum()
    return 1 - ((p - o) ** 2).sum() / sst, np.sqrt(((p - o) ** 2).mean()), len(o)


def fwd_mean(H):
    """Yh[:, e, :] = mean(Y[:, e:e+H, :]); NaN if any month missing."""
    out = np.full_like(Y, np.nan)
    for e in range(T - H + 1):
        out[:, e] = Y[:, e:e + H].mean(axis=1)
    return out


def ends_for(years, H, L=12):
    """End indices e such that window [e-L, e) and target [e, e+H) both sit
    fully inside `years`. This is the embargo that stops a target window from
    leaking across a split boundary."""
    ok = np.where((year_of >= years[0]) & (year_of <= years[1]))[0]
    lo, hi = ok.min(), ok.max()
    return np.array([e for e in range(lo, hi + 1)
                     if e - L >= 0 and e + H - 1 <= hi], dtype=int)


print(f"\nclimate features used by ridge proxy: {len(CLIM_F)}")
print("\n" + "=" * 96)
print(f"{'H':>3} {'lag1_ac':>8} {'n_test':>8} | "
      f"{'SeasNaive':>10} {'TrailPers':>10} {'CountyClim':>10} {'Ridge':>10} | "
      f"{'Ridge RMSE':>11} {'RidgeWithin':>12}")
print("=" * 96)

rows = []
for H in [1, 2, 3, 4, 6, 12]:
    Yh = fwd_mean(H)
    tr_e, va_e, te_e = ends_for(TRAIN_YEARS, H), ends_for(VAL_YEARS, H), ends_for(TEST_YEARS, H)

    # lag-1 autocorrelation of the aggregated primary target (non-overlapping
    # stride H, so the number is not inflated by window overlap)
    a = Yh[:, ::H, 0]
    aa = a[:, :-1].ravel(); bb = a[:, 1:].ravel()
    m = np.isfinite(aa) & np.isfinite(bb)
    ac = np.corrcoef(aa[m], bb[m])[0, 1]

    o = Yh[:, te_e, 0].ravel()
    sn = Yh[:, te_e - 12, 0].ravel()                     # same H months, -1yr
    tp = Yh[:, te_e - H, 0].ravel()                      # trailing H-mean
    cc = np.repeat(np.nanmean(Yh[:, tr_e, 0], axis=1)[:, None],
                   len(te_e), axis=1).ravel()

    # ---- ridge proxy: y-lags 1..12 (both targets) + climate at e-1 + calendar
    def design(ends):
        F = []
        for lag in range(1, 13):
            F.append(Y[:, ends - lag, 0]); F.append(Y[:, ends - lag, 1])
        for f in range(Xc.shape[2]):
            F.append(Xc[:, ends - 1, f])
        mo = month_of[ends]
        F.append(np.tile(np.sin(2*np.pi*mo/12), (N, 1)))
        F.append(np.tile(np.cos(2*np.pi*mo/12), (N, 1)))
        F.append(np.tile(np.ones(len(ends)), (N, 1)))
        return np.stack([f.ravel() for f in F], axis=1).astype(np.float64)

    Xtr, ytr = design(tr_e), Yh[:, tr_e, 0].ravel()
    Xte, yte = design(te_e), o
    ktr = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
    Xtr, ytr = np.nan_to_num(Xtr[ktr]), ytr[ktr]
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd
    lam = 1e2
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)
    Xte_s = (np.nan_to_num(Xte) - mu) / sd
    pr = Xte_s @ w

    # within-county R2 for the ridge (temporal skill, county means removed)
    ci_te = np.repeat(np.arange(N), len(te_e))
    dfw = pd.DataFrame({"ci": ci_te, "o": yte, "p": pr}).dropna()
    dfw["od"] = dfw.o - dfw.groupby("ci").o.transform("mean")
    dfw["pd"] = dfw.p - dfw.groupby("ci").p.transform("mean")
    wr2 = r2(dfw.od.values, dfw["pd"].values)[0]

    res = {k: r2(o, v)[0] for k, v in
           [("SeasNaive", sn), ("TrailPers", tp), ("CountyClim", cc), ("Ridge", pr)]}
    rr, rmse, n = r2(o, pr)
    print(f"{H:>3} {ac:>8.3f} {n:>8,} | {res['SeasNaive']:>10.4f} "
          f"{res['TrailPers']:>10.4f} {res['CountyClim']:>10.4f} {res['Ridge']:>10.4f} | "
          f"{rmse:>11.4f} {wr2:>12.4f}")
    rows.append(dict(H=H, lag1_ac=ac, n_test=n, **res, ridge_rmse=rmse,
                     ridge_within_r2=wr2, n_train_windows=len(tr_e),
                     n_val_windows=len(va_e), n_test_windows=len(te_e)))

print("=" * 96)
d = pd.DataFrame(rows)
d.to_csv("/projects/sandbox/yreb_resilience/horizon_scan.csv", index=False)
print("\nwindow counts (graph snapshots) per H:")
print(d[["H", "n_train_windows", "n_val_windows", "n_test_windows"]].to_string(index=False))
print("\nsaved horizon_scan.csv")
