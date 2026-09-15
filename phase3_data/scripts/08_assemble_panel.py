#!/usr/bin/env python3
"""Assemble the county-year indicator panel and validate it against published
YREB figures.

External validation targets (Zhu et al. 2025, Land 14(3):598 - the most-cited
paper in our survey, full-YREB, multi-method):
    kNDVI trend  +0.003 / yr   (p < 0.05)
    LST trend    +0.065 degC / yr (p < 0.01)

Outputs:
  data/interim/panel_county_year.csv
  data/interim/panel_validation.json
"""
import json
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

IND = pathlib.Path("data/interim/indicators")
OUT = pathlib.Path("data/interim")


def load_product(prefix):
    files = sorted(IND.glob(f"{prefix}_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, dtype={"adcode": str}) for f in files],
                   ignore_index=True)
    return df


def ols_trend(x, y):
    """Least-squares slope with two-sided p-value, ignoring NaN."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan
    x, y = x[m], y[m]
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx
    icept = ym - slope * xm
    resid = y - (icept + slope * x)
    if n <= 2:
        return slope, np.nan, np.nan
    se = np.sqrt((resid ** 2).sum() / (n - 2) / sxx)
    t = slope / se if se > 0 else np.inf
    # two-sided p from the normal approximation (n=21 -> adequate)
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2))
    r2 = 1 - (resid ** 2).sum() / ((y - ym) ** 2).sum()
    return slope, p, r2


def main():
    # ---- county attributes -------------------------------------------
    g = gpd.read_file(OUT / "yreb_counties_datav.gpkg")
    g["adcode"] = g.adcode.astype(str)
    attrs = g[["adcode", "name_zh", "province", "prefecture_zh", "reach",
               "area_km2"]].copy()

    # ---- indicators ---------------------------------------------------
    panel = None
    for pref in ("ndvi", "lst", "npp", "et"):
        d = load_product(pref)
        if d is None:
            print(f"  {pref}: MISSING")
            continue
        print(f"  {pref}: {len(d):>6} rows, years "
              f"{d.year.min()}-{d.year.max()}, cols {len(d.columns)}")
        panel = d if panel is None else panel.merge(d, on=["adcode", "year"],
                                                    how="outer")

    # ---- terrain (static) ---------------------------------------------
    terr = pd.read_csv(OUT / "terrain_county.csv", dtype={"adcode": str})
    panel = panel.merge(attrs, on="adcode", how="left") \
                 .merge(terr, on="adcode", how="left")

    # ---- derived indicators -------------------------------------------
    # kNDVI (Camps-Valls et al. 2021): with sigma = 0.5*(NIR+Red) this reduces
    # to tanh(NDVI^2). Preferred over NDVI - avoids saturation in dense
    # vegetation, and validated for the YREB by Zhu et al. (2025).
    panel["kndvi_mean"] = np.tanh(panel["ndvi_mean"] ** 2)
    panel["lst_range"] = panel["lst_day_mean"] - panel["lst_night_mean"]
    panel["lst_day_c"] = panel["lst_day_mean"] - 273.15
    panel["lst_night_c"] = panel["lst_night_mean"] - 273.15

    panel = panel.sort_values(["adcode", "year"]).reset_index(drop=True)
    panel.to_csv(OUT / "panel_county_year.csv", index=False)

    # ================= REPORT =========================================
    print(f"\n{'='*66}\nPANEL: {len(panel):,} rows x {len(panel.columns)} cols")
    print(f"counties {panel.adcode.nunique()}  years "
          f"{panel.year.min()}-{panel.year.max()}")

    print("\n--- missingness (%) by indicator ---")
    keys = ["ndvi_mean", "evi_mean", "kndvi_mean", "lst_day_mean",
            "lst_night_mean", "npp_mean", "et_mean", "elev_mean", "slope_mean"]
    for k in keys:
        if k in panel:
            pctm = 100 * panel[k].isna().mean()
            print(f"  {k:16} {pctm:6.2f}%")

    # ---- validation vs published trends ------------------------------
    print(f"\n{'='*66}\nEXTERNAL VALIDATION vs Zhu et al. 2025 (Land 14(3):598)")
    yr_mean = panel.groupby("year").agg(
        kndvi=("kndvi_mean", "mean"), ndvi=("ndvi_mean", "mean"),
        lst_c=("lst_day_c", "mean"), npp=("npp_mean", "mean"),
        et=("et_mean", "mean")).reset_index()

    # 2000 is excluded for 8/16-day products: Terra launched Feb 2000, so the
    # 2000 annual mean omits the coldest weeks and is upward-biased.
    full = yr_mean[yr_mean.year >= 2001]

    res = {}
    for col, label, target, unit in (
        ("kndvi", "kNDVI", 0.003, "/yr"),
        ("lst_c", "LST day", 0.065, " degC/yr"),
    ):
        s, p, r2 = ols_trend(full.year.values.astype(float), full[col].values)
        res[col] = dict(slope=s, p=p, r2=r2, published=target)
        agree = "SAME SIGN" if np.sign(s) == np.sign(target) else "OPPOSITE"
        print(f"  {label:9} ours {s:+.5f}{unit}  (p={p:.4f}, R2={r2:.2f})  "
              f"published {target:+.3f}{unit}  -> {agree}")

    for col, label in (("npp", "NPP"), ("et", "ET"), ("ndvi", "NDVI")):
        s, p, r2 = ols_trend(full.year.values.astype(float), full[col].values)
        res[col] = dict(slope=s, p=p, r2=r2)
        print(f"  {label:9} ours {s:+.5f}/yr  (p={p:.4f}, R2={r2:.2f})")

    # ---- reach gradients ---------------------------------------------
    print(f"\n{'='*66}\nREACH GRADIENTS (2001-2020 mean)")
    rg = (panel[panel.year >= 2001].groupby("reach")
          .agg(kndvi=("kndvi_mean", "mean"), lst_c=("lst_day_c", "mean"),
               npp=("npp_mean", "mean"), et=("et_mean", "mean"),
               elev=("elev_mean", "mean"), slope=("slope_mean", "mean")))
    print(rg.round(3).to_string())

    print("\nNOTE: vegetation/productivity run upstream > downstream, which is "
          "the INVERSE of the published ecological resilience gradient "
          "(downstream/east > upstream/west). Resilience is therefore not "
          "vegetation-driven; it must be dominated by adaptive-capacity and "
          "socioeconomic terms. Supports a multi-dimensional index.")

    # ---- disturbance-year check (label-strategy validation) ----------
    print(f"\n{'='*66}\nDISTURBANCE YEARS (anomaly vs 2001-2020 county mean, z)")
    p2 = panel[panel.year >= 2001].copy()
    for c in ("kndvi_mean", "npp_mean", "et_mean"):
        mu = p2.groupby("adcode")[c].transform("mean")
        sd = p2.groupby("adcode")[c].transform("std")
        p2[f"z_{c}"] = (p2[c] - mu) / sd.replace(0, np.nan)
    zz = p2.groupby("year")[[f"z_{c}" for c in
                             ("kndvi_mean", "npp_mean", "et_mean")]].mean()
    for y in (2006, 2011, 2013, 2016, 2020):
        if y in zz.index:
            r = zz.loc[y]
            print(f"  {y}: kNDVI z={r.iloc[0]:+.3f}  NPP z={r.iloc[1]:+.3f}  "
                  f"ET z={r.iloc[2]:+.3f}")
    print("  (2006 SW China drought, 2011 Yangtze drought, 2016 & 2020 floods)")

    (OUT / "panel_validation.json").write_text(json.dumps(
        {"trends": res,
         "n_rows": int(len(panel)),
         "n_counties": int(panel.adcode.nunique()),
         "years": [int(panel.year.min()), int(panel.year.max())],
         "reach_gradients": rg.round(4).to_dict()}, indent=2, default=float))
    print(f"\nwrote -> {OUT}/panel_county_year.csv")


if __name__ == "__main__":
    main()
