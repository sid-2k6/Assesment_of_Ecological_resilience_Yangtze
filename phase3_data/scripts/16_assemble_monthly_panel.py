#!/usr/bin/env python3
"""Assemble the PRIMARY PERSIST modelling panel: monthly county-level state +
forcing + static context + human stream, 2000-2020.

This supersedes the seasonal panel as primary input following the N1 revision
(response channel = LST, monthly resolution, event-conditioned estimation).

Output: data/interim/panel_monthly.parquet
"""
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

OUT = pathlib.Path("data/interim")
MON = OUT / "monthly"


def load(prefix):
    fs = sorted(MON.glob(f"{prefix}_*.csv"))
    return pd.concat([pd.read_csv(f, dtype={"adcode": str}) for f in fs],
                     ignore_index=True)


def main():
    g = gpd.read_file(OUT / "yreb_counties_datav.gpkg").reset_index(drop=True)
    g["adcode"] = g.adcode.astype(str)
    attrs = g[["adcode", "name_zh", "province", "reach", "area_km2"]]

    # ---- state ------------------------------------------------------
    st = load("lst").merge(load("ndvi"), on=["adcode", "year", "month"],
                           how="outer")
    print(f"state   {len(st):,}")

    # ---- forcing ----------------------------------------------------
    fc = pd.read_csv(OUT / "forcing_monthly_2000_2020.csv",
                     dtype={"adcode": str})
    print(f"forcing {len(fc):,}")

    p = st.merge(fc, on=["adcode", "year", "month"], how="inner")

    # ---- static context --------------------------------------------
    terr = pd.read_csv(OUT / "terrain_county.csv", dtype={"adcode": str})
    karst = pd.read_csv(OUT / "karst_county.csv",
                        dtype={"adcode": str})[["adcode", "karst_frac"]]
    p = p.merge(attrs, on="adcode", how="left") \
         .merge(terr, on="adcode", how="left") \
         .merge(karst, on="adcode", how="left")

    # ---- human stream (annual -> broadcast to months) ---------------
    ntl = pd.read_csv(OUT / "ntl_county_year.csv", dtype={"adcode": str})
    p = p.merge(ntl[["adcode", "year", "ntl_mean", "ntl_sum"]],
                on=["adcode", "year"], how="left")

    # ---- derived state ---------------------------------------------
    p["kndvi"] = np.tanh(p["ndvi_mean"] ** 2)
    p["lst_c"] = p["lst_day_mean"] - 273.15
    p["t"] = (p.year - p.year.min()) * 12 + p.month     # global month index

    # anomalies vs county x calendar-month climatology
    for c in ("kndvi", "lst_c", "ndvi_mean"):
        grp = p.groupby(["adcode", "month"])[c]
        p[f"{c}_z"] = (p[c] - grp.transform("mean")) / \
            grp.transform("std").replace(0, np.nan)

    p = p.sort_values(["adcode", "t"]).reset_index(drop=True)

    # float32 downcast keeps the file small without losing needed precision
    for c in p.select_dtypes("float64").columns:
        p[c] = p[c].astype("float32")
    p.to_parquet(OUT / "panel_monthly.parquet", compression="zstd", index=False)

    print(f"\n{'='*66}\nMONTHLY PANEL: {len(p):,} rows x {len(p.columns)} cols")
    print(f"counties {p.adcode.nunique()} | months {p.t.nunique()} | "
          f"expected {p.adcode.nunique()*252:,}")

    print("\nmissingness (%):")
    for c in ("lst_c", "kndvi", "tmax", "ppt", "pdsi", "elev_mean",
              "karst_frac", "ntl_mean"):
        if c in p:
            print(f"  {c:12} {100*p[c].isna().mean():6.2f}")

    print("\nkarst fraction by reach (N3 gating input):")
    print(p.groupby("reach").karst_frac.mean().round(3).to_string())
    print("\nNTL by reach (Tier-1 human stream):")
    print(p.groupby("reach").ntl_mean.mean().round(2).to_string())

    print(f"\nwrote -> {OUT}/panel_monthly.parquet")


if __name__ == "__main__":
    main()
