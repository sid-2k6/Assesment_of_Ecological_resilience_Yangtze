#!/usr/bin/env python3
"""Assemble the seasonal PERSIST panel: ecological STATE stream + climate
FORCING stream, county x year x season, 2000-2020.

Then run the decisive feasibility test for N1: does the forcing stream actually
explain state anomalies? If forcing carries no signal about state deviations,
the Disturbance-Conditioned Response Decoder cannot be identified and N1 must be
redesigned. This is the cheapest possible check before implementation.
"""
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd

SEA = pathlib.Path("data/interim/seasonal")
OUT = pathlib.Path("data/interim")
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def load(prefix):
    fs = sorted(SEA.glob(f"{prefix}_*.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f, dtype={"adcode": str}) for f in fs],
                     ignore_index=True)


def main():
    g = gpd.read_file(OUT / "yreb_counties_datav.gpkg")
    g["adcode"] = g.adcode.astype(str)
    attrs = g[["adcode", "name_zh", "province", "reach", "area_km2"]]

    state = None
    for p in ("ndvi", "lst"):
        d = load(p)
        print(f"  {p}: {len(d):,} rows, {d.year.min()}-{d.year.max()}")
        state = d if state is None else state.merge(
            d, on=["adcode", "year", "season"], how="outer")

    forcing = pd.read_csv(OUT / "climate_forcing_seasonal.csv",
                          dtype={"adcode": str})
    print(f"  forcing: {len(forcing):,} rows")

    panel = state.merge(forcing, on=["adcode", "year", "season"], how="inner")
    panel = panel.merge(attrs, on="adcode", how="left")

    # derived state variables
    panel["kndvi_mean"] = np.tanh(panel["ndvi_mean"] ** 2)
    panel["lst_day_c"] = panel["lst_day_mean"] - 273.15
    panel["lst_night_c"] = panel["lst_night_mean"] - 273.15
    panel["lst_range"] = panel["lst_day_mean"] - panel["lst_night_mean"]

    # state anomalies vs each county x season climatology (the RESPONSE variable)
    for c in ("kndvi_mean", "ndvi_mean", "evi_mean", "lst_day_c"):
        grp = panel.groupby(["adcode", "season"])[c]
        panel[f"{c}_z"] = ((panel[c] - grp.transform("mean"))
                           / grp.transform("std").replace(0, np.nan))

    # season index for temporal ordering
    panel["season_idx"] = panel.season.map({s: i for i, s in enumerate(SEASONS)})
    panel = panel.sort_values(["adcode", "year", "season_idx"]).reset_index(drop=True)
    panel.to_csv(OUT / "panel_seasonal.csv", index=False)

    print(f"\n{'='*70}\nSEASONAL PANEL: {len(panel):,} rows x {len(panel.columns)} cols")
    print(f"counties {panel.adcode.nunique()}  years "
          f"{panel.year.min()}-{panel.year.max()}  seasons {panel.season.nunique()}")
    print(f"expected: {panel.adcode.nunique()*21*4:,}")

    # ---- flag the partial first winter -------------------------------
    part = panel[(panel.year == 2000) & (panel.season == "DJF")]
    print(f"\nDJF 2000 mean ndvi_n = {part.ndvi_n.mean():.0f} vs "
          f"{panel[(panel.year==2001)&(panel.season=='DJF')].ndvi_n.mean():.0f} "
          f"in 2001 -> partial (Terra launched Feb 2000). Filter via *_n.")

    # ================= N1 FEASIBILITY TEST ============================
    print(f"\n{'='*70}\nN1 FEASIBILITY: does FORCING explain STATE anomalies?")
    print("If these correlations are ~0, the response decoder is unidentifiable.\n")

    d = panel[panel.year >= 2001].copy()
    pairs = [("dry_z", "kndvi_mean_z"), ("heat_z", "kndvi_mean_z"),
             ("dry_z", "lst_day_c_z"), ("heat_z", "lst_day_c_z"),
             ("pdsi", "kndvi_mean_z"), ("soil_z", "kndvi_mean_z")]
    print(f"  {'forcing':10} {'->':2} {'state':16} {'r (JJA)':>9} {'r (all)':>9}")
    for f, s in pairs:
        sub = d[[f, s]].dropna()
        r_all = sub[f].corr(sub[s]) if len(sub) > 10 else np.nan
        jj = d[d.season == "JJA"][[f, s]].dropna()
        r_jja = jj[f].corr(jj[s]) if len(jj) > 10 else np.nan
        print(f"  {f:10} -> {s:16} {r_jja:+9.3f} {r_all:+9.3f}")

    # within-county partial signal (removes cross-sectional confounding)
    print("\n  within-county correlation (county-demeaned, JJA only):")
    jj = d[d.season == "JJA"].copy()
    for f, s in [("dry_z", "kndvi_mean_z"), ("heat_z", "lst_day_c_z"),
                 ("heat_z", "kndvi_mean_z")]:
        t = jj[["adcode", f, s]].dropna()
        t[f + "_d"] = t[f] - t.groupby("adcode")[f].transform("mean")
        t[s + "_d"] = t[s] - t.groupby("adcode")[s].transform("mean")
        print(f"    {f:8} -> {s:16} r = {t[f+'_d'].corr(t[s+'_d']):+.3f}")

    # ---- recovery signal: does a shock in JJA persist into SON? ------
    print(f"\n{'='*70}\nRECOVERY SIGNAL (needed for N1's recovery term)")
    w = panel.pivot_table(index=["adcode", "year"], columns="season",
                          values="kndvi_mean_z")
    f_jja = panel[panel.season == "JJA"].set_index(["adcode", "year"])["dry_z"]
    both = pd.DataFrame({"dry_jja": f_jja, "kz_jja": w["JJA"],
                         "kz_son": w["SON"]}).dropna()
    print(f"  corr(dry_z JJA, kNDVI_z JJA) = {both.dry_jja.corr(both.kz_jja):+.3f}")
    print(f"  corr(dry_z JJA, kNDVI_z SON) = {both.dry_jja.corr(both.kz_son):+.3f}"
          "   <- lagged response = recovery")
    print(f"  corr(kNDVI_z JJA, kNDVI_z SON) = {both.kz_jja.corr(both.kz_son):+.3f}"
          "   <- persistence")

    print(f"\nwrote -> {OUT}/panel_seasonal.csv")


if __name__ == "__main__":
    main()
