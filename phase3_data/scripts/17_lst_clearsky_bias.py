#!/usr/bin/env python3
"""QC: quantify LST clear-sky sampling bias (PERSIST N1 primary channel).

MOD11A2 LST is retrieved ONLY under clear sky. Cloudy/rainy periods therefore
contribute fewer observations, and the surviving observations are the clear
(hotter) days. This risks a specific confound that would undermine N1:

    drought -> clear skies -> MORE observations AND higher measured LST

If that is what drives the observed dry_z -> LST coupling (r = 0.38 with a clean
0.412 decay), then N1's "response" is partly a sampling artefact rather than a
land-surface process.

Tests
  1. Is observation count driven by cloudiness?         corr(lst_n, ppt)
  2. Is measured LST inflated when sampling is sparse?   corr(lst_n, lst_c)
  3. Does dry_z -> LST survive controlling for lst_n?    partial correlation
  4. Does the IRF decay survive on WELL-SAMPLED months only?
"""
import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path("data/interim")


def partial_corr(x, y, z):
    """corr(x, y) after linearly removing z from both."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if len(x) < 100:
        return np.nan
    Z = np.c_[np.ones(len(z)), z]
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx, ry = x - Z @ bx, y - Z @ by
    return np.corrcoef(rx, ry)[0, 1]


def main():
    p = pd.read_parquet(OUT / "panel_monthly.parquet")
    p["adcode"] = p.adcode.astype(str)
    print(f"panel {len(p):,} rows")

    n_col = "lst_day_n"
    print(f"\n{'='*70}\nOBSERVATION COUNT DISTRIBUTION ({n_col})")
    print(f"  pixels-per-county-month: min {p[n_col].min():.0f} / "
          f"median {p[n_col].median():.0f} / max {p[n_col].max():.0f}")
    zero = (p[n_col] == 0).mean() * 100
    print(f"  county-months with ZERO valid LST pixels: {zero:.2f}%")

    # normalise count within county so cross-sectional size differences drop out
    g = p.groupby("adcode")[n_col]
    p["n_rel"] = p[n_col] / g.transform("median").replace(0, np.nan)
    print(f"  relative sampling (1.0 = county's own median): "
          f"p05={p.n_rel.quantile(.05):.2f}  p50={p.n_rel.quantile(.5):.2f}  "
          f"p95={p.n_rel.quantile(.95):.2f}")

    # ---- TEST 1: is sampling driven by cloudiness? -------------------
    print(f"\n{'='*70}\nTEST 1 - is observation count driven by cloud/rain?")
    for v in ("ppt", "srad", "vpd"):
        if v in p:
            r = p[[v, "n_rel"]].dropna().corr().iloc[0, 1]
            print(f"  corr({v:5}, n_rel) = {r:+.3f}")
    print("  expect ppt NEGATIVE and srad POSITIVE if clear-sky sampling")

    # ---- TEST 2: is LST inflated when sampling is sparse? ------------
    print(f"\n{'='*70}\nTEST 2 - is measured LST biased by sampling density?")
    r = p[["n_rel", "lst_c"]].dropna().corr().iloc[0, 1]
    print(f"  corr(n_rel, lst_c)   = {r:+.3f}  (raw, confounded by season)")
    r2 = p[["n_rel", "lst_c_z"]].dropna().corr().iloc[0, 1]
    print(f"  corr(n_rel, lst_c_z) = {r2:+.3f}  (anomaly, season removed)")
    print("  a LARGE POSITIVE value here would mean sparse months read cooler,")
    print("  i.e. the anomaly partly tracks sampling rather than land surface")

    # by sampling quintile
    p["nq"] = pd.qcut(p.n_rel, 5, labels=["Q1 sparse", "Q2", "Q3", "Q4",
                                          "Q5 dense"], duplicates="drop")
    print("\n  mean LST anomaly by sampling quintile:")
    print(p.groupby("nq", observed=True)[["lst_c_z", "ppt", "n_rel"]]
           .mean().round(3).to_string())

    # ---- TEST 3: does dry_z -> LST survive controlling for sampling? --
    print(f"\n{'='*70}\nTEST 3 - does dry_z -> LST survive controlling for n_rel?")
    jja = p[p.month.isin([6, 7, 8])]
    for label, d in (("all months", p), ("JJA only", jja)):
        raw = d[["dry_z", "lst_c_z"]].dropna().corr().iloc[0, 1]
        pc = partial_corr(d.dry_z.values, d.lst_c_z.values, d.n_rel.values)
        drop = (1 - pc / raw) * 100 if raw else np.nan
        print(f"  {label:10} raw r = {raw:+.3f} | partial r = {pc:+.3f} "
              f"| attenuation {drop:+.1f}%")
    print("  a small attenuation means the coupling is a real land-surface")
    print("  response; a collapse toward zero would mean it is a sampling artefact")

    # ---- TEST 4: IRF on well-sampled months only ---------------------
    print(f"\n{'='*70}\nTEST 4 - does the shock-response decay survive on")
    print("         WELL-SAMPLED months only (n_rel >= 0.9)?")
    d = p.sort_values(["adcode", "t"]).copy()
    idx = {(a, t): i for i, (a, t) in enumerate(zip(d.adcode, d.t))}
    lst = d.lst_c_z.values
    nrel = d.n_rel.values

    for tag, require_dense in (("ALL months", False), ("DENSE only", True)):
        hot = d[(d.heat_z > 1.5) & (d.month.isin([6, 7, 8]))]
        if require_dense:
            hot = hot[hot.n_rel >= 0.9]
        prof = []
        for k in range(5):
            vals = []
            for a, t in zip(hot.adcode, hot.t):
                j = idx.get((a, t + k))
                if j is None:
                    continue
                if require_dense and not (np.isfinite(nrel[j]) and nrel[j] >= 0.9):
                    continue
                if np.isfinite(lst[j]):
                    vals.append(lst[j])
            prof.append(np.mean(vals) if vals else np.nan)
        s = "  ".join(f"t+{k}={v:+.3f}" for k, v in enumerate(prof))
        pa = np.array(prof)
        ratio = pa[1] / pa[0] if np.isfinite(pa[0]) and pa[0] else np.nan
        print(f"  {tag:11} n={len(hot):>5}  {s}")
        print(f"  {'':11} decay t+1/t0 = {ratio:+.3f}")

    # ---- verdict ------------------------------------------------------
    print(f"\n{'='*70}\nVERDICT INPUTS")
    r_ppt = p[["ppt", "n_rel"]].dropna().corr().iloc[0, 1]
    r_nz = p[["n_rel", "lst_c_z"]].dropna().corr().iloc[0, 1]
    raw = jja[["dry_z", "lst_c_z"]].dropna().corr().iloc[0, 1]
    pc = partial_corr(jja.dry_z.values, jja.lst_c_z.values, jja.n_rel.values)
    print(f"  clear-sky sampling confirmed : corr(ppt, n_rel) = {r_ppt:+.3f}")
    print(f"  sampling->anomaly leakage    : corr(n_rel, lst_z) = {r_nz:+.3f}")
    print(f"  dry_z->LST attenuation (JJA) : {raw:+.3f} -> {pc:+.3f}")
    sev = "SEVERE" if abs(r_nz) > 0.3 or (raw and pc / raw < 0.5) else \
          "MODERATE" if abs(r_nz) > 0.15 else "MILD"
    print(f"\n  BIAS SEVERITY: {sev}")


if __name__ == "__main__":
    main()
