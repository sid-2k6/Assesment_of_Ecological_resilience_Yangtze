#!/usr/bin/env python3
"""DECISIVE TEST for the PERSIST N1 revision.

Phase 3c problem B: seasonal z-scored anomalies showed almost no carry-over
(kNDVI corr(JJA,SON) = +0.02), which would make N1's recovery term
unidentifiable. Hypothesis: recovery in humid subtropical vegetation completes
within weeks, so seasonal averaging integrates it away.

This script tests the hypothesis at MONTHLY resolution by estimating an impulse
response function (IRF): after a forcing shock at month t, how large is the
state anomaly at t, t+1, t+2, t+3, and does it decay geometrically?

A geometric decay is exactly what N1's recovery term parameterises, so a clean
decay here means N1 is identifiable at monthly resolution. A flat or noisy
profile means it is not, and N1 must be redesigned.
"""
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import xarray as xr
from pystac_client import Client
from rasterio.features import rasterize
from rasterio.transform import from_origin

OUT = pathlib.Path("data/interim")
MON = OUT / "monthly"
CTY = OUT / "yreb_counties_datav.gpkg"
START, END = 2010, 2014
FORCE_VARS = ["ppt", "pet", "tmax", "srad", "soil", "vpd", "pdsi"]


# ---------------------------------------------------------------- forcing
def monthly_forcing(gdf):
    f = OUT / f"forcing_monthly_{START}_{END}.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"adcode": str})
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=pc.sign_inplace)
    a = cat.get_collection("terraclimate").assets["zarr-abfs"]
    kw = dict(a.extra_fields["xarray:open_kwargs"]); kw.setdefault("engine", "zarr")
    ds = xr.open_dataset(a.href, **kw)

    g4 = gdf.to_crs(4326)
    minx, miny, maxx, maxy = g4.total_bounds
    pad = 0.1
    desc = bool(ds.lat.values[0] > ds.lat.values[-1])
    lat_s = slice(maxy + pad, miny - pad) if desc else slice(miny - pad, maxy + pad)
    sub = ds.sel(lon=slice(minx - pad, maxx + pad), lat=lat_s,
                 time=slice(f"{START}-01-01", f"{END}-12-31"))
    lons, lats = sub.lon.values, sub.lat.values
    rx, ry = float(abs(lons[1] - lons[0])), float(abs(lats[1] - lats[0]))
    tr = from_origin(lons.min() - rx / 2, lats.max() + ry / 2, rx, ry)
    shape = (len(lats), len(lons))
    n = len(gdf)

    lab = rasterize(((geom, i + 1) for i, geom in enumerate(g4.geometry)),
                    out_shape=shape, transform=tr, fill=0, dtype="int32",
                    all_touched=True)
    ins = lab > 0
    lab_in = lab[ins]
    flat = np.full(shape, -1, dtype="int64"); flat[ins] = np.arange(ins.sum())
    cnt0 = np.bincount(lab_in, minlength=n + 1)[1:]
    fb = {}
    for i in np.nonzero(cnt0 == 0)[0]:                 # centroid fallback
        cx, cy = g4.geometry.iloc[i].centroid.coords[0]
        c, r = int((cx - tr.c) / rx), int((tr.f - cy) / ry)
        if 0 <= r < shape[0] and 0 <= c < shape[1] and flat[r, c] >= 0:
            fb[i] = int(flat[r, c])
    times = pd.to_datetime(sub.time.values)

    recs = []
    for v in FORCE_VARS:
        if v not in sub:
            continue
        arr = sub[v].values
        for ti, t in enumerate(times):
            pcell = arr[ti][ins].astype("float64")
            ok = np.isfinite(pcell)
            c = np.bincount(lab_in[ok], minlength=n + 1)
            s = np.bincount(lab_in[ok], weights=pcell[ok], minlength=n + 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.where(c > 0, s / np.maximum(c, 1), np.nan)
            vals = m[1:]
            for i, pos in fb.items():
                if np.isfinite(pcell[pos]):
                    vals[i] = pcell[pos]
            recs.append(pd.DataFrame({"adcode": gdf.adcode.astype(str).values,
                                      "year": t.year, "month": t.month,
                                      "var": v, "value": vals}))
    long = pd.concat(recs, ignore_index=True)
    wide = long.pivot_table(index=["adcode", "year", "month"], columns="var",
                            values="value").reset_index()
    wide.columns.name = None
    wide.to_csv(f, index=False)
    return wide


def load_state():
    out = None
    for p in ("ndvi", "lst"):
        fs = sorted(MON.glob(f"{p}_*.csv"))
        d = pd.concat([pd.read_csv(x, dtype={"adcode": str}) for x in fs],
                      ignore_index=True)
        out = d if out is None else out.merge(d, on=["adcode", "year", "month"],
                                             how="outer")
    return out


def zscore(df, cols, by):
    for c in cols:
        g = df.groupby(by)[c]
        df[f"{c}_z"] = (df[c] - g.transform("mean")) / \
            g.transform("std").replace(0, np.nan)
    return df


def main():
    gdf = gpd.read_file(CTY).reset_index(drop=True)
    gdf["adcode"] = gdf.adcode.astype(str)

    print("building monthly forcing ...", flush=True)
    force = monthly_forcing(gdf)
    state = load_state()
    print(f"state {len(state):,} rows | forcing {len(force):,} rows")

    d = state.merge(force, on=["adcode", "year", "month"], how="inner")
    d["kndvi"] = np.tanh(d["ndvi_mean"] ** 2)
    d["lst_c"] = d["lst_day_mean"] - 273.15
    d["wbal"] = d["ppt"] - d["pet"]
    d["t"] = (d.year - START) * 12 + d.month          # global month index

    # anomalies vs county x calendar-month climatology (removes seasonal cycle)
    d = zscore(d, ["kndvi", "lst_c", "tmax", "wbal", "srad", "soil", "ppt"],
               ["adcode", "month"])
    d["heat_z"] = d["tmax_z"]
    d["dry_z"] = -d["wbal_z"]
    d = d.sort_values(["adcode", "t"]).reset_index(drop=True)

    print(f"\nmerged panel: {len(d):,} rows, {d.adcode.nunique()} counties, "
          f"{d.t.nunique()} months")
    print("NOTE: climatology uses only 5 years, so anomalies are noisier than a "
          "full-period fit would give. This is a feasibility test, not a final "
          "estimate.")

    # =============== 1. lagged persistence of anomalies ===============
    print(f"\n{'='*68}\n1. MONTHLY ANOMALY PERSISTENCE (within county)")
    print("   seasonal baseline from Phase 3c: kNDVI +0.020, LST +0.133\n")
    print(f"   {'lag':>4} {'kNDVI_z':>10} {'LST_z':>10}")
    for lag in (1, 2, 3, 4, 6):
        g = d.groupby("adcode")
        row = []
        for v in ("kndvi_z", "lst_c_z"):
            cur = d[v].values
            lg = g[v].shift(lag).values
            m = np.isfinite(cur) & np.isfinite(lg)
            row.append(np.corrcoef(cur[m], lg[m])[0, 1] if m.sum() > 100 else np.nan)
        print(f"   {lag:>4} {row[0]:>10.3f} {row[1]:>10.3f}")

    # =============== 2. impulse response function =====================
    print(f"\n{'='*68}\n2. IMPULSE RESPONSE: state anomaly after a forcing shock")
    print("   geometric decay => N1 recovery term is identifiable\n")
    for fv in ("heat_z", "dry_z"):
        for sv in ("lst_c_z", "kndvi_z"):
            g = d.groupby("adcode")
            print(f"   {fv} -> {sv}")
            coefs = []
            for k in range(0, 5):
                fut = g[sv].shift(-k).values
                shock = d[fv].values
                m = np.isfinite(fut) & np.isfinite(shock)
                r = np.corrcoef(shock[m], fut[m])[0, 1] if m.sum() > 100 else np.nan
                coefs.append(r)
                print(f"      t+{k}: r = {r:+.3f}")
            c = np.array(coefs)
            if np.isfinite(c[0]) and abs(c[0]) > 0.05:
                ratio = c[1] / c[0] if np.isfinite(c[1]) else np.nan
                print(f"      decay ratio r(t+1)/r(t0) = {ratio:+.3f}"
                      f"  {'<- geometric decay present' if 0.1 < ratio < 0.95 else '<- NOT a clean decay'}")
            print()

    # =============== 3. conditional on LARGE shocks only ==============
    print(f"{'='*68}\n3. LARGE-SHOCK EVENT RESPONSE (|heat_z| > 1.5, JJA months)")
    print("   resilience is defined by response to real disturbance, not noise\n")
    hot = d[(d.heat_z > 1.5) & (d.month.isin([6, 7, 8]))]
    print(f"   n large summer heat shocks: {len(hot):,}")
    idx = {(a, t): i for i, (a, t) in enumerate(zip(d.adcode, d.t))}
    for sv in ("lst_c_z", "kndvi_z"):
        prof = []
        for k in range(0, 5):
            vals = []
            for a, t in zip(hot.adcode, hot.t):
                j = idx.get((a, t + k))
                if j is not None:
                    v = d[sv].iat[j]
                    if np.isfinite(v):
                        vals.append(v)
            prof.append(np.mean(vals) if vals else np.nan)
        s = "  ".join(f"t+{k}={v:+.3f}" for k, v in enumerate(prof))
        print(f"   {sv:10} {s}")
        p = np.array(prof)
        if np.isfinite(p[0]) and p[0] != 0:
            print(f"   {'':10} decay t+1/t0 = {p[1]/p[0]:+.3f}, "
                  f"t+2/t0 = {p[2]/p[0]:+.3f}")

    d.to_csv(OUT / f"monthly_test_panel_{START}_{END}.csv", index=False)
    print(f"\nwrote -> {OUT}/monthly_test_panel_{START}_{END}.csv")


if __name__ == "__main__":
    main()
