#!/usr/bin/env python3
"""Seasonal climate FORCING stream per county, 2000-2020 (PERSIST N1).

N1 (Disturbance-Conditioned Response Decoder) requires a forcing stream that is
separate from the ecological state stream. Without it there is no disturbance to
condition on and the decoder cannot be identified. This script builds it.

Source: TerraClimate via Planetary Computer (Zarr), monthly, 1/24 deg (~4.6 km).

Variables
  fluxes  (season SUM) : ppt, pet, aet, def, q
  states  (season MEAN): tmax, tmin, vpd, soil, srad, pdsi

Derived forcing
  *_z         : z-score anomaly vs each county's own season climatology
  wbal        : water balance ppt - pet  (mm)
  spei_like   : standardised wbal per county x season == SPEI construction
  heat_z      : tmax anomaly (heat forcing)
  dry_z       : negated spei_like  (positive = drier == stronger drought forcing)

Seasons follow the climatological convention used for the RS stream:
DJF(Y) = Dec(Y-1), Jan(Y), Feb(Y).

Output: data/interim/climate_forcing_seasonal.csv  (adcode x year x season)
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

CTY = "data/interim/yreb_counties_datav.gpkg"
OUT = pathlib.Path("data/interim")
START, END = 2000, 2020

SUM_VARS = ["ppt", "pet", "aet", "def", "q"]
MEAN_VARS = ["tmax", "tmin", "vpd", "soil", "srad", "pdsi"]
ALL_VARS = SUM_VARS + MEAN_VARS

MONTH_SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASONS = ["DJF", "MAM", "JJA", "SON"]


def open_terraclimate():
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=pc.sign_inplace)
    # Use the abfs asset, NOT zarr-https: the https href carries the SAS token
    # as a query string, and fsspec appends '/zarr.json' *after* the query,
    # producing a malformed URL and a 403. abfs passes the credential via
    # storage_options instead, which is the intended access path.
    asset = cat.get_collection("terraclimate").assets["zarr-abfs"]
    kw = dict(asset.extra_fields["xarray:open_kwargs"])
    kw.setdefault("engine", "zarr")
    return xr.open_dataset(asset.href, **kw)


def main():
    gdf = gpd.read_file(CTY).reset_index(drop=True)
    gdf["adcode"] = gdf.adcode.astype(str)
    n = len(gdf)
    g4 = gdf.to_crs(4326)
    minx, miny, maxx, maxy = g4.total_bounds
    print(f"counties {n}  bbox {minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f}")

    ds = open_terraclimate()
    print(f"terraclimate dims: {dict(ds.sizes)}")
    print(f"time range: {str(ds.time.values[0])[:10]} .. "
          f"{str(ds.time.values[-1])[:10]}")

    # pad bbox by one cell so edge counties are fully covered
    pad = 0.1
    lat_desc = bool(ds.lat.values[0] > ds.lat.values[-1])
    lat_slice = slice(maxy + pad, miny - pad) if lat_desc \
        else slice(miny - pad, maxy + pad)
    sub = ds.sel(lon=slice(minx - pad, maxx + pad), lat=lat_slice,
                 time=slice(f"{START-1}-12-01", f"{END}-11-30"))
    print(f"subset dims: {dict(sub.sizes)}")

    lons = sub.lon.values
    lats = sub.lat.values
    res_x = float(abs(lons[1] - lons[0]))
    res_y = float(abs(lats[1] - lats[0]))
    transform = from_origin(lons.min() - res_x / 2,
                            lats.max() + res_y / 2, res_x, res_y)
    shape = (len(lats), len(lons))
    print(f"grid {shape}  res {res_x:.4f} deg (~{res_x*111:.1f} km)")

    # all_touched=True guarantees sub-cell counties still receive a value:
    # the smallest YREB county is ~20 km2 against a ~21 km2 cell.
    labels = rasterize(((geom, i + 1) for i, geom in enumerate(g4.geometry)),
                       out_shape=shape, transform=transform, fill=0,
                       dtype="int32", all_touched=True)
    inside = labels > 0
    lab_in = labels[inside]
    cell_counts = np.bincount(lab_in, minlength=n + 1)[1:]
    print(f"counties receiving >=1 cell: {int((cell_counts>0).sum())}/{n}")

    # ---- centroid fallback for sub-cell counties -----------------------
    # rasterize() assigns each cell to ONE polygon, so a small county sharing a
    # cell with a larger neighbour is overwritten and receives zero cells. The
    # affected units are tiny urban districts (median ~62 km2 against ~21 km2
    # cells) - i.e. exactly the dense cores that also lose MOD16 ET. Dropping
    # them would bias the east-west comparison, so sample the nearest grid cell
    # at the county centroid instead. Climate fields are spatially smooth at
    # ~4.6 km, so this is an adequate estimator for a 20 km2 district.
    flat_pos = np.full(shape, -1, dtype="int64")
    flat_pos[inside] = np.arange(inside.sum())
    in_rows, in_cols = np.nonzero(inside)

    fallback = {}
    missing = np.nonzero(cell_counts == 0)[0]
    for i in missing:
        cx, cy = g4.geometry.iloc[i].centroid.coords[0]
        col = int((cx - transform.c) / res_x)
        row = int((transform.f - cy) / res_y)
        if 0 <= row < shape[0] and 0 <= col < shape[1] and flat_pos[row, col] >= 0:
            fallback[i] = int(flat_pos[row, col])
        else:                                     # nearest inside cell
            d = (in_rows - row) ** 2 + (in_cols - col) ** 2
            fallback[i] = int(np.argmin(d))
    print(f"centroid fallback applied to {len(fallback)} sub-cell counties")
    print(f"cells per county: min {cell_counts.min()} / "
          f"median {int(np.median(cell_counts))} / max {cell_counts.max()}")

    # ---- seasonal aggregation ----------------------------------------
    months = pd.to_datetime(sub.time.values)
    season_of = np.array([MONTH_SEASON[m] for m in months.month])
    # December belongs to the FOLLOWING year's winter
    season_year = np.where(months.month == 12, months.year + 1, months.year)

    recs = []
    for v in ALL_VARS:
        if v not in sub:
            print(f"  {v}: absent, skipped")
            continue
        agg = "sum" if v in SUM_VARS else "mean"
        print(f"  loading {v} ({agg} over season) ...", flush=True)
        arr = sub[v].values          # (time, lat, lon)
        arr = np.where(np.isfinite(arr), arr, np.nan)

        for yr in range(START, END + 1):
            for s in SEASONS:
                m = (season_year == yr) & (season_of == s)
                if not m.any():
                    continue
                block = arr[m][:, inside]                    # (months, cells)
                with np.errstate(invalid="ignore"):
                    per_cell = (np.nansum(block, axis=0) if agg == "sum"
                                else np.nanmean(block, axis=0))
                    allnan = np.all(~np.isfinite(block), axis=0)
                    per_cell = np.where(allnan, np.nan, per_cell)
                ok = np.isfinite(per_cell)
                cnt = np.bincount(lab_in[ok], minlength=n + 1)
                tot = np.bincount(lab_in[ok], weights=per_cell[ok],
                                  minlength=n + 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cmean = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
                vals = cmean[1:]
                ncell = cnt[1:].astype(float)
                # inject centroid-sampled values for sub-cell counties
                for i, pos in fallback.items():
                    pv = per_cell[pos]
                    if np.isfinite(pv):
                        vals[i] = pv
                        ncell[i] = 0.5          # 0.5 flags a fallback estimate
                recs.append(pd.DataFrame({
                    "adcode": gdf.adcode.values, "year": yr, "season": s,
                    "var": v, "value": vals, "n_cells": ncell}))

    long = pd.concat(recs, ignore_index=True)
    wide = long.pivot_table(index=["adcode", "year", "season"], columns="var",
                            values="value").reset_index()
    wide.columns.name = None

    # ---- derived forcing ---------------------------------------------
    wide["wbal"] = wide["ppt"] - wide["pet"]

    def zscore(df, cols):
        for c in cols:
            grp = df.groupby(["adcode", "season"])[c]
            mu, sd = grp.transform("mean"), grp.transform("std")
            df[f"{c}_z"] = (df[c] - mu) / sd.replace(0, np.nan)
        return df

    wide = zscore(wide, [c for c in ALL_VARS if c in wide] + ["wbal"])
    wide["spei_like"] = wide["wbal_z"]
    wide["dry_z"] = -wide["wbal_z"]          # positive = drought forcing
    wide["heat_z"] = wide["tmax_z"]
    wide = wide.sort_values(["adcode", "year", "season"]).reset_index(drop=True)
    wide.to_csv(OUT / "climate_forcing_seasonal.csv", index=False)

    # ================= VALIDATION =====================================
    j = wide.merge(gdf[["adcode", "reach", "province"]], on="adcode")
    print(f"\n{'='*68}\nrows {len(wide):,}  ({n} counties x 21 yr x 4 seasons "
          f"= {n*21*4:,})")

    print("\n--- seasonal climatology (expect JJA monsoon max) ---")
    print(j.groupby("season")[["ppt", "tmax", "pet", "pdsi", "soil"]]
           .mean().round(1).to_string())

    print("\n--- annual precipitation by reach (expect downstream/SE wetter) ---")
    ann = j.groupby(["reach", "adcode", "year"]).ppt.sum().reset_index()
    print(ann.groupby("reach").ppt.mean().round(0).to_string())

    print("\n--- KNOWN DROUGHT YEARS: mean dry_z (positive = drier) ---")
    dz = j.groupby("year")[["dry_z", "heat_z", "pdsi"]].mean()
    for y in (2006, 2011, 2013, 2019, 2020):
        if y in dz.index:
            r = dz.loc[y]
            print(f"  {y}: dry_z={r.dry_z:+.3f}  heat_z={r.heat_z:+.3f}  "
                  f"pdsi={r.pdsi:+.3f}")

    print("\n--- 2006 SW China drought: dry_z by province (JJA) ---")
    sw = j[(j.year == 2006) & (j.season == "JJA")]
    print(sw.groupby("province").dry_z.mean().sort_values(
        ascending=False).round(2).head(6).to_string())

    print("\n--- 2013 record Yangtze heatwave: heat_z by reach (JJA) ---")
    h = j[(j.year == 2013) & (j.season == "JJA")]
    print(h.groupby("reach")[["heat_z", "dry_z"]].mean().round(3).to_string())

    print(f"\nwrote -> {OUT}/climate_forcing_seasonal.csv")


if __name__ == "__main__":
    main()
