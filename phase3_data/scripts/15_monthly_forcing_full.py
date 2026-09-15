#!/usr/bin/env python3
"""Monthly climate FORCING per county for the full 2000-2020 period
(PERSIST N1 revised: monthly resolution, event-conditioned).

Same construction as the seasonal forcing, but retains monthly granularity so it
pairs 1:1 with the monthly state stream. Includes the centroid fallback for
sub-cell counties.

Output: data/interim/forcing_monthly_2000_2020.csv
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
CTY = OUT / "yreb_counties_datav.gpkg"
START, END = 2000, 2020
VARS = ["ppt", "pet", "aet", "def", "q", "tmax", "tmin", "vpd", "soil",
        "srad", "pdsi", "swe"]


def main():
    gdf = gpd.read_file(CTY).reset_index(drop=True)
    gdf["adcode"] = gdf.adcode.astype(str)
    n = len(gdf)
    g4 = gdf.to_crs(4326)

    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=pc.sign_inplace)
    a = cat.get_collection("terraclimate").assets["zarr-abfs"]
    kw = dict(a.extra_fields["xarray:open_kwargs"])
    kw.setdefault("engine", "zarr")
    ds = xr.open_dataset(a.href, **kw)

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
    print(f"grid {shape} res {rx:.4f} deg | months {sub.sizes['time']}")

    lab = rasterize(((g, i + 1) for i, g in enumerate(g4.geometry)),
                    out_shape=shape, transform=tr, fill=0, dtype="int32",
                    all_touched=True)
    ins = lab > 0
    lab_in = lab[ins]
    flat = np.full(shape, -1, dtype="int64")
    flat[ins] = np.arange(ins.sum())
    cnt0 = np.bincount(lab_in, minlength=n + 1)[1:]
    fb = {}
    for i in np.nonzero(cnt0 == 0)[0]:
        cx, cy = g4.geometry.iloc[i].centroid.coords[0]
        c, r = int((cx - tr.c) / rx), int((tr.f - cy) / ry)
        if 0 <= r < shape[0] and 0 <= c < shape[1] and flat[r, c] >= 0:
            fb[i] = int(flat[r, c])
    print(f"counties with >=1 cell {int((cnt0>0).sum())}/{n} | "
          f"centroid fallback {len(fb)}")

    times = pd.to_datetime(sub.time.values)
    recs = []
    for v in VARS:
        if v not in sub:
            print(f"  {v}: absent")
            continue
        print(f"  {v} ...", flush=True)
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
            recs.append(pd.DataFrame({"adcode": gdf.adcode.values,
                                      "year": t.year, "month": t.month,
                                      "var": v, "value": vals}))

    long = pd.concat(recs, ignore_index=True)
    wide = long.pivot_table(index=["adcode", "year", "month"], columns="var",
                            values="value").reset_index()
    wide.columns.name = None
    wide["wbal"] = wide["ppt"] - wide["pet"]

    # anomalies vs county x calendar-month climatology (removes seasonal cycle)
    for c in [x for x in VARS if x in wide] + ["wbal"]:
        g = wide.groupby(["adcode", "month"])[c]
        wide[f"{c}_z"] = (wide[c] - g.transform("mean")) / \
            g.transform("std").replace(0, np.nan)
    wide["heat_z"] = wide["tmax_z"]
    wide["dry_z"] = -wide["wbal_z"]
    wide = wide.sort_values(["adcode", "year", "month"]).reset_index(drop=True)
    wide.to_csv(OUT / f"forcing_monthly_{START}_{END}.csv", index=False)

    print(f"\nrows {len(wide):,} (expect {n*21*12:,})")
    print(f"counties {wide.adcode.nunique()} | "
          f"months {wide.year.nunique()*12}")
    j = wide.merge(gdf[["adcode", "reach"]], on="adcode")
    print("\nmonthly climatology check (ppt mm, tmax C):")
    print(j.groupby("month")[["ppt", "tmax", "pdsi"]].mean().round(1).to_string())
    print("\nknown drought years, mean dry_z:")
    for y in (2006, 2011, 2013, 2019):
        print(f"  {y}: {j[j.year==y].dry_z.mean():+.3f}")
    print(f"\nwrote -> {OUT}/forcing_monthly_{START}_{END}.csv")


if __name__ == "__main__":
    main()
