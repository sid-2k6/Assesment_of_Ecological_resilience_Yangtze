#!/usr/bin/env python3
"""Extract static terrain indicators per county from Copernicus DEM GLO-90:
mean/std elevation (relief), mean/max slope, and roughness.

The survey identified slope and elevation among the strongest ecological
resilience drivers (Li et al. 2026, Qinling-Daba framework).

Slope must be computed on the raster BEFORE aggregation - averaging elevation
to county level and then differencing would destroy the terrain signal.
"""
import pathlib
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.features import rasterize

CTY = "data/interim/yreb_counties_datav.gpkg"
OUT = pathlib.Path("data/interim")
COLL = "cop-dem-glo-90"          # 90 m: ample for county-scale terrain stats
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def main():
    gdf = gpd.read_file(CTY).reset_index(drop=True)
    gdf["adcode"] = gdf.adcode.astype(str)
    n = len(gdf)
    bbox = tuple(gdf.to_crs(4326).total_bounds)
    print(f"counties: {n}   bbox: {np.round(bbox,2)}")

    cat = Client.open(STAC, modifier=pc.sign_inplace)
    items = list(cat.search(collections=[COLL], bbox=bbox).items())
    print(f"{COLL} tiles intersecting bbox: {len(items)}")

    acc = {k: np.zeros(n + 1) for k in
           ("elev_s", "elev_s2", "elev_c", "slope_s", "slope_s2", "rough_s")}
    elev_max = np.full(n + 1, -np.inf)
    elev_min = np.full(n + 1, np.inf)
    slope_max = np.full(n + 1, -np.inf)

    t0 = time.time()
    used = 0
    for k, it in enumerate(items, 1):
        try:
            with rasterio.open(it.assets["data"].href) as src:
                dem = src.read(1).astype("float32")
                tr, crs = src.transform, src.crs
                nod = src.nodatavals[0]
                # geographic pixel size -> metres (varies with latitude)
                lat = src.bounds.bottom + (src.bounds.top - src.bounds.bottom) / 2
                px_y = abs(tr.e) * 111_320.0
                px_x = abs(tr.a) * 111_320.0 * np.cos(np.radians(lat))

                proj = gdf.to_crs(crs)
                labels = rasterize(
                    ((g, i + 1) for i, g in enumerate(proj.geometry)),
                    out_shape=dem.shape, transform=tr, fill=0, dtype="int32")
        except Exception as e:
            print(f"  [{k}/{len(items)}] read failed: {e}")
            continue

        inside = labels > 0
        if not inside.any():
            continue
        used += 1

        if nod is not None:
            dem = np.where(dem == nod, np.nan, dem)
        # ocean / void
        dem = np.where(dem < -400, np.nan, dem)

        # slope in degrees from the elevation gradient
        gy, gx = np.gradient(dem, px_y, px_x)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        # roughness: local std via gradient magnitude of slope
        rough = np.hypot(*np.gradient(slope))

        m = inside & np.isfinite(dem) & np.isfinite(slope)
        if not m.any():
            continue
        li = labels[m]
        e, s, r = dem[m], slope[m], rough[m]

        acc["elev_c"] += np.bincount(li, minlength=n + 1)
        acc["elev_s"] += np.bincount(li, weights=e, minlength=n + 1)
        acc["elev_s2"] += np.bincount(li, weights=e * e, minlength=n + 1)
        acc["slope_s"] += np.bincount(li, weights=s, minlength=n + 1)
        acc["slope_s2"] += np.bincount(li, weights=s * s, minlength=n + 1)
        acc["rough_s"] += np.bincount(li, weights=r, minlength=n + 1)

        # per-county running extremes across tiles
        for arr, tgt, ufunc in ((e, elev_max, np.maximum),
                                (e, elev_min, np.minimum),
                                (s, slope_max, np.maximum)):
            init = -np.inf if ufunc is np.maximum else np.inf
            agg = np.full(n + 1, init)
            ufunc.at(agg, li, arr)
            ufunc(tgt, agg, out=tgt)

        if k % 50 == 0:
            print(f"  [{k}/{len(items)}] tiles with counties so far: {used}  "
                  f"{time.time()-t0:.0f}s")

    c = acc["elev_c"][1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        e_mean = np.where(c > 0, acc["elev_s"][1:] / np.maximum(c, 1), np.nan)
        e_var = np.where(c > 1, acc["elev_s2"][1:] / np.maximum(c, 1) - e_mean**2,
                         np.nan)
        s_mean = np.where(c > 0, acc["slope_s"][1:] / np.maximum(c, 1), np.nan)
        s_var = np.where(c > 1, acc["slope_s2"][1:] / np.maximum(c, 1) - s_mean**2,
                         np.nan)
        r_mean = np.where(c > 0, acc["rough_s"][1:] / np.maximum(c, 1), np.nan)

    fin = lambda a: np.where(np.isfinite(a), a, np.nan)
    df = pd.DataFrame({
        "adcode": gdf.adcode.values,
        "elev_mean": e_mean,
        "elev_std": np.sqrt(np.clip(e_var, 0, None)),
        "elev_min": fin(elev_min[1:]),
        "elev_max": fin(elev_max[1:]),
        "relief": fin(elev_max[1:]) - fin(elev_min[1:]),
        "slope_mean": s_mean,
        "slope_std": np.sqrt(np.clip(s_var, 0, None)),
        "slope_max": fin(slope_max[1:]),
        "roughness": r_mean,
        "dem_n": c.astype(int),
    })
    df.to_csv(OUT / "terrain_county.csv", index=False)

    print(f"\ntiles used: {used}/{len(items)}   elapsed {time.time()-t0:.0f}s")
    print(f"coverage: {(df.dem_n>0).sum()}/{n}")
    j = gdf[["adcode", "name_zh", "province", "reach"]].merge(df, on="adcode")

    print("\n--- VALIDATION: elevation by reach "
          "(expect upstream Tibetan-margin high -> downstream delta low) ---")
    print(j.groupby("reach")[["elev_mean", "slope_mean", "relief"]]
           .mean().round(1).to_string())

    print("\n--- highest-elevation counties (expect W Sichuan / NW Yunnan) ---")
    print(j.nlargest(6, "elev_mean")[
        ["name_zh", "province", "elev_mean", "slope_mean"]].round(1)
        .to_string(index=False))

    print("\n--- lowest-elevation counties (expect Yangtze delta) ---")
    print(j.nsmallest(6, "elev_mean")[
        ["name_zh", "province", "elev_mean", "slope_mean"]].round(1)
        .to_string(index=False))

    print(f"\nwrote -> {OUT}/terrain_county.csv")


if __name__ == "__main__":
    main()
