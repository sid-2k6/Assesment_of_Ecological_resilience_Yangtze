#!/usr/bin/env python3
"""End-to-end validation: extract county-level annual NDVI for the whole YREB
for one year, then check the result against known geography.

This proves the full pipeline works before committing to 21 years x N variables.
"""
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.features import rasterize

CTY = "data/interim/yreb_counties_datav.gpkg"
ASSET = "250m_16_days_NDVI"
DECIM = 4              # 250 m -> 1 km
SCALE = 1e-4
VALID = (-2000, 10000)  # raw DN bounds for MOD13Q1 NDVI
YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2020

cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                  modifier=pc.sign_inplace)


def tile_of(item):
    for p in item.id.split("."):
        if p.startswith("h") and "v" in p and len(p) == 6:
            return p
    return None


def main():
    gdf = gpd.read_file(CTY).reset_index(drop=True)
    bbox = tuple(gdf.to_crs(4326).total_bounds)
    n = len(gdf)

    items = list(cat.search(collections=["modis-13Q1-061"], bbox=bbox,
                            datetime=f"{YEAR}-01-01/{YEAR}-12-31").items())
    terra = [i for i in items if i.id.startswith("MOD")]   # Terra only
    by_tile = {}
    for i in terra:
        by_tile.setdefault(tile_of(i), []).append(i)
    print(f"year {YEAR}: {len(terra)} Terra composites across "
          f"{len(by_tile)} tiles")

    # accumulate per-county sum/count across all tiles
    tot = np.zeros(n + 1)
    cnt = np.zeros(n + 1)
    t_start = time.time()

    for ti, (tile, its) in enumerate(sorted(by_tile.items()), 1):
        t0 = time.time()
        href0 = its[0].assets[ASSET].href
        with rasterio.open(href0) as src:
            out_shape = (src.height // DECIM, src.width // DECIM)
            tr = src.transform * src.transform.scale(DECIM, DECIM)
            tile_crs = src.crs

        # rasterize counties once per tile (geometry is static)
        selp = gdf.to_crs(tile_crs)
        labels = rasterize(((g, i + 1) for i, g in enumerate(selp.geometry)),
                           out_shape=out_shape, transform=tr, fill=0,
                           dtype="int32")
        inside = labels > 0
        if not inside.any():
            print(f"  [{ti}/{len(by_tile)}] {tile}: no counties, skipped")
            continue

        lab_in = labels[inside]
        acc_sum = np.zeros(n + 1)
        acc_cnt = np.zeros(n + 1)

        for it in its:
            with rasterio.open(it.assets[ASSET].href) as src:
                arr = src.read(1, out_shape=out_shape)
            a = arr[inside]
            ok = (a > VALID[0]) & (a < VALID[1])
            if not ok.any():
                continue
            acc_sum += np.bincount(lab_in[ok], weights=a[ok] * SCALE,
                                   minlength=n + 1)
            acc_cnt += np.bincount(lab_in[ok], minlength=n + 1)

        tot += acc_sum
        cnt += acc_cnt
        ncty = int((acc_cnt[1:] > 0).sum())
        print(f"  [{ti}/{len(by_tile)}] {tile}: {len(its)} composites, "
              f"{ncty} counties, {time.time()-t0:.1f}s")

    mean = np.divide(tot, cnt, out=np.full_like(tot, np.nan), where=cnt > 0)
    gdf["ndvi_mean"] = mean[1:]
    gdf["n_obs"] = cnt[1:].astype(int)
    gdf["year"] = YEAR

    res = gdf.drop(columns="geometry")
    res.to_csv(f"data/interim/ndvi_county_{YEAR}.csv", index=False)

    print(f"\ntotal elapsed: {time.time()-t_start:.0f}s")
    print(f"counties with data: {int((gdf.n_obs>0).sum())}/{n}")
    print(f"NDVI range: {gdf.ndvi_mean.min():.3f} to {gdf.ndvi_mean.max():.3f}")
    print(f"NDVI mean : {gdf.ndvi_mean.mean():.3f}")

    print("\n--- VALIDATION: mean NDVI by reach "
          "(expect upstream/midstream forested > downstream urban) ---")
    print(gdf.groupby("reach").ndvi_mean.agg(["mean", "std", "count"]).round(3))

    print("\n--- by province ---")
    print(gdf.groupby("province").ndvi_mean.mean().sort_values(
        ascending=False).round(3).to_string())

    print("\n--- 8 LOWEST NDVI counties (expect dense urban cores) ---")
    lo = gdf.nsmallest(8, "ndvi_mean")[["name_zh", "province", "ndvi_mean"]]
    print(lo.to_string(index=False))

    print("\n--- 8 HIGHEST NDVI counties (expect remote mountain/forest) ---")
    hi = gdf.nlargest(8, "ndvi_mean")[["name_zh", "province", "ndvi_mean"]]
    print(hi.to_string(index=False))

    print(f"\nwrote -> data/interim/ndvi_county_{YEAR}.csv")


if __name__ == "__main__":
    main()
