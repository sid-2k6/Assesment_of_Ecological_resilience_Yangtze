#!/usr/bin/env python3
"""Benchmark the MODIS -> county zonal-statistics extraction to establish
whether full-belt, multi-decade, multi-variable extraction is tractable here.

Measures, for one MODIS tile-year:
  (a) STAC search time
  (b) raster read time at full res vs decimated overviews
  (c) zonal statistics time
and extrapolates to the full job.
"""
import time

import geopandas as gpd
import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

CTY = "data/interim/yreb_counties_datav.gpkg"
YEAR = 2020

cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                  modifier=pc.sign_inplace)


def tile_of(item):
    # MOD13Q1.A2020209.h27v05.061.*  -> h27v05
    for p in item.id.split("."):
        if p.startswith("h") and "v" in p and len(p) == 6:
            return p
    return None


def main():
    gdf = gpd.read_file(CTY)
    belt = gdf.to_crs(4326)
    bbox = tuple(belt.total_bounds)
    print(f"counties: {len(gdf)}   belt bbox: "
          f"{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}")

    # ---- (a) how many MODIS tiles cover the belt? ----------------------
    t0 = time.time()
    items = list(cat.search(collections=["modis-13Q1-061"], bbox=bbox,
                            datetime=f"{YEAR}-01-01/{YEAR}-12-31").items())
    t_search = time.time() - t0
    terra = [i for i in items if i.id.startswith("MOD")]
    tiles = sorted({tile_of(i) for i in terra} - {None})
    print(f"\n(a) STAC search: {t_search:.1f}s")
    print(f"    items (Terra+Aqua) : {len(items)}")
    print(f"    Terra-only items   : {len(terra)}")
    print(f"    distinct tiles     : {len(tiles)} -> {tiles}")
    per_tile = len(terra) / max(len(tiles), 1)
    print(f"    composites/tile/yr : {per_tile:.1f}  (expect ~23)")

    # ---- pick the tile with most counties in it -----------------------
    tgt = tiles[len(tiles) // 2]
    sample = [i for i in terra if tile_of(i) == tgt]
    # MODIS STAC items carry start_datetime/end_datetime, not a single datetime
    sample.sort(key=lambda i: (i.properties.get("start_datetime")
                               or i.properties.get("end_datetime") or i.id))
    print(f"\n    benchmarking tile {tgt} ({len(sample)} composites)")

    asset = "250m_16_days_NDVI"
    href = sample[0].assets[asset].href

    with rasterio.open(href) as src:
        crs, shape_full, tr = src.crs, src.shape, src.transform
        ovr = src.overviews(1)
    print(f"    tile shape {shape_full}  overviews {ovr}")

    # counties intersecting this tile
    with rasterio.open(href) as src:
        b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    sel = belt[belt.intersects(
        gpd.GeoSeries.from_wkt([f"POLYGON(({b[0]} {b[1]},{b[2]} {b[1]},"
                                f"{b[2]} {b[3]},{b[0]} {b[3]},{b[0]} {b[1]}))"],
                               crs=4326).iloc[0])]
    print(f"    counties intersecting tile: {len(sel)}")

    # ---- (b) read timing: full res vs decimated -----------------------
    for factor in (1, 4):
        n_read = 3   # time a few, extrapolate
        t0 = time.time()
        for it in sample[:n_read]:
            with rasterio.open(it.assets[asset].href) as src:
                out_shape = (src.height // factor, src.width // factor)
                _ = src.read(1, out_shape=out_shape)
        dt = (time.time() - t0) / n_read
        res_m = 250 * factor
        print(f"\n(b) read @ 1/{factor} ({res_m} m): {dt:.2f}s per composite")
        print(f"    -> per tile-year ({len(sample)} composites): "
              f"{dt*len(sample)/60:.1f} min")

    # ---- (c) zonal stats timing ---------------------------------------
    factor = 4
    with rasterio.open(href) as src:
        out_shape = (src.height // factor, src.width // factor)
        arr = src.read(1, out_shape=out_shape).astype("float32")
        tr_dec = src.transform * src.transform.scale(factor, factor)
        selp = sel.to_crs(src.crs)

    t0 = time.time()
    shapes = ((geom, i + 1) for i, geom in enumerate(selp.geometry))
    labels = rasterize(shapes, out_shape=out_shape, transform=tr_dec,
                       fill=0, dtype="int32")
    t_rast = time.time() - t0

    t0 = time.time()
    vals = arr * 1e-4
    valid = (arr > -3000) & (arr < 10000) & (labels > 0)
    lab = labels[valid]
    v = vals[valid]
    cnt = np.bincount(lab, minlength=len(selp) + 1)
    tot = np.bincount(lab, weights=v, minlength=len(selp) + 1)
    mean = np.divide(tot, cnt, out=np.full_like(tot, np.nan), where=cnt > 0)
    t_zonal = time.time() - t0

    print(f"\n(c) rasterize {len(selp)} polygons : {t_rast:.2f}s")
    print(f"    zonal stats (bincount)     : {t_zonal:.3f}s")
    print(f"    counties with data         : {int((cnt[1:]>0).sum())}/{len(selp)}")
    print(f"    mean pixels per county     : {cnt[1:].mean():.0f}")
    print(f"    sample NDVI means          : "
          f"{np.round(mean[1:6],3)}")

    # ---- extrapolate ---------------------------------------------------
    print(f"\n{'='*62}")
    n_tiles, n_years = len(tiles), 21
    read_dec = 0.0
    n_read = 3
    t0 = time.time()
    for it in sample[:n_read]:
        with rasterio.open(it.assets[asset].href) as src:
            _ = src.read(1, out_shape=(src.height // 4, src.width // 4))
    read_dec = (time.time() - t0) / n_read

    per_tileyear = read_dec * per_tile + t_rast + t_zonal
    total_h = per_tileyear * n_tiles * n_years / 3600
    print(f"EXTRAPOLATION (NDVI, 1 km decimation, 2000-2020)")
    print(f"  tiles={n_tiles}  years={n_years}  "
          f"composites/tile-yr={per_tile:.0f}")
    print(f"  per tile-year : {per_tileyear/60:.1f} min")
    print(f"  ONE variable  : {total_h:.1f} h serial | "
          f"{total_h/8:.1f} h on 8 workers")
    print(f"  SIX variables : {total_h*6:.1f} h serial | "
          f"{total_h*6/8:.1f} h on 8 workers")


if __name__ == "__main__":
    main()
