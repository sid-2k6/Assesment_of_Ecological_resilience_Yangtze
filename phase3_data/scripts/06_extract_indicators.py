#!/usr/bin/env python3
"""Extract county-level annual ecological indicators for the YREB, 2000-2020,
from Microsoft Planetary Computer (anonymous access).

Design notes
------------
* County label rasters are rasterized ONCE per MODIS tile and cached, since
  geometry is static across years. This is the single biggest speedup.
* Composite reads are parallelised with threads (network-bound, not CPU-bound).
* Fill values are masked against each product's documented valid range.
  MODIS assets frequently do NOT declare nodata (MOD17A3HGF Npp_500m returns
  32767 as a real value), so src.nodatavals is deliberately NOT trusted.
* Terra (MOD*) only. MPC serves Terra and Aqua in the same collection; mixing
  them breaks series consistency.
* Resumable: completed (product, year) outputs are skipped.

Usage
-----
  python 06_extract_indicators.py --products ndvi lst npp et --start 2000 --end 2020
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.features import rasterize

CTY = "data/interim/yreb_counties_datav.gpkg"
OUTDIR = pathlib.Path("data/interim/indicators")
OUTDIR.mkdir(parents=True, exist_ok=True)

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WORKERS = 8

# --------------------------------------------------------------------------
# Product specifications.
#   valid   : (lo, hi) EXCLUSIVE bounds on the raw DN, per MODIS user guides
#   scale   : multiply raw DN to get physical units
#   decim   : decimation factor (native 250 m -> 1 km at 4; 1 km stays 1)
#   annual  : True if one file per tile-year (no temporal compositing needed)
# --------------------------------------------------------------------------
PRODUCTS = {
    "ndvi": dict(
        collection="modis-13Q1-061", decim=4, annual=False,
        bands={"250m_16_days_NDVI": ("ndvi", 1e-4, (-2001, 10001)),
               "250m_16_days_EVI":  ("evi",  1e-4, (-2001, 10001))},
        note="16-day, 250 m -> 1 km",
    ),
    "lst": dict(
        collection="modis-11A2-061", decim=1, annual=False,
        bands={"LST_Day_1km":   ("lst_day",   0.02, (7499, 65536)),
               "LST_Night_1km": ("lst_night", 0.02, (7499, 65536))},
        note="8-day, 1 km, Kelvin",
    ),
    "npp": dict(
        collection="modis-17A3HGF-061", decim=1, annual=True,
        bands={"Npp_500m": ("npp", 1e-4, (-30001, 32701))},
        note="annual, 500 m, kgC/m2/yr; fill 32767 leaks - masked explicitly",
    ),
    "et": dict(
        collection="modis-16A3GF-061", decim=1, annual=True,
        bands={"ET_500m": ("et", 0.1, (-32768, 32701))},
        note="annual, 500 m, mm/yr",
    ),
}


def tile_of(item_id: str) -> str | None:
    for p in item_id.split("."):
        if len(p) == 6 and p.startswith("h") and p[3] == "v":
            return p
    return None


def dedup_reprocessed(items):
    """MPC serves MULTIPLE reprocessed versions of the same MODIS granule,
    distinguished only by the trailing production timestamp:

        MOD16A3GF.A2014001.h29v06.061.2022077135259   <- keep (latest)
        MOD16A3GF.A2014001.h29v06.061.2021343120212   <- discard

    Summing all versions double/triple-counts pixels and blends processing
    generations, which manifests as a spurious step change in the time series
    (observed around 2013-2014 for MOD16A3GF/MOD17A3HGF). Keep only the latest
    production timestamp per (product, acquisition date, tile).
    """
    best: dict[tuple, object] = {}
    for it in items:
        parts = it.id.split(".")
        if len(parts) < 5:
            best[it.id] = it
            continue
        key = (parts[0], parts[1], parts[2])       # product, A-date, tile
        prod_ts = parts[-1]                        # production timestamp
        cur = best.get(key)
        if cur is None or prod_ts > cur.id.split(".")[-1]:
            best[key] = it
    return list(best.values())


class LabelCache:
    """Rasterized county labels per MODIS tile grid. Geometry is static, so this
    is computed once per (tile, decim) and reused for every year and band."""

    def __init__(self, gdf: gpd.GeoDataFrame):
        self.gdf = gdf
        self.n = len(gdf)
        self._c: dict = {}

    def get(self, href: str, tile: str, decim: int):
        with rasterio.open(href) as src:
            out_shape = (src.height // decim, src.width // decim)
            tr = src.transform * src.transform.scale(decim, decim)
            crs = src.crs
        # Key MUST include out_shape, not just (tile, decim): NPP/ET are 500 m
        # (2400x2400) and LST is 1 km (1200x1200) yet both use decim=1, so a
        # (tile, decim) key silently returns the wrong label raster.
        key = (tile, out_shape)
        if key in self._c:
            return self._c[key]
        proj = self.gdf.to_crs(crs)
        labels = rasterize(((g, i + 1) for i, g in enumerate(proj.geometry)),
                           out_shape=out_shape, transform=tr, fill=0,
                           dtype="int32")
        inside = labels > 0
        lab_in = labels[inside]
        self._c[key] = (out_shape, inside, lab_in, int((labels > 0).sum()))
        return self._c[key]


def read_band(args):
    href, band, out_shape = args
    try:
        with rasterio.open(href) as src:
            return src.read(1, out_shape=out_shape)
    except Exception as e:                      # transient network / SAS issues
        return e


def extract(product: str, years: range, gdf: gpd.GeoDataFrame, cache: LabelCache):
    spec = PRODUCTS[product]
    coll, decim, annual = spec["collection"], spec["decim"], spec["annual"]
    n = len(gdf)
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    bbox = tuple(gdf.to_crs(4326).total_bounds)

    print(f"\n{'='*70}\n{product.upper()}  [{coll}]  {spec['note']}\n{'='*70}")

    for year in years:
        outf = OUTDIR / f"{product}_{year}.csv"
        if outf.exists():
            print(f"  {year}: cached, skipped")
            continue
        t0 = time.time()

        items = list(cat.search(collections=[coll], bbox=bbox,
                                datetime=f"{year}-01-01/{year}-12-31").items())
        terra = [i for i in items if i.id.startswith("MOD")]   # Terra only
        n_raw = len(terra)
        terra = dedup_reprocessed(terra)
        n_dropped = n_raw - len(terra)
        if not terra:
            print(f"  {year}: NO Terra items - skipped")
            continue

        by_tile: dict[str, list] = {}
        for i in terra:
            by_tile.setdefault(tile_of(i.id), []).append(i)
        by_tile.pop(None, None)

        # accumulators per output band
        acc = {out: dict(s=np.zeros(n + 1), c=np.zeros(n + 1),
                         mx=np.full(n + 1, -np.inf), s2=np.zeros(n + 1))
               for _, (out, _, _) in
               [(k, v) for k, v in spec["bands"].items()]}

        n_reads = n_fail = 0
        tiles_used, tiles_empty = [], []
        for tile, its in sorted(by_tile.items()):
            href0 = its[0].assets[list(spec["bands"])[0]].href
            out_shape, inside, lab_in, npix = cache.get(href0, tile, decim)
            if npix == 0:
                tiles_empty.append(tile)
                continue
            tiles_used.append(f"{tile}({len(np.unique(lab_in))})")

            for asset, (out, scale, (lo, hi)) in spec["bands"].items():
                jobs = [(it.assets[asset].href, asset, out_shape)
                        for it in its if asset in it.assets]
                with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                    arrays = list(ex.map(read_band, jobs))

                A = acc[out]
                for arr in arrays:
                    n_reads += 1
                    if isinstance(arr, Exception):
                        n_fail += 1
                        continue
                    a = arr[inside]
                    ok = (a > lo) & (a < hi)
                    if not ok.any():
                        continue
                    v = a[ok].astype("float64") * scale
                    li = lab_in[ok]
                    A["s"] += np.bincount(li, weights=v, minlength=n + 1)
                    A["s2"] += np.bincount(li, weights=v * v, minlength=n + 1)
                    A["c"] += np.bincount(li, minlength=n + 1)
                    # running per-county max of the composite means
                    mx = np.full(n + 1, -np.inf)
                    cnt = np.bincount(li, minlength=n + 1)
                    tot = np.bincount(li, weights=v, minlength=n + 1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        cm = np.where(cnt > 0, tot / np.maximum(cnt, 1), -np.inf)
                    np.maximum(A["mx"], cm, out=A["mx"])

        # assemble. adcode is kept as a zero-padded STRING throughout: GB/T 2260
        # codes are identifiers, and int64 coercion breaks joins downstream.
        df = pd.DataFrame({"adcode": gdf.adcode.astype(str).values,
                           "year": year})
        for out, A in acc.items():
            c = A["c"][1:]
            s = A["s"][1:]
            s2 = A["s2"][1:]
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
                var = np.where(c > 1, s2 / np.maximum(c, 1) - mean ** 2, np.nan)
            df[f"{out}_mean"] = mean
            df[f"{out}_std"] = np.sqrt(np.clip(var, 0, None))
            df[f"{out}_max"] = np.where(np.isfinite(A["mx"][1:]),
                                        A["mx"][1:], np.nan)
            df[f"{out}_n"] = c.astype(int)

        df.to_csv(outf, index=False)
        first = list(acc)[0]
        cov = int((df[f"{first}_n"] > 0).sum())
        dmsg = f" [dedup -{n_dropped}]" if n_dropped else ""
        print(f"  {year}: {len(terra):>3} items{dmsg}, {n_reads:>4} reads "
              f"({n_fail} fail), cov {cov}/{n}, {time.time()-t0:>5.1f}s  "
              f"{first}_mean={np.nanmean(df[f'{first}_mean']):.4f}")
        if year == years.start:
            print(f"        tiles with counties: {' '.join(tiles_used)}")
            print(f"        tiles empty        : {tiles_empty or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["npp", "et", "ndvi", "lst"],
                    choices=list(PRODUCTS))
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2020)
    a = ap.parse_args()

    gdf = gpd.read_file(CTY).reset_index(drop=True)
    print(f"counties: {len(gdf)}   years: {a.start}-{a.end}")
    cache = LabelCache(gdf)

    for p in a.products:
        extract(p, range(a.start, a.end + 1), gdf, cache)

    print("\nDONE. outputs in", OUTDIR)


if __name__ == "__main__":
    sys.exit(main())
