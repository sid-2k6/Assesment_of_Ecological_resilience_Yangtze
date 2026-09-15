#!/usr/bin/env python3
"""Seasonal county-level RS compositing for the YREB, 2000-2020 (PERSIST N5/N1).

Motivation (Phase 3 finding F2): annual aggregation destroys the recovery
signal. The 2013 record Yangtze heatwave is invisible in annual kNDVI yet plain
in LST. Since recovery IS resilience, seasonal resolution is mandatory.

Seasons use the standard climatological convention, where December belongs to
the FOLLOWING year's winter:
    DJF(Y) = Dec(Y-1), Jan(Y), Feb(Y)
    MAM(Y) = Mar-May(Y)      JJA(Y) = Jun-Aug(Y)      SON(Y) = Sep-Nov(Y)
The search window for target year Y is therefore (Y-1)-12-01 .. Y-11-30.

Caveat recorded in the output: Terra launched Feb 2000, so DJF(2000) is
necessarily partial. The *_n columns allow this to be filtered downstream.

Output: data/interim/seasonal/<product>_<year>.csv  (long: adcode x season)
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.features import rasterize

CTY = "data/interim/yreb_counties_datav.gpkg"
OUTDIR = pathlib.Path("data/interim/seasonal")
OUTDIR.mkdir(parents=True, exist_ok=True)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WORKERS = 8

MONTH_SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
                6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
SEASONS = ["DJF", "MAM", "JJA", "SON"]

PRODUCTS = {
    "ndvi": dict(collection="modis-13Q1-061", decim=4,
                 bands={"250m_16_days_NDVI": ("ndvi", 1e-4, (-2001, 10001)),
                        "250m_16_days_EVI":  ("evi",  1e-4, (-2001, 10001))}),
    "lst":  dict(collection="modis-11A2-061", decim=1,
                 bands={"LST_Day_1km":   ("lst_day",   0.02, (7499, 65536)),
                        "LST_Night_1km": ("lst_night", 0.02, (7499, 65536))}),
}


def tile_of(item_id):
    for p in item_id.split("."):
        if len(p) == 6 and p.startswith("h") and p[3] == "v":
            return p
    return None


def adate_to_month(item_id):
    """MOD13Q1.A2020209.h27v05... -> month of day-of-year 209 in 2020."""
    for p in item_id.split("."):
        if p.startswith("A") and len(p) == 8 and p[1:].isdigit():
            yr, doy = int(p[1:5]), int(p[5:8])
            return (datetime(yr, 1, 1) + timedelta(days=doy - 1)).month
    return None


def dedup_reprocessed(items):
    """Keep only the latest production timestamp per (product, A-date, tile).
    MPC serves multiple reprocessed generations that would otherwise be summed."""
    best = {}
    for it in items:
        p = it.id.split(".")
        if len(p) < 5:
            best[it.id] = it
            continue
        key = (p[0], p[1], p[2])
        if key not in best or p[-1] > best[key].id.split(".")[-1]:
            best[key] = it
    return list(best.values())


class LabelCache:
    def __init__(self, gdf):
        self.gdf = gdf
        self._c = {}

    def get(self, href, tile, decim):
        with rasterio.open(href) as src:
            out_shape = (src.height // decim, src.width // decim)
            tr = src.transform * src.transform.scale(decim, decim)
            crs = src.crs
        key = (tile, out_shape)          # out_shape, not decim - see Phase 3b
        if key in self._c:
            return self._c[key]
        proj = self.gdf.to_crs(crs)
        labels = rasterize(((g, i + 1) for i, g in enumerate(proj.geometry)),
                           out_shape=out_shape, transform=tr, fill=0,
                           dtype="int32")
        inside = labels > 0
        self._c[key] = (out_shape, inside, labels[inside], int(inside.sum()))
        return self._c[key]


def read_one(job):
    href, out_shape = job
    try:
        with rasterio.open(href) as src:
            return src.read(1, out_shape=out_shape)
    except Exception as e:
        return e


def extract(product, years, gdf, cache):
    spec = PRODUCTS[product]
    coll, decim = spec["collection"], spec["decim"]
    n = len(gdf)
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    bbox = tuple(gdf.to_crs(4326).total_bounds)
    outs = {o for _, (o, _, _) in spec["bands"].items()}

    print(f"\n{'='*72}\n{product.upper()} seasonal  [{coll}]\n{'='*72}")

    for year in years:
        outf = OUTDIR / f"{product}_{year}.csv"
        if outf.exists():
            print(f"  {year}: cached, skipped")
            continue
        t0 = time.time()

        # window spanning Dec(Y-1) .. Nov(Y) so DJF(Y) is complete
        items = list(cat.search(
            collections=[coll], bbox=bbox,
            datetime=f"{year-1}-12-01/{year}-11-30").items())
        terra = dedup_reprocessed([i for i in items if i.id.startswith("MOD")])
        if not terra:
            print(f"  {year}: no Terra items, skipped")
            continue

        # accumulators: [season][outband] -> arrays
        acc = {s: {o: dict(s=np.zeros(n + 1), s2=np.zeros(n + 1),
                           c=np.zeros(n + 1), mx=np.full(n + 1, -np.inf))
                   for o in outs} for s in SEASONS}

        by_tile = {}
        for i in terra:
            by_tile.setdefault(tile_of(i.id), []).append(i)
        by_tile.pop(None, None)

        n_reads = n_fail = 0
        for tile, its in sorted(by_tile.items()):
            href0 = its[0].assets[list(spec["bands"])[0]].href
            out_shape, inside, lab_in, npix = cache.get(href0, tile, decim)
            if npix == 0:
                continue

            # group this tile's items by season
            per_season = {}
            for it in its:
                mon = adate_to_month(it.id)
                if mon is None:
                    continue
                per_season.setdefault(MONTH_SEASON[mon], []).append(it)

            for season, sits in per_season.items():
                for asset, (out, scale, (lo, hi)) in spec["bands"].items():
                    jobs = [(it.assets[asset].href, out_shape)
                            for it in sits if asset in it.assets]
                    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                        arrays = list(ex.map(read_one, jobs))
                    A = acc[season][out]
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
                        cnt = np.bincount(li, minlength=n + 1)
                        tot = np.bincount(li, weights=v, minlength=n + 1)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            cm = np.where(cnt > 0, tot / np.maximum(cnt, 1),
                                          -np.inf)
                        np.maximum(A["mx"], cm, out=A["mx"])

        rows = []
        for season in SEASONS:
            d = pd.DataFrame({"adcode": gdf.adcode.astype(str).values,
                              "year": year, "season": season})
            for o in sorted(outs):
                A = acc[season][o]
                c, s, s2 = A["c"][1:], A["s"][1:], A["s2"][1:]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
                    var = np.where(c > 1, s2 / np.maximum(c, 1) - mean ** 2,
                                   np.nan)
                d[f"{o}_mean"] = mean
                d[f"{o}_std"] = np.sqrt(np.clip(var, 0, None))
                d[f"{o}_max"] = np.where(np.isfinite(A["mx"][1:]),
                                         A["mx"][1:], np.nan)
                d[f"{o}_n"] = c.astype(int)
            rows.append(d)
        out_df = pd.concat(rows, ignore_index=True)
        out_df.to_csv(outf, index=False)

        first = sorted(outs)[0]
        cov = out_df.groupby("season")[f"{first}_n"].apply(lambda x: (x > 0).sum())
        print(f"  {year}: {len(terra):>3} items, {n_reads:>4} reads "
              f"({n_fail} fail), {time.time()-t0:>5.1f}s | coverage "
              + " ".join(f"{s}={cov.get(s,0)}" for s in SEASONS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["ndvi", "lst"],
                    choices=list(PRODUCTS))
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=2020)
    a = ap.parse_args()

    gdf = gpd.read_file(CTY).reset_index(drop=True)
    print(f"counties {len(gdf)}   years {a.start}-{a.end}   seasons {SEASONS}")
    cache = LabelCache(gdf)
    for p in a.products:
        extract(p, range(a.start, a.end + 1), gdf, cache)
    print("\nDONE ->", OUTDIR)


if __name__ == "__main__":
    sys.exit(main())
