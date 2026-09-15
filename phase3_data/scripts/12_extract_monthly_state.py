#!/usr/bin/env python3
"""Monthly county-level STATE extraction (PERSIST N1 revision test).

Phase 3c found that seasonal anomalies carry almost no carry-over
(z-scored kNDVI corr(JJA,SON) = +0.02), which would make N1's recovery term
unidentifiable. Hypothesis: humid-subtropical vegetation recovers within weeks,
so recovery happens INSIDE a season and seasonal averaging integrates it away.

This script produces MONTHLY state series (12 steps/yr instead of 4) so the
hypothesis can be tested directly. Monthly also matches TerraClimate's native
cadence exactly, which lets forcing and response be paired without resampling.

Output: data/interim/monthly/state_<year>.csv   (adcode x month)
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
OUTDIR = pathlib.Path("data/interim/monthly")
OUTDIR.mkdir(parents=True, exist_ok=True)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WORKERS = 8

PRODUCTS = {
    "ndvi": dict(collection="modis-13Q1-061", decim=4,
                 bands={"250m_16_days_NDVI": ("ndvi", 1e-4, (-2001, 10001))}),
    "lst":  dict(collection="modis-11A2-061", decim=1,
                 bands={"LST_Day_1km": ("lst_day", 0.02, (7499, 65536))}),
}


def tile_of(i):
    for p in i.split("."):
        if len(p) == 6 and p.startswith("h") and p[3] == "v":
            return p
    return None


def adate_month(i):
    for p in i.split("."):
        if p.startswith("A") and len(p) == 8 and p[1:].isdigit():
            yr, doy = int(p[1:5]), int(p[5:8])
            return (datetime(yr, 1, 1) + timedelta(days=doy - 1)).month
    return None


def dedup(items):
    best = {}
    for it in items:
        p = it.id.split(".")
        if len(p) < 5:
            best[it.id] = it
            continue
        k = (p[0], p[1], p[2])
        if k not in best or p[-1] > best[k].id.split(".")[-1]:
            best[k] = it
    return list(best.values())


class LabelCache:
    def __init__(self, gdf):
        self.gdf, self._c = gdf, {}

    def get(self, href, tile, decim):
        with rasterio.open(href) as src:
            shp = (src.height // decim, src.width // decim)
            tr = src.transform * src.transform.scale(decim, decim)
            crs = src.crs
        key = (tile, shp)
        if key in self._c:
            return self._c[key]
        proj = self.gdf.to_crs(crs)
        lab = rasterize(((g, i + 1) for i, g in enumerate(proj.geometry)),
                        out_shape=shp, transform=tr, fill=0, dtype="int32")
        ins = lab > 0
        self._c[key] = (shp, ins, lab[ins], int(ins.sum()))
        return self._c[key]


def rd(job):
    href, shp = job
    try:
        with rasterio.open(href) as src:
            return src.read(1, out_shape=shp)
    except Exception as e:
        return e


def run(product, years, gdf, cache):
    spec = PRODUCTS[product]
    n = len(gdf)
    cat = Client.open(STAC, modifier=pc.sign_inplace)
    bbox = tuple(gdf.to_crs(4326).total_bounds)
    print(f"\n{'='*66}\n{product.upper()} monthly [{spec['collection']}]\n{'='*66}")

    for year in years:
        outf = OUTDIR / f"{product}_{year}.csv"
        if outf.exists():
            print(f"  {year}: cached")
            continue
        t0 = time.time()
        items = dedup([i for i in cat.search(
            collections=[spec["collection"]], bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31").items()
            if i.id.startswith("MOD")])
        if not items:
            print(f"  {year}: none")
            continue

        acc = {m: {o: dict(s=np.zeros(n + 1), c=np.zeros(n + 1))
                   for _, (o, _, _) in spec["bands"].items()}
               for m in range(1, 13)}

        by_tile = {}
        for i in items:
            by_tile.setdefault(tile_of(i.id), []).append(i)
        by_tile.pop(None, None)

        nrd = 0
        for tile, its in sorted(by_tile.items()):
            href0 = its[0].assets[list(spec["bands"])[0]].href
            shp, ins, lab_in, npix = cache.get(href0, tile, spec["decim"])
            if npix == 0:
                continue
            per_month = {}
            for it in its:
                m = adate_month(it.id)
                if m:
                    per_month.setdefault(m, []).append(it)
            for m, mits in per_month.items():
                for asset, (o, scale, (lo, hi)) in spec["bands"].items():
                    jobs = [(it.assets[asset].href, shp) for it in mits
                            if asset in it.assets]
                    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                        arrs = list(ex.map(rd, jobs))
                    A = acc[m][o]
                    for arr in arrs:
                        nrd += 1
                        if isinstance(arr, Exception):
                            continue
                        a = arr[ins]
                        ok = (a > lo) & (a < hi)
                        if not ok.any():
                            continue
                        v = a[ok].astype("float64") * scale
                        li = lab_in[ok]
                        A["s"] += np.bincount(li, weights=v, minlength=n + 1)
                        A["c"] += np.bincount(li, minlength=n + 1)

        rows = []
        for m in range(1, 13):
            d = pd.DataFrame({"adcode": gdf.adcode.astype(str).values,
                              "year": year, "month": m})
            for _, (o, _, _) in spec["bands"].items():
                A = acc[m][o]
                c, s = A["c"][1:], A["s"][1:]
                with np.errstate(invalid="ignore", divide="ignore"):
                    d[f"{o}_mean"] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
                d[f"{o}_n"] = c.astype(int)
            rows.append(d)
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(outf, index=False)
        o0 = list(spec["bands"].values())[0][0]
        nm = out.groupby("month")[f"{o0}_n"].apply(lambda x: (x > 0).sum())
        print(f"  {year}: {len(items):>3} items, {nrd:>4} reads, "
              f"{time.time()-t0:>5.1f}s, months covered "
              f"{int((nm>0).sum())}/12, min county cov {nm.min()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["ndvi", "lst"])
    ap.add_argument("--start", type=int, default=2009)
    ap.add_argument("--end", type=int, default=2014)
    a = ap.parse_args()
    gdf = gpd.read_file(CTY).reset_index(drop=True)
    print(f"counties {len(gdf)}  years {a.start}-{a.end}")
    cache = LabelCache(gdf)
    for p in a.products:
        run(p, range(a.start, a.end + 1), gdf, cache)
    print("\nDONE ->", OUTDIR)


if __name__ == "__main__":
    sys.exit(main())
