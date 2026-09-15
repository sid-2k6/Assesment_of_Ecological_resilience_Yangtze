#!/usr/bin/env python3
"""Feasibility test: can we search AND read actual pixel values from Microsoft
Planetary Computer anonymously (no Google Earth Engine, no NASA Earthdata)?

Reads a small window over Wuhan (central YREB) from three collections.
"""
import numpy as np
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from pystac_client import Client

# Small AOI over Wuhan, central YREB
AOI = {"type": "Point", "coordinates": [114.30, 30.59]}
BBOX = (114.25, 30.54, 114.35, 30.64)

cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                  modifier=pc.sign_inplace)

TESTS = [
    ("modis-13Q1-061",     "2020-07-01/2020-07-31", "250m_16_days_NDVI", 1e-4),
    ("modis-11A2-061",     "2020-07-01/2020-07-31", "LST_Day_1km",       0.02),
    ("modis-17A3HGF-061",  "2020-01-01/2020-12-31", "Npp_500m",          1e-4),
]

for coll, when, asset, scale in TESTS:
    print(f"\n--- {coll} / {asset} ---")
    try:
        search = cat.search(collections=[coll], intersects=AOI, datetime=when)
        items = list(search.items())
        print(f"items found: {len(items)}")
        if not items:
            print("  NO ITEMS")
            continue
        it = items[0]
        print(f"item: {it.id}")
        if asset not in it.assets:
            print(f"  asset missing. available: {list(it.assets)[:6]}")
            continue
        href = it.assets[asset].href
        with rasterio.open(href) as src:
            print(f"crs={src.crs} shape={src.shape} dtype={src.dtypes[0]}")
            # reproject AOI bbox into the raster CRS
            from rasterio.warp import transform_bounds
            b = transform_bounds("EPSG:4326", src.crs, *BBOX)
            win = from_bounds(*b, transform=src.transform)
            arr = src.read(1, window=win).astype("float64")
            nod = src.nodatavals[0]
            if nod is not None:
                arr = np.where(arr == nod, np.nan, arr)
            vals = arr * scale
        print(f"window shape: {arr.shape}")
        print(f"scaled min/mean/max: {np.nanmin(vals):.4f} / "
              f"{np.nanmean(vals):.4f} / {np.nanmax(vals):.4f}")
        print("  >>> PIXEL READ OK")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")

# Landsat + DEM quick check
print("\n--- landsat-c2-l2 ---")
try:
    items = list(cat.search(collections=["landsat-c2-l2"], intersects=AOI,
                            datetime="2020-01-01/2020-12-31",
                            query={"eo:cloud_cover": {"lt": 20},
                                   "platform": {"in": ["landsat-8"]}}).items())
    print(f"low-cloud Landsat-8 scenes over Wuhan in 2020: {len(items)}")
    if items:
        print(f"assets: {sorted(items[0].assets)[:14]}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n--- cop-dem-glo-30 ---")
try:
    items = list(cat.search(collections=["cop-dem-glo-30"], intersects=AOI).items())
    print(f"DEM tiles: {len(items)}")
    if items:
        with rasterio.open(items[0].assets["data"].href) as src:
            from rasterio.warp import transform_bounds
            b = transform_bounds("EPSG:4326", src.crs, *BBOX)
            arr = src.read(1, window=from_bounds(*b, transform=src.transform))
        print(f"elevation min/mean/max (m): {arr.min()} / {arr.mean():.1f} / {arr.max()}")
        print("  >>> DEM READ OK")
except Exception as e:
    print(f"  FAILED: {e}")
