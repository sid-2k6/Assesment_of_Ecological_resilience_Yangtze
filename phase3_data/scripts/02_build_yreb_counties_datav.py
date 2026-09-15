#!/usr/bin/env python3
"""Build YREB county-level boundaries from DataV (Alibaba) open boundary
service, which carries official GB/T 2260 administrative codes and therefore
matches China's official county-level division (县级行政区) — unlike GADM,
which aggregates urban districts.

Descends: province -> city -> county/district, keeping county-level units.

Outputs:
  data/interim/yreb_counties_datav.gpkg
  data/interim/yreb_counties_datav.csv
  data/raw/boundaries/datav_cache/*.json   (raw responses, for provenance)
"""
import json
import pathlib
import time
import urllib.request

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

BASE = "https://geo.datav.aliyun.com/areas_v3/bound/{}_full.json"
CACHE = pathlib.Path("data/raw/boundaries/datav_cache")
OUT = pathlib.Path("data/interim")
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# 11 provincial-level units of the YREB with official adcodes
PROVINCES = {
    "310000": ("Shanghai",  "downstream"),
    "320000": ("Jiangsu",   "downstream"),
    "330000": ("Zhejiang",  "downstream"),
    "340000": ("Anhui",     "downstream"),
    "360000": ("Jiangxi",   "midstream"),
    "420000": ("Hubei",     "midstream"),
    "430000": ("Hunan",     "midstream"),
    "500000": ("Chongqing", "upstream"),
    "510000": ("Sichuan",   "upstream"),
    "520000": ("Guizhou",   "upstream"),
    "530000": ("Yunnan",    "upstream"),
}

COUNTY_LEVELS = {"district", "county"}   # county-level (县级)
CITY_LEVELS = {"city"}                   # prefecture-level (地级) -> descend


def fetch(adcode, tries=3):
    f = CACHE / f"{adcode}_full.json"
    if f.exists():
        return json.loads(f.read_text())
    url = BASE.format(adcode)
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                data = json.loads(r.read().decode("utf-8", "ignore"))
            f.write_text(json.dumps(data))
            time.sleep(0.25)
            return data
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    print(f"    WARN fetch failed {adcode}: {last}")
    return None


def main():
    rows = []
    for padcode, (pname, reach) in PROVINCES.items():
        pdata = fetch(padcode)
        if not pdata:
            continue
        kids = pdata["features"]
        n_before = len(rows)

        for k in kids:
            kp = k["properties"]
            lvl = (kp.get("level") or "").lower()
            adc = str(kp.get("adcode"))

            if lvl in COUNTY_LEVELS:
                # Municipality child, or province-directly-administered unit
                rows.append(dict(adcode=adc, name_zh=kp.get("name"),
                                 level=lvl, province=pname, reach=reach,
                                 prefecture_zh=None,
                                 geometry=shape(k["geometry"])))
            elif lvl in CITY_LEVELS:
                cdata = fetch(adc)
                # Province-directly-administered county-level units (e.g. Hubei's
                # 仙桃/潜江/天门/神农架林区) are tagged level='city' but have no
                # children. They ARE county-level, so keep them as themselves.
                if not cdata or not cdata.get("features"):
                    rows.append(dict(adcode=adc, name_zh=kp.get("name"),
                                     level="county-level-city", province=pname,
                                     reach=reach, prefecture_zh=None,
                                     geometry=shape(k["geometry"])))
                    print(f"    -> kept as county-level unit: "
                          f"{kp.get('name')} ({adc})")
                    continue
                for c in cdata["features"]:
                    cp = c["properties"]
                    clvl = (cp.get("level") or "").lower()
                    if clvl in COUNTY_LEVELS or clvl in CITY_LEVELS:
                        rows.append(dict(
                            adcode=str(cp.get("adcode")), name_zh=cp.get("name"),
                            level=clvl, province=pname, reach=reach,
                            prefecture_zh=kp.get("name"),
                            geometry=shape(c["geometry"])))
            else:
                print(f"    note: {pname} child level='{lvl}' adcode={adc} "
                      f"name={kp.get('name')}")

        print(f"{pname:10} ({padcode})  children={len(kids):>3}  "
              f"county-level collected={len(rows) - n_before:>4}")

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")

    # Drop duplicate adcodes (a unit can appear via two paths)
    dup = gdf.adcode.duplicated().sum()
    if dup:
        print(f"\ndropping {dup} duplicate adcode rows")
        gdf = gdf.drop_duplicates(subset="adcode").reset_index(drop=True)

    gdf = gdf.to_crs(ALBERS)
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf["county_id"] = gdf.adcode

    gdf.to_file(OUT / "yreb_counties_datav.gpkg", driver="GPKG")
    gdf.drop(columns="geometry").to_csv(OUT / "yreb_counties_datav.csv",
                                        index=False)

    print(f"\n{'='*62}")
    print(f"YREB county-level units (DataV, official adcodes): {len(gdf)}")
    print(f"literature reference (Li et al. 2026)            : 1,070")
    print(f"difference                                       : {len(gdf)-1070:+d}")
    print(f"GADM ADM_3 comparison                            : 925")
    print(f"\ntotal area: {gdf.area_km2.sum():,.0f} km2  (published ~2,050,000)")

    print("\nby reach:")
    for reach, g in gdf.groupby("reach"):
        print(f"  {reach:11} {len(g):>4} units  {g.area_km2.sum():>12,.0f} km2")

    print("\nby province:")
    t = (gdf.groupby(["reach", "province"])
            .agg(n=("adcode", "size"), km2=("area_km2", "sum"))
            .reset_index().sort_values(["reach", "n"], ascending=[True, False]))
    for _, r in t.iterrows():
        print(f"  {r['reach']:11} {r['province']:10} {r['n']:>4} units "
              f"{r['km2']:>12,.0f} km2")

    print("\nlevel composition:", gdf.level.value_counts().to_dict())
    print(f"area: min {gdf.area_km2.min():.1f} / median "
          f"{gdf.area_km2.median():.0f} / max {gdf.area_km2.max():,.0f} km2")
    print(f"\nwrote -> {OUT}/yreb_counties_datav.gpkg")


if __name__ == "__main__":
    main()
