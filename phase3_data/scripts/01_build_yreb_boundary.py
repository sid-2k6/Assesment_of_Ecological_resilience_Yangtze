#!/usr/bin/env python3
"""Extract the Yangtze River Economic Belt (11 provinces) county-level
boundaries from GADM 4.1, reproject to an equal-area CRS, and report coverage
against the 1,070-county figure used in the literature (Li et al. 2026).

Outputs:
  data/interim/yreb_counties.gpkg   county polygons (ADM_3), Albers equal-area
  data/interim/yreb_provinces.gpkg  province polygons (ADM_1)
  data/interim/yreb_mask.gpkg       single dissolved belt outline
  data/interim/yreb_counties.csv    attribute table (no geometry)
"""
import pathlib

import geopandas as gpd
import pandas as pd

GADM = "data/raw/boundaries/gadm41_CHN.gpkg"
OUT = pathlib.Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)

# The 11 provincial-level units of the YREB, as GADM NAME_1 spellings.
YREB = ["Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Jiangxi", "Hubei",
        "Hunan", "Chongqing", "Sichuan", "Yunnan", "Guizhou"]

# China Albers Equal Area — correct for area/landscape-metric computation.
# (Degrees-based area across a 24-35 N span is materially distorted.)
ALBERS = ("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
          "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

REACH = {  # upstream / midstream / downstream, per the standard YREB division
    "Sichuan": "upstream", "Yunnan": "upstream", "Guizhou": "upstream",
    "Chongqing": "upstream",
    "Hubei": "midstream", "Hunan": "midstream", "Jiangxi": "midstream",
    "Anhui": "downstream", "Jiangsu": "downstream", "Zhejiang": "downstream",
    "Shanghai": "downstream",
}


def main():
    print("reading GADM ADM_3 ...")
    adm3 = gpd.read_file(GADM, layer="ADM_ADM_3")
    print(f"  national county-level units: {len(adm3)}")

    cty = adm3[adm3.NAME_1.isin(YREB)].copy()
    missing = set(YREB) - set(cty.NAME_1.unique())
    if missing:
        raise SystemExit(f"province name mismatch: {missing}")

    cty["reach"] = cty.NAME_1.map(REACH)
    cty = cty.to_crs(ALBERS)
    cty["area_km2"] = cty.geometry.area / 1e6

    # Stable county identifier. GID_3 is GADM's own key and is unique.
    cty["county_id"] = cty.GID_3
    assert cty.county_id.is_unique, "GID_3 not unique"

    keep = ["county_id", "GID_1", "NAME_1", "GID_2", "NAME_2", "NAME_3",
            "NL_NAME_3", "TYPE_3", "ENGTYPE_3", "reach", "area_km2", "geometry"]
    cty = cty[keep].rename(columns={
        "NAME_1": "province", "NAME_2": "prefecture", "NAME_3": "county",
        "NL_NAME_3": "county_zh", "TYPE_3": "type_zh", "ENGTYPE_3": "type_en"})

    prov = (cty.dissolve(by="province", as_index=False)
               [["province", "reach", "geometry"]])
    prov["area_km2"] = prov.geometry.area / 1e6

    mask = cty.dissolve()[["geometry"]]
    mask["area_km2"] = mask.geometry.area / 1e6

    cty.to_file(OUT / "yreb_counties.gpkg", driver="GPKG")
    prov.to_file(OUT / "yreb_provinces.gpkg", driver="GPKG")
    mask.to_file(OUT / "yreb_mask.gpkg", driver="GPKG")
    cty.drop(columns="geometry").to_csv(OUT / "yreb_counties.csv", index=False)

    # ---- report -------------------------------------------------------
    print(f"\nYREB counties extracted: {len(cty)}")
    print(f"literature reference    : 1,070 (Li et al. 2026)")
    print(f"difference              : {len(cty) - 1070:+d}")
    print(f"\ntotal belt area: {mask.area_km2.iloc[0]:,.0f} km2 "
          f"(published figure ~2,050,000 km2)")

    print("\nby province:")
    t = (cty.groupby(["reach", "province"])
            .agg(counties=("county_id", "size"), area_km2=("area_km2", "sum"))
            .reset_index().sort_values(["reach", "counties"], ascending=[True, False]))
    for _, r in t.iterrows():
        print(f"  {r['reach']:11} {r['province']:11} {r['counties']:>4} counties "
              f"{r['area_km2']:>12,.0f} km2")

    print("\nby reach:")
    for reach, grp in cty.groupby("reach"):
        print(f"  {reach:11} {len(grp):>4} counties "
              f"{grp.area_km2.sum():>12,.0f} km2")

    b = cty.to_crs(4326).total_bounds
    print(f"\nbbox (WGS84): lon {b[0]:.2f} to {b[2]:.2f}, "
          f"lat {b[1]:.2f} to {b[3]:.2f}")
    print(f"county area: min {cty.area_km2.min():.1f} / "
          f"median {cty.area_km2.median():.0f} / max {cty.area_km2.max():,.0f} km2")
    print(f"\nwrote -> {OUT}/yreb_counties.gpkg (+ provinces, mask, csv)")


if __name__ == "__main__":
    main()
