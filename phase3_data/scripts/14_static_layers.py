#!/usr/bin/env python3
"""Static / annual auxiliary layers -> county table (PERSIST N2, N3, Tier-1 human).

Produces
  1. KARST FRACTION per county          (N3 lithology gating)  <- WOKAM
  2. DIRECTED HYDROLOGICAL GRAPH        (N2 second topology)
  3. NIGHTTIME LIGHTS per county-year   (Tier-1 human stream)  <- harmonized DMSP/VIIRS

Hydrological graph note
-----------------------
hydrosheds.org returns 403 and HydroRIVERS is not mirrored on figshare, so the
directed graph is derived from terrain instead: for every spatially adjacent
county pair, the edge is directed from higher to lower mean elevation. This is a
first-order downhill-flow ordering built from data already in hand (DEM +
contiguity), it costs seconds, and it captures the upstream->downstream
asymmetry that N2 requires. It can be upgraded to true river topology later
without changing the model interface.

Outputs
  data/interim/karst_county.csv
  data/interim/hydro_edges.csv
  data/interim/ntl_county_year.csv
"""
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds

OUT = pathlib.Path("data/interim")
CTY = OUT / "yreb_counties_datav.gpkg"
WOKAM = pathlib.Path("data/raw/static/wokam/WHYMAP_WOKAM/shp/whymap_karst__v1_poly.shp")
NTL_DIR = pathlib.Path("data/raw/ntl")


# ---------------------------------------------------------------- 1. karst
def karst(gdf):
    print("=" * 66, "\n1. KARST FRACTION (WOKAM)")
    k = gpd.read_file(WOKAM)
    print(f"   WOKAM polygons: {len(k)}  crs={k.crs}")
    if "ROCK_TYPE" in k.columns:
        print(f"   classes: {k.ROCK_TYPE.value_counts().to_dict()}")
    k = k.to_crs(gdf.crs)

    # clip to belt then intersect per county
    belt = gdf.union_all()
    k = k[k.intersects(belt)].copy()
    print(f"   polygons intersecting the YREB: {len(k)}")

    inter = gpd.overlay(gdf[["adcode", "area_km2", "geometry"]],
                        k[["geometry"]], how="intersection")
    inter["karst_km2"] = inter.geometry.area / 1e6
    agg = inter.groupby("adcode", as_index=False).karst_km2.sum()

    out = gdf[["adcode", "name_zh", "province", "reach", "area_km2"]].merge(
        agg, on="adcode", how="left")
    out["karst_km2"] = out.karst_km2.fillna(0.0)
    out["karst_frac"] = (out.karst_km2 / out.area_km2).clip(0, 1)
    out.to_csv(OUT / "karst_county.csv", index=False)

    print(f"   counties with any karst: {(out.karst_frac>0.01).sum()}/{len(out)}")
    print(f"   mean karst fraction by reach:")
    print(out.groupby("reach").karst_frac.mean().round(3).to_string()
          .replace("\n", "\n     ").rjust(0))
    print("   top karst counties (expect Guizhou / Yunnan / Guangxi margin):")
    print(out.nlargest(6, "karst_frac")[
        ["name_zh", "province", "karst_frac"]].round(3).to_string(index=False))
    return out


# ------------------------------------------------- 2. directed hydro graph
def hydro_graph(gdf):
    print("\n" + "=" * 66, "\n2. DIRECTED HYDROLOGICAL GRAPH (terrain-derived)")
    edges = pd.read_csv(OUT / "adjacency_edges.csv",
                        dtype={"u_adcode": str, "v_adcode": str})
    terr = pd.read_csv(OUT / "terrain_county.csv", dtype={"adcode": str})
    el = terr.set_index("adcode").elev_mean.to_dict()

    rows = []
    for r in edges.itertuples(index=False):
        a, b = r.u_adcode, r.v_adcode
        ea, eb = el.get(a), el.get(b)
        if ea is None or eb is None or not (np.isfinite(ea) and np.isfinite(eb)):
            continue
        # direct the edge downhill: source = higher, target = lower
        src, dst = (a, b) if ea >= eb else (b, a)
        rows.append(dict(src=src, dst=dst,
                         drop_m=abs(ea - eb),
                         src_elev=max(ea, eb), dst_elev=min(ea, eb),
                         cross_reach=r.cross_reach,
                         cross_province=r.cross_province))
    h = pd.DataFrame(rows)
    h.to_csv(OUT / "hydro_edges.csv", index=False)

    outdeg = h.groupby("src").size()
    indeg = h.groupby("dst").size()
    print(f"   directed edges: {len(h)} (from {len(edges)} undirected)")
    print(f"   mean out-degree {outdeg.mean():.2f} | mean in-degree {indeg.mean():.2f}")
    print(f"   elevation drop across edge: median {h.drop_m.median():.0f} m, "
          f"max {h.drop_m.max():.0f} m")
    print(f"   headwater counties (no upstream neighbour): "
          f"{len(set(h.src)-set(h.dst))}")
    print(f"   outlet counties (no downstream neighbour):  "
          f"{len(set(h.dst)-set(h.src))}")
    # sanity: net flow should run upstream -> downstream reach
    cr = h[h.cross_reach]
    print(f"   cross-reach edges: {len(cr)}")
    return h


# ------------------------------------------------------------------ 3. NTL
def ntl(gdf):
    print("\n" + "=" * 66, "\n3. NIGHTTIME LIGHTS (harmonized DMSP/VIIRS)")
    files = sorted(NTL_DIR.glob("ntl_*.tif"))
    print(f"   files: {len(files)}")
    if not files:
        return None

    recs = []
    labels = inside = lab_in = None
    for f in files:
        year = int(f.stem.split("_")[1])
        with rasterio.open(f) as src:
            if labels is None:
                b = transform_bounds(gdf.crs, src.crs, *gdf.total_bounds)
                win = src.window(*b)
                win = win.round_offsets().round_lengths()
                tr = src.window_transform(win)
                shp = (int(win.height), int(win.width))
                proj = gdf.to_crs(src.crs)
                labels = rasterize(
                    ((g, i + 1) for i, g in enumerate(proj.geometry)),
                    out_shape=shp, transform=tr, fill=0, dtype="int32",
                    all_touched=True)
                inside = labels > 0
                lab_in = labels[inside]
                print(f"   window {shp}  crs={src.crs}  "
                      f"counties hit {len(np.unique(lab_in))}")
                keep_win = win
            arr = src.read(1, window=keep_win).astype("float64")
        a = arr[inside]
        ok = np.isfinite(a) & (a >= 0)
        n = len(gdf)
        c = np.bincount(lab_in[ok], minlength=n + 1)
        s = np.bincount(lab_in[ok], weights=a[ok], minlength=n + 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        recs.append(pd.DataFrame({"adcode": gdf.adcode.values, "year": year,
                                  "ntl_mean": mean[1:], "ntl_sum": s[1:],
                                  "ntl_n": c[1:]}))
    d = pd.concat(recs, ignore_index=True).sort_values(["adcode", "year"])
    d.to_csv(OUT / "ntl_county_year.csv", index=False)

    print(f"   rows {len(d):,}  years {d.year.min()}-{d.year.max()}")
    yr = d.groupby("year").ntl_mean.mean()
    print("\n   mean NTL by year (checking for a 2013/2014 DMSP->VIIRS break):")
    for y, v in yr.items():
        bar = "#" * int(v * 3)
        flag = "  <-- transition" if y in (2013, 2014) else ""
        print(f"     {y} {v:7.2f} {bar}{flag}")
    if 2013 in yr.index and 2014 in yr.index:
        pre = yr.loc[max(2010, yr.index.min()):2013].mean()
        post = yr.loc[2014:min(2017, yr.index.max())].mean()
        jump = (post - pre) / pre * 100 if pre else np.nan
        print(f"\n   pre-2014 mean {pre:.2f} vs post-2014 mean {post:.2f} "
              f"-> {jump:+.1f}%")
        print("   (a large abrupt jump would indicate the harmonisation failed)")
    return d


def main():
    gdf = gpd.read_file(CTY).reset_index(drop=True)
    gdf["adcode"] = gdf.adcode.astype(str)
    print(f"counties {len(gdf)}\n")
    karst(gdf)
    hydro_graph(gdf)
    ntl(gdf)
    print("\nDONE")


if __name__ == "__main__":
    main()
