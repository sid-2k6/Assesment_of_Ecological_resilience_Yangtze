# Phase 3 — Data Acquisition

Status: **foundation complete and validated.** Study-area geometry, county graph, and a working remote-sensing extraction pipeline are in place. Indicator extraction across all years/variables is the next step.

## Headline result: no Google Earth Engine account needed

The original plan assumed GEE, which would have required your Google credentials. Probing showed **Microsoft Planetary Computer serves everything we need anonymously**, and pixel reads were verified working. This removes the main credential blocker.

| Requirement | Source | Verified |
|---|---|---|
| NDVI / kNDVI | MPC `modis-13Q1-061` | ✅ pixels read |
| Land surface temperature | MPC `modis-11A2-061` | ✅ pixels read |
| NPP | MPC `modis-17A3HGF-061` | ✅ pixels read |
| Evapotranspiration | MPC `modis-16A3GF-061` | ✅ catalogued |
| Surface reflectance (wetness) | MPC `modis-09A1-061` | ✅ catalogued |
| Landsat C2 L2 | MPC `landsat-c2-l2` | ✅ 6 low-cloud 2020 scenes over Wuhan |
| DEM → slope/elevation | MPC `cop-dem-glo-30` | ✅ pixels read |
| Climate | MPC `terraclimate`, `era5-pds` | ✅ catalogued |
| Surface water | MPC `jrc-gsw` | ✅ catalogued |
| County boundaries | DataV (official GB/T 2260 adcodes) | ✅ built |
| Land cover (30 m annual China) | Zenodo record `18180184` | ✅ located |

Still credential-gated (not on the critical path): NASA Earthdata, Copernicus CDS, OpenTopography, and the Chinese portals (RESDC / TPDC / Geospatial Data Cloud). Only needed if we add CLCD-specific or CMFD-specific layers.

## Study area — validated against the literature

County boundaries were **not** taken from GADM. GADM's ADM_3 aggregates Chinese urban districts (市辖区), yielding only 925 units for the belt — 13.6 % short. DataV carries official GB/T 2260 codes and matches China's real county-level division (县级行政区).

| Metric | Ours | Literature | Agreement |
|---|---|---|---|
| County-level units | **1,068** | 1,070 (Li et al. 2026) | −2 (0.19 %) |
| Total belt area | **2,052,264 km²** | ~2,050,000 km² | 0.1 % |
| GADM comparison | 925 | — | rejected |

Reach split: upstream 438 units / 1,128,276 km², midstream 325 / 564,904 km², downstream 305 / 359,084 km².

The 2-unit residual is administrative vintage — DataV reflects current divisions, while Li et al. harmonised to a 2000–2020 panel. Documented rather than forced.

Four province-directly-administered county-level units in Hubei (仙桃市, 潜江市, 天门市, 神农架林区) are tagged `level='city'` but have no children; the builder keeps them as county-level units rather than dropping them.

## County adjacency graph

Built to address the survey's primary methodological gap: spatial autocorrelation is *confirmed* in the resilience literature (Fu et al. 2026) but never *modelled*.

| Property | Value |
|---|---|
| Nodes / edges | 1,068 / 3,032 |
| Mean / median degree | 5.68 / 6 |
| Degree range | 1–12 |
| Components | **1 (fully connected)** |
| Cross-province edges | 287 |
| Cross-reach edges | 63 |

Mean degree 5.68 is the correct signature for a planar contiguity graph (theoretical mean approaches 6). A 50 m buffer absorbs digitisation slivers that would otherwise silently drop real neighbours. One genuine island isolate (岱山县, Zhoushan archipelago) is linked to its 3 nearest centroids.

## Extraction pipeline — benchmarked and validated

Decimating MODIS 250 m → 1 km gives an **11× speedup** (0.51 s vs 5.71 s per composite) while retaining ~1,738 pixels per county, which is statistically ample.

Full-belt annual NDVI for one year: **86 seconds**, all 1,068 counties populated, zero gaps. Extrapolates to roughly 30 min per variable for 2000–2020, or a few hours for the full indicator set.

### Validation against known geography

The 2020 NDVI extraction reproduces real geography without tuning:

**Lowest NDVI** — all dense urban cores: 黄浦区 Huangpu/Shanghai (0.168), 渝中区 Yuzhong/Chongqing peninsula (0.200), 江汉区 & 武昌区 Wuhan (0.214, 0.223), 吴中区 Suzhou (0.161).

**Highest NDVI** — all remote forest: 勐腊县 Mengla/Xishuangbanna rainforest (0.734), 景洪市 Jinghong (0.723), 资溪县 Zixi/Jiangxi forested mountains (0.696), 炎陵县 Yanling/Hunan (0.689).

**Provincial ordering**: Yunnan 0.583 (tropical forest) → Shanghai 0.315 (megacity). Correct.

**Reach ordering**: upstream 0.525 > midstream 0.510 > downstream 0.456.

### Analytically important observation

The NDVI reach gradient (**upstream > downstream**) is the *inverse* of the ecological resilience gradient reported throughout the literature (**downstream/east > upstream/west**). Vegetation greenness alone therefore cannot explain resilience — the published gradient must be driven by the adaptive-capacity and socioeconomic dimensions, not the vegetation dimension. This directly supports building a multi-dimensional rather than vegetation-dominated index, and is worth stating explicitly in the paper.

## Two preprocessing traps already identified

1. **MODIS NPP fill values leak.** `Npp_500m` returns 32767 (int16 fill) without declaring nodata, producing a spurious 3.2766 kg C/m² maximum. Must mask explicitly against the valid range, not rely on `nodatavals`.
2. **Terra/Aqua mixing.** MPC serves `MOD*` (Terra) and `MYD*` (Aqua) in the same collection — 384 items/year where 192 are expected. Filtering to Terra-only is required for a consistent series; the pipeline does this.

## Files

| Path | Contents |
|---|---|
| `boundaries/yreb_counties_datav.gpkg` | 1,068 county polygons, Albers equal-area, official adcodes |
| `tables/yreb_counties_datav.csv` | County attributes (adcode, name, province, reach, area) |
| `tables/adjacency_edges.csv` | 3,032 undirected edges with cross-reach/province flags |
| `tables/adjacency_stats.json` | Graph diagnostics |
| `tables/ndvi_county_2020.csv` | Validated 2020 county NDVI |
| `scripts/probe_sources.sh` | Endpoint reachability/credential probe |
| `scripts/test_mpc_read.py` | MPC anonymous pixel-read feasibility test |
| `scripts/01_build_yreb_boundary.py` | GADM extraction (superseded, kept for comparison) |
| `scripts/02_build_yreb_counties_datav.py` | **Authoritative** county boundary builder |
| `scripts/03_build_adjacency.py` | County graph construction |
| `scripts/04_benchmark_extraction.py` | Timing benchmark |
| `scripts/05_extract_ndvi_year.py` | Per-year county NDVI extraction |

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/02_build_yreb_counties_datav.py
.venv/bin/python scripts/03_build_adjacency.py
.venv/bin/python scripts/05_extract_ndvi_year.py 2020
```

Requires Python 3.12 (geopandas 1.x / numpy 2.x wheels). Albers equal-area CRS is used throughout — **not** WGS84 geographic — because area and landscape metrics are computed, and degree-based area across the belt's 21–35 °N span is materially distorted.

## Next steps

1. Extract remaining indicators for 2000–2020: kNDVI, LST (day/night), NPP, ET, wetness, precipitation, temperature, DEM/slope, surface water.
2. Add nighttime lights — **requires DMSP-OLS/VIIRS harmonisation**; naive concatenation creates a false 2013 discontinuity.
3. Acquire CLCD from Zenodo for landscape metrics (fragmentation, connectivity, patch density).
4. Add karst extent (region-adaptive novelty) and hydrological network (connectivity novelty). Note `hydrosheds.org` returned 403 — needs an alternative host.
5. Compile the county socioeconomic panel (yearbooks) — expect gaps pre-2005 and in western counties; imputation must be reported, since dropped rows would be systematically western and poor and would bias the east–west finding.
6. Build the resilience index once the label strategy is fixed.
