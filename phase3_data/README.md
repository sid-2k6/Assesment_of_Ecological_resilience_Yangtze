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


---

# Phase 3b — Indicator extraction complete (2000–2020)

## The panel

**22,428 rows × 45 columns — 1,068 counties × 21 years.** Built entirely from Planetary Computer, no credentials.

| Indicator | Product | Native | Years | Missing |
|---|---|---|---|---|
| NDVI, EVI (mean/std/max) | `modis-13Q1-061` | 250 m → 1 km | 2000–2020 | 0.00 % |
| kNDVI (derived) | `tanh(NDVI²)` | — | 2000–2020 | 0.00 % |
| LST day, night, diurnal range | `modis-11A2-061` | 1 km | 2000–2020 | 0.00 % |
| NPP | `modis-17A3HGF-061` | 500 m | **2001**–2020 | 5.12 % |
| ET | `modis-16A3GF-061` | 500 m | 2000–2020 | 0.37 % |
| Elevation, relief, slope, roughness | `cop-dem-glo-90` | 90 m | static | 0.00 % |

All missingness is explained, none is unexplained: NPP's 5.12 % is the absent 2000 (4.76 %) plus 4 urban counties; ET's 0.37 % is those same 4 counties × 21 years.

**kNDVI is used in preference to NDVI** (Camps-Valls et al. 2021 formulation, which reduces to `tanh(NDVI²)`), because it avoids saturation in dense vegetation and was validated for the YREB by Zhu et al. (2025).

## External validation against published YREB figures

Benchmarked against **Zhu et al. 2025, *Land* 14(3):598** — the most-cited paper in our survey (37 citations), full-belt, multi-method:

| Metric | Ours | Published | Agreement |
|---|---|---|---|
| kNDVI trend | **+0.00206/yr** (p<0.0001, R²=0.62) | +0.003/yr (p<0.05) | same sign, 69 % of magnitude |
| LST day trend | **+0.0322 °C/yr** (p=0.033, R²=0.20) | +0.065 °C/yr (p<0.01) | same sign, ~50 % of magnitude |

Both trends reproduce the published **sign and statistical significance**. Magnitudes are within a factor of two but not identical — stated plainly rather than overclaimed. Likely causes: differing kNDVI σ parameterisation, county-area aggregation versus pixel-level regression, and annual-mean versus growing-season compositing. Additional trends: NPP +0.0034/yr (p=0.0002), ET +5.35 mm/yr (p<0.0001, R²=0.71), NDVI +0.00218/yr.

2000 is **excluded from all trend fits** for 8- and 16-day products: Terra launched Feb 2000, so the 2000 annual mean omits the coldest weeks and is upward-biased (LST 2000 = 296.2 K, the highest in the series, for this reason alone).

## Reach gradients (2001–2020 means)

| Reach | kNDVI | LST day °C | NPP | ET mm | Elev m | Slope ° |
|---|---|---|---|---|---|---|
| downstream | 0.205 | 21.17 | 0.520 | 686 | 98 | 5.0 |
| midstream | 0.259 | 22.41 | 0.583 | 800 | 271 | 9.1 |
| upstream | 0.260 | 21.61 | 0.735 | 744 | 1,400 | 15.8 |

Terrain validation is exact: highest counties are 石渠县 (4,497 m), 理塘县 (4,335 m), 德格县, 巴塘县 — all Garzê Tibetan Prefecture, western Sichuan. Lowest are 射阳县 (1.7 m), 崇明区 (2.0 m, Chongming Island at the Yangtze mouth), all Jiangsu delta.

**Vegetation and productivity run upstream > downstream — the inverse of the published resilience gradient (downstream/east > upstream/west).** Now confirmed across four independent indicators (kNDVI, NPP, ET, and NDVI), not one. Ecological resilience in the YREB is therefore *not* vegetation-driven; the published gradient must be dominated by adaptive-capacity and socioeconomic terms. This is a substantive argument for a multi-dimensional index and belongs in the paper.

## Disturbance-response check (validates the label strategy)

Mean z-score anomaly against each county's own 2001–2020 baseline:

| Year | Event | kNDVI z | NPP z | ET z |
|---|---|---|---|---|
| 2006 | SW China drought | **−0.44** | **−0.57** | **−0.64** |
| 2011 | Yangtze drought | −0.03 | **−0.30** | **−0.45** |
| 2013 | Record heatwave | +0.61 | +0.21 | +0.16 |
| 2016 | Yangtze floods | +0.83 | +0.10 | +1.19 |
| 2020 | Yangtze floods | +0.41 | +0.61 | +0.90 |

Droughts produce clear negative anomalies and floods positive ones, so **the data does respond to known disturbance events** — the disturbance-validation label strategy is viable.

### Important caveat discovered

2013 was a *record* Yangtze heatwave/drought, yet kNDVI shows **+0.61**, not a dip. Inspecting midstream counties, 2013 LST day reaches **23.18 °C — the highest of 2010–2016** — so the thermal signal is captured correctly while the vegetation drought response is diluted by annual averaging.

**Consequence for Phase 5:** annual aggregation is adequate for *state* indicators but insufficient for *recovery* dynamics. Resistance/recovery estimation needs seasonal or monthly compositing, or must lean on thermal indicators that survive annual aggregation. This changes the extraction requirement and is better known now than after model training.

## Three data-integrity bugs found and fixed

1. **Duplicate reprocessed granules (most serious).** MPC serves multiple processing generations of the same granule, distinguished only by a trailing production timestamp:
   ```
   MOD16A3GF.A2014001.h29v06.061.2022077135259   <- keep
   MOD16A3GF.A2014001.h29v06.061.2021343120212   <- discard
   MOD17A3HGF.A2013001.h29v06.061  ->  THREE versions
   ```
   Summing all versions inflates pixel counts 2–3×. Fixed by keeping the latest timestamp per (product, acquisition date, tile). Verified the *means* were unaffected (identical to 7 s.f., so duplicates were identical rasters) but counts were not — and had the reprocessings differed, this would have manufactured a false step change around 2013–2014 that could have been written up as an ecological regime shift.

2. **`LabelCache` key collision.** Keyed on `(tile, decim)`, but NPP/ET are 500 m (2400×2400) and LST is 1 km (1200×1200) with `decim=1` for both — so running LST after NPP would silently return the wrong label raster. Key now includes `out_shape`.

3. **MODIS fill values leak.** `Npp_500m` returns 32767 without declaring nodata, giving a spurious 2.74 kgC/m²/yr mean. Masked against documented valid ranges; `src.nodatavals` is deliberately not trusted. Post-fix NPP is 0.59–0.72 kgC/m²/yr, correct for subtropical China.

## Outstanding QC item

Small, fully urbanised counties have very few valid pixels for 500 m products — 黄浦区 (20.4 km²) has **only 4 valid ET pixels**, making its ET value statistically unreliable. A minimum-pixel threshold (`*_n`) must be applied before modelling; the `*_n` columns are retained in the panel for exactly this purpose. Because affected counties are systematically the densest urban cores, dropping them without care would bias the east–west comparison.

## Files added

| Path | Contents |
|---|---|
| `tables/panel_county_year.csv` | **Main deliverable** — 22,428 × 45 county-year panel |
| `tables/terrain_county.csv` | Static terrain per county |
| `tables/panel_validation.json` | Trend fits, reach gradients, machine-readable |
| `scripts/06_extract_indicators.py` | Multi-product extractor (dedup, fill masking, threaded, resumable) |
| `scripts/07_extract_terrain.py` | DEM → elevation/relief/slope/roughness |
| `scripts/08_assemble_panel.py` | Panel assembly + external validation |

Per-year indicator CSVs (`data/interim/indicators/`, 9.1 MB) are reproducible from the scripts and not committed.

## Next

1. **Seasonal/monthly compositing** for recovery dynamics (raised to priority by the 2013 finding).
2. Nighttime lights with DMSP/VIIRS harmonisation.
3. CLCD land cover → landscape fragmentation/connectivity metrics.
4. Karst extent + hydrological network (`hydrosheds.org` returns 403 — needs another host).
5. Socioeconomic panel from county yearbooks.
6. Changepoint test on the ET series — the +5.35 mm/yr trend has a step-like character near 2014 that survived deduplication and needs explaining before use.
