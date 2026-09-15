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


---

# Phase 3c — Seasonal RS compositing + climate forcing stream (PERSIST N1, N5)

## The seasonal panel

**89,712 rows × 59 columns — 1,068 counties × 21 years × 4 seasons.** A 4× increase in temporal samples over the annual panel, which also materially relieves the sample-size risk flagged in the PERSIST design (§7): 22,428 → 89,712.

Seasons use the standard climatological convention where December belongs to the **following** year's winter: `DJF(Y) = Dec(Y-1), Jan(Y), Feb(Y)`. The search window per target year is therefore `(Y-1)-12-01 .. Y-11-30`.

| Stream | Variables | Source |
|---|---|---|
| **State** (seasonal) | kNDVI, NDVI, EVI, LST day/night/range — mean, std, max, n | MODIS 13Q1, 11A2 |
| **Forcing** (seasonal) | ppt, pet, aet, def, q *(season sums)*; tmax, tmin, vpd, soil, srad, **pdsi** *(season means)* | TerraClimate |
| **Derived forcing** | `*_z` anomalies vs county×season climatology, `wbal = ppt−pet`, `spei_like`, `dry_z`, `heat_z` | computed |

Seasonality validates cleanly: NDVI runs **DJF 0.376 < MAM 0.449 < SON 0.471 < JJA 0.597**, and precipitation **JJA 527 mm >> MAM 308 > SON 238 > DJF 112** — the East Asian monsoon. Annual precipitation by reach is midstream 1,372 mm > downstream 1,161 > upstream 1,068, correctly placing the Poyang/Dongting lake region as wettest.

## Forcing stream independently identifies known disturbances

This is what N1 requires, and it works:

| Year | Event | dry_z | heat_z | PDSI |
|---|---|---|---|---|
| 2006 | Chongqing/Sichuan drought | +0.19 | +0.54 | **−1.93** |
| 2011 | Yangtze drought | **+0.61** | −0.25 | **−2.76** |
| 2013 | Record heatwave | +0.34 | +0.59 | −1.79 |
| 2020 | Floods | −0.11 | +0.54 | −1.58 |

**Spatial targeting is exact.** For JJA 2006 the highest `dry_z` values are **Chongqing 2.18 and Sichuan 2.06** — precisely the provinces hit by the catastrophic 2006 drought, Chongqing's worst in a century. For JJA 2013 `heat_z` reaches **1.89 downstream / 1.68 midstream / 1.70 upstream**, matching a middle-lower-Yangtze-centred heatwave.

Critically, this **closes the loop on finding F2**: annual kNDVI showed *+0.61* in 2013 (no drought signal at all), while the seasonal forcing stream shows heat_z ≈ +1.7 to +1.9. The forcing stream sees the disturbance the annual state stream missed — exactly the two-stream separation N1 depends on.

## N1 feasibility test — mixed result, honestly reported

The cheapest possible check before implementing N1: **does forcing actually explain state anomalies?**

| Forcing → State | r (JJA) | r (all) |
|---|---|---|
| heat_z → LST day | **+0.524** | **+0.577** |
| dry_z → LST day | +0.323 | +0.365 |
| dry_z → kNDVI | +0.205 | +0.083 |
| heat_z → kNDVI | +0.191 | +0.197 |
| pdsi → kNDVI | +0.024 | +0.098 |
| soil_z → kNDVI | −0.091 | +0.026 |

Within-county (county-demeaned) correlations are essentially identical, so these are not cross-sectional artefacts.

**Verdict: the thermal channel is strongly identifiable (r ≈ 0.52); the vegetation channel is weak (r ≈ 0.19–0.21).**

### Two problems this exposed

**Problem A — the drought→vegetation sign is inverted.** `dry_z → kNDVI = +0.205` means *drier → greener*, which looks ecologically backwards until the supporting correlations are checked:

| | r with kNDVI anomaly (JJA) |
|---|---|
| precipitation | **−0.145** |
| solar radiation | **+0.145** |
| dry_z | +0.205 |
| soil moisture | −0.012 |

More rain → less green; more radiation → more green; soil moisture irrelevant. **YREB summer vegetation is light-limited, not water-limited** — expected in a humid monsoon basin receiving 1,000–1,400 mm, and partly reinforced by MODIS cloud contamination during rainy periods.

**Consequence for PERSIST: drought is the wrong primary disturbance axis for vegetation in this basin.** N1's forcing definition must shift toward extreme heat, flooding/waterlogging, and cold/frost events rather than moisture deficit. The thermal coupling (r = 0.52) is where the identifiable signal lives.

**Problem B — anomaly persistence is near zero, threatening N1's recovery term.**

| Series | corr(JJA, SON) |
|---|---|
| **raw** kNDVI | +0.462 |
| **z-scored** kNDVI | **+0.020** |
| **raw** LST day | +0.774 |
| **z-scored** LST day | +0.133 |

Year-over-year, same season: kNDVI `corr(t, t−1) = −0.015`, LST `+0.090`.

Raw series persist strongly, but that persistence is **cross-sectional** (green counties stay green). Once county×season means are removed, residual interannual anomalies carry almost no seasonal carry-over. Physically: humid subtropical vegetation recovers within weeks, so at seasonal resolution recovery is already complete before the next observation.

**Consequence: N1's recovery term is probably not identifiable at seasonal resolution.** Options in order of preference:
1. Move to **monthly or native 16-day** resolution for the response channel — recovery happens *inside* a season
2. Use **LST as the primary response channel**, where coupling and persistence are both far stronger
3. Estimate recovery on **physically integrated** annual variables (NPP, ET) rather than instantaneous greenness

This is precisely what the feasibility test was for. Discovering it now cost one script; discovering it after implementing N1 would have cost the phase.

## Bug fixed — sub-cell counties silently dropped

The first climate run returned 1,042 of 1,068 counties. `rasterize()` assigns each cell to exactly **one** polygon, so a small county sharing a TerraClimate cell (~4.6 km, ~21 km²) with a larger neighbour is overwritten and receives zero cells — even with `all_touched=True`.

All 26 casualties were tiny urban districts: median **62 km²** against an overall median of 1,586 km² (黄浦区 20 km², 虹口区 23 km², 渝中区 24 km², 江汉区 29 km²…).

**These are the same dense urban cores that already lose MOD16 ET.** Urban-core counties are therefore losing data across *multiple independent products*, extending finding F3: dropping them would systematically remove the most-developed downstream units and bias the east–west comparison the study exists to test.

Fixed with a **centroid-fallback estimator** — sample the nearest grid cell at the county centroid. Climate fields are spatially smooth at 4.6 km, so this is adequate for a 20 km² district. Fallback rows are flagged `n_cells = 0.5` so they remain filterable.

## Also fixed — TerraClimate Zarr access

The STAC `zarr-https` asset carries its SAS token as a query string, and fsspec appends `/zarr.json` *after* the query, producing a malformed URL and a 403. The `zarr-abfs` asset passes credentials via `storage_options` and works. Separately, `xr.open_zarr()` rejects the `engine` kwarg that STAC advertises — route through `xr.open_dataset()`.

## Files added

| Path | Contents |
|---|---|
| `tables/panel_seasonal.parquet` | **Main deliverable** — 89,712 × 59 seasonal panel (state + forcing) |
| `tables/climate_forcing_seasonal.parquet` | 89,712 × 30 forcing stream alone |
| `scripts/09_extract_seasonal_rs.py` | Seasonal MODIS compositing |
| `scripts/10_extract_climate_forcing.py` | TerraClimate → seasonal county forcing |
| `scripts/11_assemble_seasonal_panel.py` | Panel assembly + N1 feasibility test |

Stored as Parquet with float32 downcasting: 83 MB CSV → **22.5 MB Parquet**. Per-year seasonal CSVs (25 MB) are reproducible and not committed.

## Impact on the PERSIST design

| Novelty | Status |
|---|---|
| **N5** hierarchical seasonal encoder | ✅ **Data ready.** Justification now doubly confirmed. |
| **N1** disturbance-response decoder | ⚠️ **Needs revision.** Forcing stream built and validated, but the disturbance axis must shift from drought to thermal/flood, and the recovery term needs sub-seasonal resolution. |
| N2, N3, N4, N6 | Unaffected; still awaiting river network, karst extent, CLCD. |


---

# Phase 3d — Monthly panel, karst, hydro graph, nighttime lights

Data collection push following the N1 revision. **The monthly panel is now the primary modelling input.**

## Primary deliverable: monthly panel

**269,136 rows × 58 columns — 1,068 counties × 21 years × 12 months.** Exactly the expected count, no gaps.

| Stream | Contents | Missing |
|---|---|---|
| **State** (monthly) | LST day, NDVI, kNDVI + anomalies | 0.40–0.48 % |
| **Forcing** (monthly) | ppt, pet, aet, def, q, tmax, tmin, vpd, soil, srad, pdsi, swe + `*_z` anomalies, `wbal`, `heat_z`, `dry_z` | 0.00 % |
| **Static** | elevation, relief, slope, roughness, **karst_frac** | 0.00 % |
| **Human (Tier 1)** | harmonised nighttime lights (mean, sum) | 0.00 % |

Sample size progression across the project: **22,428 annual → 89,712 seasonal → 269,136 monthly.** The sample-size risk in the design's §7 is now retired.

Monthly forcing climatology validates: precipitation peaks June at 190.8 mm, tmax peaks July at 30.1 °C — the East Asian monsoon. All four known drought years show positive `dry_z`.

## A result that confirms finding F1 directly

| Reach | NTL (human) | kNDVI (vegetation) | Karst |
|---|---|---|---|
| downstream | **19.00** | 0.205 | 0.141 |
| midstream | 7.45 | 0.259 | 0.364 |
| upstream | 4.75 | 0.260 | 0.584 |

**Nighttime lights follow the same gradient as the published resilience gradient (downstream > upstream); vegetation follows the inverse.** F1 predicted resilience must be driven by adaptive-capacity/socioeconomic terms rather than vegetation, and the human stream now demonstrably has the right gradient direction while the vegetation stream does not. This belongs in the paper as direct empirical support for a multi-dimensional index.

## Karst extent (N3 gating) — acquired

Source: **WOKAM** (World Karst Aquifer Map, BGR/WHYMAP), `whymap_karst__v1_poly.shp`, intersected per county in Albers equal-area.

Mean karst fraction: **upstream 0.584, midstream 0.364, downstream 0.141** — correctly concentrating karst in the southwest, which is exactly the heterogeneity N3 exists to handle. 696 of 1,068 counties have >1 % karst.

⚠️ **Caveat to resolve before use:** WOKAM is a coarse global map, so small counties can fall entirely inside a single polygon and register `karst_frac = 1.0`. Several top-ranked counties are genuine karst (恩施市, 建始县, 长阳, 秭归县 — the Wuling/Three Gorges karst belt), but a few small urban districts (杜集区 Anhui, 安源区 Jiangxi) are almost certainly resolution artefacts. The `rock_type` / `RTypeLabel` attributes need filtering to separate continuous from discontinuous carbonate before N3 uses this layer.

## Directed hydrological graph (N2) — built from terrain

`hydrosheds.org` returns **403** and HydroRIVERS is not mirrored on figshare, so the second topology is derived from data already in hand: for each spatially adjacent county pair, the edge is directed **from higher to lower mean elevation**.

| Property | Value |
|---|---|
| Directed edges | 3,032 |
| Median elevation drop | 63 m (max 2,067 m) |
| Headwater counties (no upstream neighbour) | 68 |
| Outlet counties (no downstream neighbour) | 100 |
| Cross-reach edges | 63 |

Because elevation decreases strictly along every edge, the result is **acyclic by construction** — a desirable property for directed message passing.

⚠️ This is a first-order downhill-flow proxy, **not** true river topology. It captures the upstream→downstream asymmetry N2 requires and can be swapped for real river network data later without changing the model interface. It must be described honestly as a terrain-derived proxy in the paper.

## Nighttime lights — a silent product-mixing bug, caught

Source: harmonised DMSP/VIIRS 1992–2024 (figshare 9828827), 21 annual GeoTIFFs for 2000–2020.

**The bug:** my filename matcher selected `DN_NTL_2013_simVIIRS.tif` instead of `Harmonized_DN_NTL_2013_calDMSP.tif`, because alphabetical ordering puts `D…` before `H…`. The chosen file **lacks the `Harmonized_` prefix** and is therefore on a different radiometric scale. It also consumed the slot that should have held 2020.

Effect: county-mean NTL for 2013 read **12.932**, making 2014 the largest year-over-year jump in the entire series (−2.673 against a median absolute jump of 0.591) — a textbook false DMSP→VIIRS discontinuity, precisely the artefact the design warned about.

**After correction** (2013 = `Harmonized_..._calDMSP`, 2020 added):

| | Before | After |
|---|---|---|
| 2013 value | 12.932 | **11.600** |
| 2013→2014 jump | −2.673 | **−1.341** |
| Largest jump year | **2014** | 2017 (1.889) |
| Median abs jump | 0.591 | 0.591 |

The transition is no longer the largest discontinuity in the series. **Verdict: harmonisation is sound and the series is usable.** A mild residual elevation remains at 2013 (final DMSP year, F18 sensor degradation) and should be noted but is not disqualifying.

Had this gone unnoticed, the fake 2014 break would have propagated into every model using the human stream.

## Files added

| Path | Contents |
|---|---|
| `tables/panel_monthly.parquet` | **Primary modelling input** — 269,136 × 58 |
| `tables/karst_county.csv` | Per-county karst fraction (N3) |
| `tables/hydro_edges.csv` | Directed hydrological graph, 3,032 edges (N2) |
| `tables/ntl_county_year.csv` | Harmonised nighttime lights, 2000–2020 |
| `scripts/12_extract_monthly_state.py` | Monthly MODIS state extraction |
| `scripts/13_test_monthly_irf.py` | N1 impulse-response feasibility test |
| `scripts/14_static_layers.py` | Karst, hydro graph, NTL → county |
| `scripts/15_monthly_forcing_full.py` | Monthly TerraClimate forcing, full period |
| `scripts/16_assemble_monthly_panel.py` | Monthly panel assembly |

## Dataset status

| Stream | Status |
|---|---|
| County boundaries + spatial graph | ✅ complete |
| Monthly state (LST, NDVI) 2000–2020 | ✅ complete |
| Monthly forcing 2000–2020 | ✅ complete |
| Seasonal panel | ✅ complete |
| Annual panel | ✅ complete |
| Static terrain | ✅ complete |
| **Karst extent (N3)** | ✅ acquired, ⚠️ needs class filtering |
| **Hydro graph (N2)** | ✅ built, ⚠️ terrain proxy not true rivers |
| **Nighttime lights (Tier 1)** | ✅ complete and verified |
| Impervious surface / population / GDP | ❌ not started |
| CLCD land cover + fragmentation | ❌ not started |
| Disturbance event catalogue | ❌ not started (partly derivable from forcing) |
| Tier 2 socioeconomic | ❌ optional, needs institutional access |

**Outstanding QC:** LST clear-sky sampling bias unquantified (highest priority — it sits under the revised N1's primary channel); ET changepoint near 2014 unexplained; minimum-pixel thresholds not yet applied; no Moran's I / VIF diagnostics; train/test splits not constructed.

---

# Phase 3e — QC: LST clear-sky sampling bias

LST is N1's primary response channel and MOD11A2 retrieves it **only under clear sky**. The specific confound to rule out was:

> drought → clear skies → MORE observations AND higher measured LST

If that drove the observed `dry_z → LST` coupling, N1's "response" would be a sampling artefact rather than a land-surface process.

## Test 1 — clear-sky sampling is confirmed, but modest

| | r with relative sampling density |
|---|---|
| precipitation | **−0.178** |
| solar radiation | **+0.264** |
| VPD | +0.236 |

Sampling density is genuinely driven by cloudiness. Zero-observation county-months are only 0.48 %.

## Test 2 — there IS leakage into the anomaly

`corr(n_rel, lst_c_z) = +0.204`. Sparse months read cooler, dense months warmer:

| Sampling quintile | LST anomaly | precipitation (mm) |
|---|---|---|
| Q1 sparse | **−0.316** | 128.6 |
| Q2 | −0.048 | 106.5 |
| Q3 | +0.032 | 96.4 |
| Q4 | +0.101 | 78.0 |
| Q5 dense | **+0.224** | 84.7 |

A monotonic 0.54 σ spread across quintiles, with precipitation falling across them — the mechanism is exactly as hypothesised.

## Test 3 — the coupling survives the control

| | raw r | partial r (controlling n_rel) | attenuation |
|---|---|---|---|
| all months | +0.358 | +0.322 | 10.0 % |
| JJA only | +0.360 | +0.311 | **13.8 %** |

**~86 % of the `dry_z → LST` coupling is not explained by sampling.** The land-surface response is real.

## Test 4 — filtering to well-sampled months *improves* the signal

| Subset | t+0 | t+1 | t+2 | decay t+1/t0 |
|---|---|---|---|---|
| all months | +1.004 | +0.135 | +0.199 | 0.135 |
| **n_rel ≥ 0.9** | +1.064 | **+0.454** | +0.216 | **0.427** |

Restricting to well-sampled months **strengthens** the decay (0.135 → 0.427) while retaining 3,527 of 4,237 shocks (83 %). The sparse months were adding noise, not manufacturing the signal.

## Verdict: MODERATE bias — N1 does not need redesigning

The bias is real but bounded, and the mitigation is simple and beneficial:

1. **Include `n_rel` as a model covariate** so residual sampling effects are absorbed rather than attributed to forcing.
2. **Restrict event-conditioned recovery estimation to `n_rel ≥ 0.9`** — improves the decay signal and costs only 17 % of shock events.
3. **Report the sensitivity** (raw vs partial correlation, and both decay profiles) in the paper.

⚠️ One honest caveat: the decay profile is non-monotonic at t+3 (+0.339 > t+2 +0.216) over the full 21 years, whereas the 2010–2014 pilot gave a clean monotonic decay. The recovery term is identifiable but noisier at full-period scale than the pilot suggested, and per-county estimates will need regularisation.

`*_n` columns are retained throughout the panel to support this filtering.
