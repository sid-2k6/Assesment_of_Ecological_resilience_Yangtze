#!/usr/bin/env bash
# Probe each candidate data endpoint: is it reachable from this sandbox, and
# does it require credentials? Reports HTTP status + content-length only.
probe () {
  local name="$1" url="$2"
  local out
  out=$(curl -sSL -o /dev/null -m 25 -w "%{http_code} %{size_download} %{content_type}" \
        -r 0-1024 "$url" 2>&1) || out="ERR"
  printf "%-34s %s\n" "$name" "$out"
}

echo "=== OPEN / NO AUTH EXPECTED ==="
probe "GADM China (boundaries)"   "https://geodata.ucdavis.edu/gadm/gadm4.1/gpkg/gadm41_CHN.gpkg"
probe "Natural Earth admin1"      "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
probe "HydroSHEDS (hydrosheds.org)" "https://data.hydrosheds.org/file/HydroRIVERS/HydroRIVERS_v10_as_shp.zip"
probe "HydroLAKES"                "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"
probe "CHIRPS precip (UCSB)"      "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/chirps-v2.0.2020.01.tif.gz"
probe "SoilGrids WCS"             "https://maps.isric.org/mapserv?map=/map/phh2o.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCapabilities"
probe "WorldPop CHN"              "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/CHN/chn_ppp_2020.tif"
probe "GHSL GHS-POP"              "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E2020_GLOBE_R2023A_54009_1000/V1-0/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip"
probe "OpenTopography SRTM API"   "https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1&south=30&north=30.1&west=110&east=110.1&outputFormat=GTiff"
probe "Copernicus DEM (AWS)"      "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N30_00_E110_00_DEM/Copernicus_DSM_COG_10_N30_00_E110_00_DEM.tif"
probe "Zenodo API (CLCD search)"  "https://zenodo.org/api/records?q=CLCD+China+land+cover&size=3"

echo
echo "=== CREDENTIALS EXPECTED ==="
probe "NASA LP DAAC (MODIS)"      "https://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.061/2020.01.01/"
probe "NASA Earthdata login"      "https://urs.earthdata.nasa.gov/"
probe "Copernicus CDS (ERA5)"     "https://cds.climate.copernicus.eu/api/v2/resources/reanalysis-era5-land"
probe "Google Earth Engine API"   "https://earthengine.googleapis.com/v1/projects"
probe "Microsoft Planetary Comp"  "https://planetarycomputer.microsoft.com/api/stac/v1/collections"

echo
echo "=== CHINESE PORTALS ==="
probe "RESDC (resdc.cn)"          "https://www.resdc.cn/"
probe "TPDC (data.tpdc.ac.cn)"    "https://data.tpdc.ac.cn/en/"
probe "Geospatial Data Cloud"     "https://www.gscloud.cn/"
probe "National Earth Sys Sci DC" "https://www.geodata.cn/"
