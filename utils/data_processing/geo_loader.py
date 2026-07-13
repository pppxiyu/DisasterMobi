"""
Geometry loading for the analysis pipeline.

This module loads each city's per-resolution geography CSV (block-group or
census-tract, a Snowflake `geography_registry` export with columns
`geography_id`, `geometry_wkt`) into a GeoDataFrame for the spatial steps of the
pattern-analysis scripts (distance arrays, choropleths, land-use joins).

Data / graph GENERATION lives elsewhere
---------------------------------------
The raw-trips → OD-matrix → temporal-graph pipeline (net vs. all-legs flow,
adjacency, distances, pickling) is NOT in this repo any more.  It is generated
by the notebook

    notebook_drafts/Cubique/main_20260627.ipynb   (get_accum_od_matrix, …)

which is the source of the `data/*.pkl` graphs used here.  This file keeps only
the geometry loaders that the analysis scripts import; to (re)build graphs, run
that notebook.

Public API
----------
  load_city_geo(city_label, agg_level, geo_csv_map)  -> GeoDataFrame  (raises if missing)
"""
import os

import pandas as pd
import geopandas as gpd
from shapely import wkt


# ── Geography from Snowflake CSV export ───────────────────────────────────────
#
# The CSVs are exports from the Snowflake geography_registry table produced by
# get_msa_geographies() in the original notebooks.  Required columns:
#   geography_id  – e.g. 'US.LA.033.004006'  (census_tract)
#                        'US.LA.033.004006.2' (block_group)
#   geometry_wkt  – WKT polygon string
# Optional (used for diagnostics): centroid_lat, centroid_lng

def _load_geo_df(geo_path: str) -> gpd.GeoDataFrame:
    """
    Load a geography CSV and return a GeoDataFrame indexed by geography_id.

    Mirrors what the notebooks do after calling get_msa_geographies():
        df['geometry'] = df['geometry_wkt'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    """
    df = pd.read_csv(geo_path)
    df = df.rename(columns={'geography_id': 'aggr_id'})
    df['geometry'] = df['geometry_wkt'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    return gdf.set_index('aggr_id')


def load_city_geo(city_label, agg_level, geo_csv_map):
    """
    Load a city's geometry as a GeoDataFrame with 'aggr_id' and 'centroid'
    columns from the geography CSV for the given AGG_LEVEL.

    Centroids are computed in EPSG:3857 and stored back in EPSG:4326.
    The geometry file is mandatory.  Raises FileNotFoundError if it is absent.

    Parameters
    ----------
    city_label  : str          — for the error message, e.g. 'Fort Myers'.
    agg_level   : str          — 'census_tract' or 'block_group' (config.AGG_LEVEL).
    geo_csv_map : dict | None   — {agg_level: csv_path}, e.g. config.FM_GEO_CSV.
    """
    csv = geo_csv_map.get(agg_level) if geo_csv_map else None
    if not csv or not os.path.exists(csv):
        raise FileNotFoundError(
            f"{city_label}: a '{agg_level}' geometry file is required but was "
            f"not found ({csv}).  Provide the geo CSV "
            f"(columns: geography_id, geometry_wkt)."
        )
    gdf = _load_geo_df(csv).reset_index()
    gdf['centroid'] = gdf.geometry.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    return gdf
