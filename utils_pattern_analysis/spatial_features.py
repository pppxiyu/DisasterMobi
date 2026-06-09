"""
Spatiotemporal feature extraction using OpenStreetMap data (via srai).

Functions
---------
union_filters          – merge per-category OSM tag filters
aggregate_to_categories– sum fine-grained OSM counts into high-level categories
get_component_features – weighted POI profiles per decomposition component
tf_iwf                 – TF-IWF normalisation for component × POI matrices
get_comp_features_df   – full pipeline: load OSM, embed, aggregate, profile
vis_heatmap_component_similarity – cosine-similarity heatmap between two cities' components

Constants
---------
functional_categories  – predefined land-use / POI category filter dictionary
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from srai.loaders import OSMPbfLoader
from srai.joiners import IntersectionJoiner
from srai.embedders import CountEmbedder


# ── Category filter definition ────────────────────────────────────────────────

functional_categories = {
    "residential": {
        "landuse":  ["residential"],
        "building": ["residential", "apartments", "house"],
    },
    "commercial_retail": {
        "landuse": ["commercial", "retail"],
        "shop":    True,
        "amenity": ["marketplace", "bank", "atm"],
    },
    "food_service": {
        "amenity": ["restaurant", "cafe", "fast_food", "bar", "pub", "food_court"],
    },
    "education": {
        "amenity":  ["school", "university", "college", "kindergarten", "library"],
        "building": ["school", "university"],
    },
    "healthcare": {
        "amenity":    ["hospital", "clinic", "doctors", "pharmacy", "dentist"],
        "healthcare": True,
    },
    "recreation_green": {
        "leisure": ["park", "playground", "sports_centre", "pitch", "garden"],
        "landuse": ["grass", "forest", "recreation_ground"],
    },
}


# ── OSM helpers ───────────────────────────────────────────────────────────────

def union_filters(category_dict):
    """Merge all per-category OSM tag filters into a single filter dict."""
    combined = {}
    for cat_filter in category_dict.values():
        for key, values in cat_filter.items():
            if key not in combined:
                combined[key] = values if values is True else list(values)
            elif combined[key] is True or values is True:
                combined[key] = True
            else:
                combined[key] = list(set(combined[key]) | set(values))
    return combined


def aggregate_to_categories(raw_counts, category_dict):
    """
    Sums fine-grained OSM count columns into high-level functional categories.

    Parameters
    ----------
    raw_counts    : DataFrame with OSM tag columns (e.g. 'shop_supermarket')
    category_dict : dict mapping category name → OSM tag filter

    Returns
    -------
    DataFrame with one column per category, rows aligned to raw_counts.
    """
    aggregated = pd.DataFrame(index=raw_counts.index)
    for cat_name, cat_filter in category_dict.items():
        matching = []
        for key, values in cat_filter.items():
            if values is True:
                matching += [c for c in raw_counts.columns if c.startswith(f"{key}_")]
            else:
                matching += [f"{key}_{v}" for v in values if f"{key}_{v}" in raw_counts.columns]
        matching = list(set(matching))
        aggregated[cat_name] = raw_counts[matching].sum(axis=1) if matching else 0
    return aggregated


def get_component_features(U, spatial_mapping, share_features):
    """
    Computes a weighted POI profile for each decomposition component.

    Parameters
    ----------
    U              : ndarray (zones × components)
    spatial_mapping: dict {index → tract_name}
    share_features : DataFrame (tract_name × POI categories)

    Returns
    -------
    DataFrame (components × POI categories)
    """
    ordered = [spatial_mapping[i] for i in range(len(spatial_mapping))]
    aligned = share_features.reindex(ordered).fillna(0)
    matrix  = np.dot(U.T, aligned.values)
    names   = [f"Component {i}" for i in range(U.shape[1])]
    return pd.DataFrame(matrix, index=names, columns=share_features.columns)


def tf_iwf(df, scale_iwf=1, normalize=None):
    """
    Computes TF-IWF feature vectors for a (components × POI types) count matrix.

    normalize : None | 'L2' | 'total'
    """
    counts    = df.to_numpy(dtype=float)
    row_tots  = counts.sum(axis=1, keepdims=True)
    tf        = np.divide(counts, row_tots, out=np.zeros_like(counts), where=row_tots != 0)
    nt        = counts.sum(axis=0)
    iwf       = np.log(nt.sum() / (nt + 1e-9)) ** scale_iwf
    result    = tf * iwf

    if normalize == "L2":
        norms  = np.linalg.norm(result, axis=1, keepdims=True)
        result = np.divide(result, norms, out=np.zeros_like(result), where=norms != 0)
    elif normalize == "total":
        sums   = result.sum(axis=1, keepdims=True)
        result = np.divide(result, sums, out=np.zeros_like(result), where=sums != 0)

    return pd.DataFrame(result, index=df.index, columns=df.columns)


def get_comp_features_df(tracts_gdf, U, s_mapping, scale_iwf=1, norm_method='total'):
    """
    Full pipeline: load OSM POI counts for tracts, embed, aggregate to
    functional categories, and compute per-component feature profiles.

    Parameters
    ----------
    tracts_gdf  : GeoDataFrame with 'aggr_id' column, CRS EPSG:4326
    U           : spatial factor matrix (zones × components)
    s_mapping   : dict {tensor_index → aggr_id}
    """
    gdf         = tracts_gdf.set_index('aggr_id')
    gdf.index   = gdf.index.astype(str)
    gdf.index.name = 'region_id'

    print(f"  tracts: {len(gdf)}, CRS: {gdf.crs}")

    loader      = OSMPbfLoader()
    features    = loader.load(gdf, tags=union_filters(functional_categories))

    joiner      = IntersectionJoiner()
    joint       = joiner.transform(gdf, features)

    embedder    = CountEmbedder()
    raw_counts  = embedder.transform(gdf, features, joint)

    cat_features = aggregate_to_categories(raw_counts, functional_categories)
    return get_component_features(U, s_mapping, cat_features)


# ── Visualisation ─────────────────────────────────────────────────────────────

def vis_heatmap_component_similarity(df1, df2, output_dir='outputs', tag=''):
    """
    Cosine-similarity heatmap between rows of df1 (Baton Rouge) and df2 (New Orleans),
    based on OSM land-use feature vectors per component.
    Saves to: <output_dir>/component_similarity<tag>.png
    """
    import os
    sim = cosine_similarity(df1, df2)
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim, annot=True, fmt=".2f", cmap='YlGnBu',
                xticklabels=df2.index, yticklabels=df1.index)
    plt.xlabel("Target Components", fontsize=12)
    plt.ylabel("Basis Components",  fontsize=12)
    plt.xticks(rotation=45); plt.yticks(rotation=0)
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'component_similarity{tag}.png'), bbox_inches='tight', dpi=150)
    plt.close()
