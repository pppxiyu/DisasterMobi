"""
Download per-block-group land-use / functional data from the EPA Smart Location
Database (SLD) and assign each block group a dominant functional category, as a
U.S. analogue to the EULUC-China land-use labelling used in the reference paper.

Why
---
The paper (Gong et al.) labels each 1 km grid with a *dominant* land-use category
(>50% share, else "Mix") from EULUC-China, then summarises each NMF component's
flows by the land use of the origin and destination zones (Figure 9).  EULUC has
no U.S. equivalent; the closest free, block-group-native source is EPA's Smart
Location Database (SLD v3), which gives, per census block group, housing units
and employment broken down by sector.

Data source
-----------
ArcGIS REST service (queryable, no key required):
    https://geodata.epa.gov/arcgis/rest/services/OA/SmartLocationDatabase/MapServer/7/query
Every sub-layer exposes the full SLD attribute table; layer 7 is used here.
Supports pagination (maxRecordCount 5000) and returns JSON.

Functional mapping (SLD → EULUC-style categories)
-------------------------------------------------
SLD has no "residential" employment sector, so residential intensity is proxied
by housing units (CountHU * RESIDENTIAL_WEIGHT).  Employment-by-sector (8-tier,
E8_*) maps to the non-residential categories — see FUNCTION_FIELD_MAP and
CATEGORIES below for the authoritative definition.

A block group's dominant category is the one whose (reweighted) share of the
category scores exceeds the dominant threshold; otherwise it is "Mix".  Zones
with no housing and no jobs are "Unknown".

NOTE: EULUC's "transportation" category has no clean SLD employment/land-use
counterpart and is intentionally NOT produced here.  RESIDENTIAL_WEIGHT puts one
housing unit on the same footing as one job; it is a modelling knob — adjust it
(or the FUNCTION_FIELD_MAP) to change the classification, then re-run with
force=True to refresh the cached CSV.

Storage model
-------------
data/ holds ONLY the raw SLD columns (downloaded once, then reused).  Every
derived column — per-category shares and the dominant_category label — is
recomputed on-the-fly from the raw data, so the classification (category map,
TF-IWF scale, residential weight) can be tuned per run without re-downloading.

Public API
----------
  aggr_id_to_geoid20(aggr_id)
  fetch_sld_for_counties(county_prefixes, ...)            -> DataFrame (raw SLD fields)
  ensure_city_landuse_raw(city_label, aggr_ids, out_csv)  -> path | None  (cache-aware, RAW only)
  classify_dominant_function(df, ..., iwf_scale)          -> DataFrame (+ shares + label)
  load_city_landuse(raw_csv, ..., iwf_scale)              -> DataFrame  (raw → on-the-fly classify)
"""
import os
import json
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

try:                       # requests preferred; urllib is a stdlib fallback
    import requests
    _HAVE_REQUESTS = True
except Exception:          # pragma: no cover
    _HAVE_REQUESTS = False


# ── Service / field configuration ─────────────────────────────────────────────

SLD_QUERY_URL = ("https://geodata.epa.gov/arcgis/rest/services/OA/"
                 "SmartLocationDatabase/MapServer/{layer}/query")
DEFAULT_LAYER = 7
PAGE_SIZE     = 2000       # < maxRecordCount (5000)
HTTP_TIMEOUT  = 90

# Fields requested from SLD (canonical casing; the service is case-insensitive on
# input but returns mixed casing, e.g. "E8_off", so responses are lower-cased).
RAW_FIELDS = [
    'GEOID20', 'TotPop', 'CountHU', 'HH', 'TotEmp',
    'E8_Ret', 'E8_Off', 'E8_Ind', 'E8_Svc', 'E8_Ent', 'E8_Ed', 'E8_Hlth', 'E8_Pub',
]

# Functional category → SLD E8 employment fields (lower-cased keys).
#   commercial : retail + office + service (prof./admin/other services)
#   leisure    : entertainment (arts/recreation + accommodation/food)
#   industrial : industrial
#   health     : health care and social assistance
#   public     : education + public administration
# 'residential' is handled separately via housing units (counthu).
FUNCTION_FIELD_MAP = {
    'commercial': ['e8_ret', 'e8_off', 'e8_svc'],
    'leisure':    ['e8_ent'],
    'industrial': ['e8_ind'],
    'health':     ['e8_hlth'],
    'public':     ['e8_ed', 'e8_pub'],
}
CATEGORIES         = ['residential', 'commercial', 'leisure',
                      'industrial', 'health', 'public']
RESIDENTIAL_WEIGHT = 1.0   # housing-unit ↔ job equivalence (modelling knob)
DOMINANT_THRESHOLD = 0.5   # >50% (reweighted) share → that category, else "Mix"

# Category reweighting before labelling.  'tf_iwf' applies Term-Frequency /
# Inverse-Word-Frequency (mirrors spatial_features.tf_iwf) so the near-ubiquitous
# residential category is down-weighted and rare/concentrated functions surface;
# 'raw_share' uses plain score shares (residential then dominates almost
# everywhere).  These are DEFAULTS only — the classification (incl. IWF_SCALE)
# is computed on-the-fly by load_city_landuse; the run script passes its own
# scale so it can be tuned without touching the cached raw data.
WEIGHTING = 'tf_iwf'
IWF_SCALE = 1.0

# State USPS abbreviation → 2-digit FIPS (for aggr_id → GEOID20 conversion).
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
    'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12', 'GA': '13', 'HI': '15',
    'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21',
    'LA': '22', 'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27',
    'MS': '28', 'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
    'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46',
    'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53',
    'WV': '54', 'WI': '55', 'WY': '56', 'PR': '72',
}


# ── ID conversion ─────────────────────────────────────────────────────────────

def aggr_id_to_geoid20(aggr_id):
    """
    Convert a Snowflake-style block-group id to a 12-digit census FIPS (GEOID20).

        'US.LA.033.004006.3'  ->  '22' + '033' + '004006' + '3'  =  '220330040063'

    Raises ValueError for non-block-group ids (e.g. census-tract ids with 4 parts)
    or unknown state abbreviations.
    """
    parts = aggr_id.split('.')
    if len(parts) != 5 or parts[0] != 'US':
        raise ValueError(
            f"Expected a block-group id 'US.<ST>.<CCC>.<TRACT>.<BG>', got '{aggr_id}'. "
            f"This loader only supports AGG_LEVEL='block_group'."
        )
    st, county, tract, bg = parts[1], parts[2], parts[3], parts[4]
    state_fips = STATE_FIPS.get(st)
    if state_fips is None:
        raise ValueError(f"Unknown state abbreviation '{st}' in id '{aggr_id}'.")
    return f"{state_fips}{county}{tract}{bg}"


def _county_prefixes(aggr_ids):
    """Unique 5-digit state+county GEOID prefixes (e.g. {'22033'}) for the ids."""
    return sorted({aggr_id_to_geoid20(a)[:5] for a in aggr_ids})


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _http_get_json(url, params, timeout=HTTP_TIMEOUT):
    """GET url?params and parse JSON, using requests if available else urllib."""
    if _HAVE_REQUESTS:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    full = url + '?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(full, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _query_sld_paged(where, out_fields, layer=DEFAULT_LAYER,
                     page_size=PAGE_SIZE, timeout=HTTP_TIMEOUT):
    """
    Run a paginated SLD query and return a list of attribute dicts (keys lower-cased).

    Pagination uses resultOffset/resultRecordCount with a stable orderBy so pages
    don't overlap; stops once a short page is returned.
    """
    url  = SLD_QUERY_URL.format(layer=layer)
    rows, offset, page = [], 0, 0
    while True:
        params = {
            'where':            where,
            'outFields':        ','.join(out_fields),
            'returnGeometry':   'false',
            'orderByFields':    'GEOID20',
            'resultOffset':     offset,
            'resultRecordCount': page_size,
            'f':                'json',
        }
        data = _http_get_json(url, params, timeout=timeout)
        if 'error' in data:
            raise RuntimeError(f"SLD query error: {data['error']}")
        feats = data.get('features', [])
        rows.extend({k.lower(): v for k, v in f['attributes'].items()} for f in feats)
        page += 1
        if len(feats) < page_size:
            break
        offset += page_size
        if page > 200:                      # safety backstop (~400k rows)
            raise RuntimeError("SLD pagination exceeded 200 pages; aborting.")
        time.sleep(0.2)                      # be polite to the public service
    return rows


def fetch_sld_for_counties(county_prefixes, layer=DEFAULT_LAYER,
                           out_fields=RAW_FIELDS, timeout=HTTP_TIMEOUT):
    """
    Fetch raw SLD block-group rows for the given state+county prefixes.

    Returns a DataFrame with lower-cased column names (one row per block group in
    those counties).  Empty DataFrame if nothing is returned.
    """
    where = ' OR '.join(f"GEOID20 LIKE '{p}%'" for p in county_prefixes)
    print(f"  SLD query: WHERE {where}")
    rows = _query_sld_paged(where, out_fields, layer=layer, timeout=timeout)
    df = pd.DataFrame(rows)
    if not df.empty:
        df['geoid20'] = df['geoid20'].astype(str)
    print(f"  SLD returned {len(df):,} block-group rows "
          f"for {len(county_prefixes)} county prefix(es).")
    return df


# ── Classification ────────────────────────────────────────────────────────────

def classify_dominant_function(df, residential_weight=RESIDENTIAL_WEIGHT,
                               dominant_threshold=DOMINANT_THRESHOLD,
                               weighting=WEIGHTING, iwf_scale=IWF_SCALE):
    """
    Add per-category shares and a `dominant_category` label to each block group.

    Raw scores
    ----------
        score_residential = counthu * residential_weight   (housing proxy)
        score_<cat>        = sum of that category's E8 employment fields

    Why not raw shares
    ------------------
    Housing units (residential) and job counts live on different scales AND
    housing is near-ubiquitous, so a plain `score / row_total` share labels
    almost every block group "residential".  To get a discriminative label we
    reweight by how *distinctive* each category is, using TF-IWF — the same
    Term-Frequency / Inverse-Word-Frequency formula as
    `utils_pattern_analysis.spatial_features.tf_iwf` (documents = block groups,
    "words" = the functional categories):

        tf [i,c]  = score[i,c] / sum_c score[i,c]            (within-BG fraction)
        iwf[c]    = ( ln( sum(nt) / nt[c] ) ) ** iwf_scale,  nt[c] = sum_i score[i,c]
        weight    = tf * iwf
        share[i,c]= weight[i,c] / sum_c weight[i,c]          (renormalised)

    A category that carries a large share of the region's total mass (housing)
    gets a small IWF and is down-weighted; a rare/concentrated category
    (industrial, public) gets a large IWF and is up-weighted.

    Labelling
    ---------
        dominant_category = argmax category if its (reweighted) share exceeds
        dominant_threshold, else 'Mix'; 'Unknown' when no housing and no jobs.

    Parameters
    ----------
    weighting : 'tf_iwf' (default) or 'raw_share' (no IWF reweighting).
    iwf_scale : exponent on the IWF term (higher = stronger distinctiveness pull).

    Negative SLD sentinels and NaNs are clamped to 0.  The IWF weights used are
    printed for transparency.
    """
    df = df.copy()

    def col(name):
        return (df[name].astype(float).clip(lower=0).fillna(0.0)
                if name in df else pd.Series(0.0, index=df.index))

    score = pd.DataFrame(index=df.index)
    score['residential'] = col('counthu') * residential_weight   # housing proxy
    for cat, fields in FUNCTION_FIELD_MAP.items():               # employment categories
        score[cat] = sum(col(c) for c in fields)
    score = score[CATEGORIES]

    raw_total = score.sum(axis=1)                     # for the Unknown test
    S = score.to_numpy(dtype=float)

    if weighting == 'tf_iwf':
        row_tot = S.sum(axis=1, keepdims=True)
        tf      = np.divide(S, row_tot, out=np.zeros_like(S), where=row_tot != 0)
        nt      = S.sum(axis=0)                        # total mass per category
        iwf     = np.log(nt.sum() / (nt + 1e-9)) ** iwf_scale
        weighted = tf * iwf
        print("    TF-IWF category weights (lower = more ubiquitous → down-weighted):")
        print("      " + ", ".join(f"{c}={w:.3f}" for c, w in zip(CATEGORIES, iwf)))
    elif weighting == 'raw_share':
        weighted = S
    else:
        raise ValueError(f"weighting must be 'tf_iwf' or 'raw_share', got '{weighting}'")

    wsum = weighted.sum(axis=1, keepdims=True)
    shares = np.divide(weighted, wsum, out=np.zeros_like(weighted), where=wsum != 0)
    for j, c in enumerate(CATEGORIES):
        df[f'share_{c}'] = shares[:, j]

    best_i = shares.argmax(axis=1)
    best_v = shares.max(axis=1)
    rtot   = raw_total.to_numpy()

    labels = []
    for i in range(len(df)):
        if rtot[i] <= 0:
            labels.append('Unknown')
        elif best_v[i] > dominant_threshold:
            labels.append(CATEGORIES[best_i[i]])
        else:
            labels.append('Mix')
    df['dominant_category'] = labels
    return df


# ── Cache-aware RAW download + on-the-fly classification ──────────────────────

def ensure_city_landuse_raw(city_label, aggr_ids, out_csv,
                            layer=DEFAULT_LAYER, force=False):
    """
    Ensure a city's RAW SLD block-group table is cached on disk, downloading it
    only if missing.

    ONLY raw SLD columns are stored (aggr_id, geoid20, population, housing,
    employment-by-sector).  No derived columns (shares, dominant_category) are
    written — those are computed on-the-fly by load_city_landuse, so the
    classification can be re-tuned (e.g. a different TF-IWF scale) without
    re-downloading or rewriting the cached data.

    If `out_csv` exists and `force` is False, the download is skipped.

    Parameters
    ----------
    city_label : str   — for log messages, e.g. 'Baton Rouge'.
    aggr_ids   : list  — block-group ids ('US.ST.CCC.TRACT.BG') for the city.
    out_csv    : str   — output CSV path (raw data only).
    layer, force : see module-level defaults.

    Returns
    -------
    str  the CSV path on success, or None if the download failed (a warning is
         printed; callers may continue without land-use data).
    """
    if os.path.exists(out_csv) and not force:
        print(f"  ✓ {city_label}: raw SLD cache present — skipping download "
              f"({out_csv}).")
        return out_csv

    try:
        # GEOID20 → aggr_id map so output aligns 1:1 with the city's block groups.
        geoid_to_aggr = {}
        for a in aggr_ids:
            try:
                geoid_to_aggr[aggr_id_to_geoid20(a)] = a
            except ValueError as e:
                print(f"  ⚠ {city_label}: skipping id — {e}")

        if not geoid_to_aggr:
            print(f"  ⚠ {city_label}: no usable block-group ids — skipping.")
            return None

        prefixes = _county_prefixes(geoid_to_aggr.values())
        print(f"  {city_label}: {len(geoid_to_aggr)} block groups across "
              f"counties {prefixes} — querying SLD …")

        df = fetch_sld_for_counties(prefixes, layer=layer)
        if df.empty:
            print(f"  ⚠ {city_label}: SLD returned no rows — skipping.")
            return None

        # Keep only this city's block groups; attach aggr_id.
        df = df[df['geoid20'].isin(geoid_to_aggr)].copy()
        df['aggr_id'] = df['geoid20'].map(geoid_to_aggr)
        matched, requested = len(df), len(geoid_to_aggr)
        print(f"  {city_label}: matched {matched}/{requested} block groups in SLD "
              f"({100*matched/requested:.1f}%).")

        # Store RAW columns only — no derived/classified columns.
        raw_cols = [c.lower() for c in RAW_FIELDS if c.lower() != 'geoid20']
        ordered  = ['aggr_id', 'geoid20'] + [c for c in raw_cols if c in df.columns]
        df = df[ordered].sort_values('aggr_id').reset_index(drop=True)

        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"  ✓ {city_label}: wrote {len(df)} raw rows → {out_csv}")
        return out_csv

    except Exception as e:                  # network / service / parse failure
        print(f"  ⚠ {city_label}: raw SLD download FAILED ({e!r}). "
              f"Continuing without land-use data.")
        return None


def load_city_landuse(raw_csv, residential_weight=RESIDENTIAL_WEIGHT,
                      weighting=WEIGHTING, iwf_scale=IWF_SCALE,
                      dominant_threshold=DOMINANT_THRESHOLD):
    """
    Read a cached RAW SLD CSV and compute the functional classification
    ON-THE-FLY (per-category shares + dominant_category) with the given
    TF-IWF scale / weighting.  Nothing is written back to disk — the returned
    DataFrame is for in-memory use (e.g. the O×D functional analysis).

    Raises FileNotFoundError if the raw CSV is absent (call
    ensure_city_landuse_raw first).
    """
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(
            f"Raw SLD CSV not found: {raw_csv}. Call ensure_city_landuse_raw first."
        )
    df = pd.read_csv(raw_csv)
    if 'geoid20' in df.columns:
        df['geoid20'] = df['geoid20'].astype(str)
    return classify_dominant_function(
        df, residential_weight=residential_weight,
        weighting=weighting, iwf_scale=iwf_scale,
        dominant_threshold=dominant_threshold,
    )
