"""
Download per-block-group ACS median household income (table B19013) from the
Census Bureau API, as a socioeconomic per-zone attribute for the NMF components.

Mirrors fetch_sld_landuse's cache pattern: the RAW Census value is cached once to
data/space_function/<city>_block_group_acs_income_raw.csv
(columns: aggr_id, geoid20, median_household_income); downstream code reads that
cache and aggregates the per-block-group income to each component.

Data source
-----------
Census ACS 5-Year Detailed Tables API (block-group income REQUIRES the 5-year
dataset).  No key is required at this volume; an optional CENSUS_API_KEY env var
is appended when present.

    https://api.census.gov/data/<year>/acs/acs5
        ?get=B19013_001E&for=block group:*&in=state:<SS> county:<CCC> tract:*

B19013_001E = median household income in the past 12 months (inflation-adjusted
dollars of the survey year).  Census suppresses/omits some block groups and
returns negative "jam" sentinels (e.g. -666666666); those are stored as NaN.

Public API
----------
  fetch_acs_income_for_counties(county_prefixes, year, var) -> DataFrame [geoid20, median_household_income]
  ensure_city_income_raw(city_label, aggr_ids, out_csv, year) -> path | None   (cache-aware, RAW only)
  load_city_income(raw_csv) -> DataFrame [aggr_id, geoid20, median_household_income]
"""
import os

import numpy as np
import pandas as pd

from config import DATA_DIR
# Reuse the SLD fetcher's GEOID + HTTP machinery (single source of truth).
from utils_data_processing.fetch_sld_landuse import (
    aggr_id_to_geoid20, _county_prefixes, _http_get_json, HTTP_TIMEOUT,
)

ACS_BASE_URL = "https://api.census.gov/data/{year}/acs/acs5"
DEFAULT_YEAR = 2021
DEFAULT_VAR  = 'B19013_001E'     # median household income (past 12 months)

# Cache CSVs and the API-key file live in their own data subfolder
# (data/ is gitignored, so neither the cache nor the key is ever committed).
ACS_DATA_DIR = os.path.join(DATA_DIR, 'socioeconomic')
KEY_FILE     = os.path.join(ACS_DATA_DIR, 'census_api_key.txt')


def _resolve_api_key():
    """Census API key, in priority order: the CENSUS_API_KEY env var (override),
    else the plain-text KEY_FILE under data/socioeconomic/.  Returns the key
    string, or None if neither is set."""
    key = os.environ.get('CENSUS_API_KEY')
    if key and key.strip():
        return key.strip()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, encoding='utf-8') as f:
            k = f.read().strip()
        return k or None
    return None


def fetch_acs_income_for_counties(county_prefixes, year=DEFAULT_YEAR,
                                  var=DEFAULT_VAR, timeout=HTTP_TIMEOUT):
    """
    Fetch raw ACS block-group median household income for the given 5-digit
    state+county GEOID prefixes (e.g. '22033'), one API call per county.

    Returns a DataFrame [geoid20, median_household_income] (one row per block
    group in those counties); suppressed/unavailable cells are NaN.  Empty
    DataFrame if nothing is returned.  A county whose query fails is skipped
    (warned) so the others still contribute.
    """
    url = ACS_BASE_URL.format(year=year)
    api_key = _resolve_api_key()
    rows = []
    for prefix in sorted(set(county_prefixes)):
        state, county = prefix[:2], prefix[2:5]
        params = {
            'get': var,
            'for': 'block group:*',
            'in':  f'state:{state} county:{county} tract:*',
        }
        if api_key:
            params['key'] = api_key
        try:
            data = _http_get_json(url, params, timeout=timeout)
        except Exception as e:                      # one bad county must not kill all
            print(f"  ⚠ ACS query failed for county {prefix} ({e!r}); skipping.")
            continue
        if not data or len(data) < 2:
            continue
        header, body = data[0], data[1:]
        idx = {name: i for i, name in enumerate(header)}
        vi = idx[var]
        si, ci, ti, bi = idx['state'], idx['county'], idx['tract'], idx['block group']
        for r in body:
            geoid = f"{r[si]}{r[ci]}{r[ti]}{r[bi]}"
            raw = r[vi]
            val = float(raw) if raw not in (None, '') else np.nan
            if val < 0:                             # Census jam sentinel → suppressed
                val = np.nan
            rows.append((geoid, val))
    df = pd.DataFrame(rows, columns=['geoid20', 'median_household_income'])
    if not df.empty:
        df['geoid20'] = df['geoid20'].astype(str)
        df = df.drop_duplicates('geoid20').reset_index(drop=True)
    return df


def ensure_city_income_raw(city_label, aggr_ids, out_csv, year=DEFAULT_YEAR,
                           var=DEFAULT_VAR, force=False):
    """
    Ensure a city's RAW ACS median-income block-group table is cached on disk,
    downloading it only if missing (mirrors ensure_city_landuse_raw).

    Returns the CSV path on success.  RAISES RuntimeError if the data cannot be
    obtained — no rows returned, or nothing matched the city's block groups — so
    the failure is LOUD, not a silent skip.  The most common cause is a missing or
    invalid CENSUS_API_KEY (the Census Data API is no longer keyless).
    """
    if os.path.exists(out_csv) and not force:
        print(f"  ✓ {city_label}: raw ACS income cache present — skipping download "
              f"({out_csv}).")
        return out_csv

    # GEOID20 → aggr_id so output aligns 1:1 with the city's block groups.
    geoid_to_aggr = {}
    for a in aggr_ids:
        try:
            geoid_to_aggr[aggr_id_to_geoid20(a)] = a
        except ValueError as e:
            print(f"  ⚠ {city_label}: skipping id — {e}")
    if not geoid_to_aggr:
        raise RuntimeError(f"{city_label}: no usable block-group ids for ACS income.")

    prefixes = _county_prefixes(geoid_to_aggr.values())
    print(f"  {city_label}: {len(geoid_to_aggr)} block groups across counties "
          f"{prefixes} — querying ACS {year} {var} …")
    df = fetch_acs_income_for_counties(prefixes, year=year, var=var)
    if df.empty:
        raise RuntimeError(
            f"{city_label}: ACS returned NO rows for {var} ({year} 5-year). The "
            f"Census Data API requires a key — set the CENSUS_API_KEY environment "
            f"variable (free key: https://api.census.gov/data/key_signup.html) and "
            f"re-run, or check the network / dataset year."
        )

    df = df[df['geoid20'].isin(geoid_to_aggr)].copy()
    if df.empty:
        raise RuntimeError(
            f"{city_label}: ACS returned rows but NONE matched the city's "
            f"{len(geoid_to_aggr)} block groups (GEOID mismatch?)."
        )
    df['aggr_id'] = df['geoid20'].map(geoid_to_aggr)
    matched, requested = len(df), len(geoid_to_aggr)
    n_inc = int(df['median_household_income'].notna().sum())
    print(f"  {city_label}: matched {matched}/{requested} block groups "
          f"({100*matched/requested:.1f}%); {n_inc} with income, "
          f"{matched - n_inc} suppressed.")

    df = (df[['aggr_id', 'geoid20', 'median_household_income']]
          .sort_values('aggr_id').reset_index(drop=True))
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  ✓ {city_label}: wrote {len(df)} raw rows → {out_csv}")
    return out_csv


def load_city_income(raw_csv):
    """
    Read a cached RAW ACS income CSV.  Returns a DataFrame
    [aggr_id, geoid20, median_household_income] (NaN where the Census suppressed
    the value).  Raises FileNotFoundError if the cache is absent (call
    ensure_city_income_raw first).
    """
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(
            f"Raw ACS income CSV not found: {raw_csv}. "
            f"Call ensure_city_income_raw first."
        )
    df = pd.read_csv(raw_csv)
    if 'geoid20' in df.columns:
        df['geoid20'] = df['geoid20'].astype(str)
    return df
