"""
Shared configuration: file paths and time-slot constants used by all pipelines.

City #1 = Baton Rouge (Hurricane Ida, 2021).
City #2 = Fort Myers   (Hurricane Ian, 2022).   ← replaced New Orleans (2026-06)

The 2026-06 datasets are BLOCK-GROUP resolution, 2-hour interval, and use a
bare-name convention (no `_2h_block_group` suffix).  A legacy `census_tract`
branch is kept for reproducing older runs (those used New Orleans as city #2).
"""

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = 'data'
OUTPUT_DIR      = 'outputs'
MODEL_CACHE_DIR = '.cache/model'   # NN checkpoints (best_model.pth); under the .cache root

# ── Spatial resolution ───────────────────────────────────────────────────────
# 'block_group' = current data (Baton Rouge + Fort Myers, bare-name pkls).
# 'census_tract' = legacy data (Baton Rouge + New Orleans, suffix-name pkls).
AGG_LEVEL = 'block_group'

if AGG_LEVEL == 'block_group':
    # Current bare-name block-group files (2h).
    BR_GRAPH_PATH = 'data/Baton_Rouge_Ida_2021_graph_intersection.pkl'
    FM_GRAPH_PATH = 'data/Fort_Myers_Ian_2022_graph_intersection.pkl'
else:
    # Legacy census-tract files; city #2 was New Orleans.
    BR_GRAPH_PATH = 'data/Baton_Rouge_Ida_2021_graph_intersection_2h.pkl'
    FM_GRAPH_PATH = 'data/New_Orleans_Ida_2021_graph_intersection_2h.pkl'

# ── Analysis-window alignment (disaster placement) ────────────────────────────
BR_ANALYSIS_DAYS = 151    # trim BR data, counting from start end
FM_ANALYSIS_DAYS = 44     # trim Fort Myers'data, counting from start end

# Per-resolution geography CSVs (columns: geography_id, geometry_wkt, …).
# Used by the AGG_LEVEL-aware loader (load_city_geo_or_warn in build_graphs).
# The active block-group geometry for Baton Rouge and Fort Myers lives directly
# under data/; the census_tract entries are legacy (Baton Rouge + New Orleans).
BR_GEO_CSV = {'census_tract': 'data/cubique_raw/Baton_Rouge_census_tract_geo.csv',  # legacy
              'block_group' : 'data/Baton_Rouge_block_group_geo.csv'}
FM_GEO_CSV = {'census_tract': 'data/cubique_raw/New_Orleans_census_tract_geo.csv',  # legacy NO
              'block_group' : 'data/Fort_Myers_block_group_geo.csv'}

# ── Temporal resolution ───────────────────────────────────────────────────────
# 2h (current):
#   12 slots/day;  remove 0-6am (3 slots) and 22-24h (1 slot)
#   → 8 active slots covering 6am-22h
#
#   Slot map (2h):  0:0-2  1:2-4  2:4-6 | 3:6-8  4:8-10  5:10-12
#                   6:12-14  7:14-16  8:16-18  9:18-20  10:20-22 | 11:22-24
#                   active = slots 3–10 (inclusive)
#
# To switch to 3h resolution, change to:
#   SLOT_PER_DAY=8, SLOT_TRIM_START=2, SLOT_TRIM_END=1, PROPHET_FREQ='3H'
#   and update graph paths to *_3h.pkl
SLOT_PER_DAY    = 12  # 2-hour slots per 24 h
SLOT_TRIM_START = 3   # remove first 3 slots (0:00–6:00)
SLOT_TRIM_END   = 1   # remove last  1 slot  (22:00–24:00)
SLOTS_ACTIVE    = SLOT_PER_DAY - SLOT_TRIM_START - SLOT_TRIM_END  # 8 daytime slots

PROPHET_FREQ       = '2H'
PROPHET_START_DATE = '2021-04-15'   # Baton Rouge (city #1) start date
