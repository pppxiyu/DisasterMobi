"""
Shared configuration: file paths and time-slot constants used by all pipelines.

TWO city-definition mechanisms coexist in this repo:
  * The production NMF pipeline (run_pattern_nmf.py) defines its own 5-city-event
    registry, CITY_EVENTS (BR_Ida, FM_Ian, WM_Dorian, WM_Isaias, LC_Laura), and does
    NOT read the per-city BR_/FM_ constants below.
  * The BR_/FM_ constants and PROPHET_* below serve the archived exploration scripts
    (archive/run_pattern_*.py) and the prediction pipeline (run_prediction_*.py),
    which still operate on the original two cities:
      City #1 = Baton Rouge (Hurricane Ida, 2021).
      City #2 = Fort Myers  (Hurricane Ian, 2022).   <- replaced New Orleans (2026-06)

The 2026-06 datasets are BLOCK-GROUP resolution, 2-hour interval, and use a
bare-name convention (no `_2h_block_group` suffix).
"""

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = 'data'
OUTPUT_DIR      = 'outputs'
MODEL_CACHE_DIR = '.cache/model'   # NN checkpoints (best_model.pth); under the .cache root
OPTUNA_CACHE_DIR = '.cache/optuna' # Optuna studies: .cache/optuna/<task>/<task>.db (NMF + GRU tuners)

# ── Spatial resolution ───────────────────────────────────────────────────────
AGG_LEVEL = 'block_group'

BR_GRAPH_PATH = 'data/Baton_Rouge_Ida_2021_graph_intersection.pkl'
FM_GRAPH_PATH = 'data/Fort_Myers_Ian_2022_graph_intersection.pkl'

# ── Analysis-window alignment (disaster placement) ────────────────────────────
BR_ANALYSIS_DAYS = 151    # keep the FIRST 151 days of the Baton Rouge graphs
FM_ANALYSIS_DAYS = 44     # keep the FIRST 44 days of the Fort Myers graphs

# Per-resolution geography CSVs (columns: geography_id, geometry_wkt, …).
# Used by the AGG_LEVEL-aware loader (load_city_geo in utils/data_processing).
BR_GEO_CSV = {'block_group': 'data/Baton_Rouge_block_group_geo.csv'}
FM_GEO_CSV = {'block_group': 'data/Fort_Myers_block_group_geo.csv'}

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
