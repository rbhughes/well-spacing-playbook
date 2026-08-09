"""Project-wide constants. Tune later; these are sane starting points (see docs/RECIPE.md)."""

from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# --- study scope ---
PROVINCES = ["SK", "AB"]          # SK GeoHub anchors (explicit leg typing); AB ST37 adds Clearwater
FIRST_PROD_MIN = "2014-01-01"     # modern horizontal / multilateral era

# --- geometry ---
MIN_LEG_M = 150.0                 # ignore stub legs shorter than this
NEIGHBOR_RADIUS_M = 2000.0        # candidate-pair search radius (grid-cell size)
GRID_CELL_M = 2000.0
MAX_NEIGHBOR_WELLS = 12           # neighbor-set encoder input cap
MAX_OWN_LEGS = 24                 # own-leg set encoder input cap (fishbones reach ~34)

# --- target (well-level pace; EUR is fit here, not sourced — see legs/context) ---
MIN_MONTHS = 6                    # a well needs >= this much production history to have a measured pace
DCA_MIN_MONTHS = 6               # minimum history to fit a decline curve for EUR
BOE_GAS_DIVISOR = 6.0            # gas->BOE convention (6 mcf ~= 1 boe); keep consistent everywhere

# --- splits ---
TEMPORAL_HOLDOUT = "2023-01-01"   # first_prod_date >= this -> test set (never touched in training)

# --- model / training ---
QUANTILES = (0.1, 0.5, 0.9)
SEED = 17
