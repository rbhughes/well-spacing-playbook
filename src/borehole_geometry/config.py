"""Project-wide constants. Tune later; these are sane starting points (see docs/RECIPE.md)."""

from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

# --- study scope ---
# Alberta only. SK/BC dropped 2026-08-23: SK publishes 2-point 2-D sticks with no directional
# survey and no production volumes. See data/README.md.
PROVINCES = ["AB"]

# The free Petrinex volumetric archive is a rolling ~5-year window (2022-01 .. 2026-07 as of
# 2026-08-23). A well that came on stream before the window opens has its early, highest-rate
# production truncated, so its pace is not comparable to a well captured from month one.
# FIRST_PROD_MIN must therefore not precede VOLUMETRIC_WINDOW_START.
VOLUMETRIC_WINDOW_START = "2022-01-01"   # re-check with: fetch_data.py probe_vol
FIRST_PROD_MIN = "2022-01-01"

assert FIRST_PROD_MIN >= VOLUMETRIC_WINDOW_START, (
    "FIRST_PROD_MIN precedes the public volumetric window: those wells' early production is "
    "unobservable, so their pace target would be measured from a truncated history."
)

# --- PPDM identity ---
# UWI is the Petrinex WellIdentifier (e.g. "100010100115W400" -- the trailing two digits are the
# well EVENT sequence, which is the granularity production is reported at). Chosen over the ST37
# UWI label because production is the scarcer join: a geometry row with no production is useless,
# so the key that production is published under wins. ST37's label and the AER licence number are
# carried in WELL_ALIAS instead, so nothing is lost.
UWI_SOURCE_FIELD = "WellIdentifier"          # Petrinex Well Infrastructure column

# PPDM SOURCE: "the individual, company, state, or government agency designated as the source of
# information for this row" -- it is per-ROW provenance, and it is part of the primary key of
# WELL_DIR_SRVY, WELL_DIR_SRVY_STATION and PDEN. Checked 2026-08-24: neither the Petrinex Terms
# of Use (petrinex.ca/terms) nor the linked disclaimer prescribes a value for a field like this.
# The terms require only that Crown copyright (Government of Alberta) be acknowledged. So these
# are our own short labels, split by actual provider rather than lumped together.
SOURCE_PETRINEX = "PETRINEX"                 # Well Infrastructure + Volumetrics
SOURCE_AER = "AER"                           # ST37 shapefiles (AER, not Petrinex)

# --- geometry ---
MIN_LEG_M = 150.0                 # ignore stub legs shorter than this
NEIGHBOR_RADIUS_M = 2000.0        # candidate-pair search radius (grid-cell size)
GRID_CELL_M = 2000.0
MAX_NEIGHBOR_WELLS = 12           # neighbor-set encoder input cap
MAX_OWN_LEGS = 24                 # own-leg set encoder input cap (fishbones reach ~34)

# --- geometry: leg extraction from surveys ---
LANDING_TVD_TOL_M = 25.0          # a leg "lands" at the first station within this of its max TVD;
                                  # the lateral is everything from there to the toe

# --- target normalization (Phase 1) ---
# Peer group for the P50 reference: (play, lateral-length class). Chosen 2026-08-25: tight
# enough to compare like with like, loose enough not to divide away the interference signal.
# Length classes are on TOTAL lateral length per well (all legs summed) — a 30-leg fishbone is
# a different animal from a single 2 km lateral even at the same per-leg length.
LENGTH_CLASS_EDGES_M = (1000.0, 2000.0, 4000.0, 8000.0)   # 5 classes: <1k,1-2k,2-4k,4-8k,>=8k
MIN_PEER_WELLS = 20               # a (play, class) cell smaller than this falls back to the
                                  # class-only P50, then to the global P50
PACE_MONTHS = 12                  # pace = BOE in the first N calendar months, annualized
WINSOR_PCTL = 0.99                # target cap percentile for BASELINE training (decided
                                  # 2026-08-25): raw targets stay in cohort.parquet; the
                                  # winsorized column exists because ridge/GBM baselines use
                                  # squared error and one 3,000% freak well would steer them,
                                  # making the must-beat comparison unfairly easy. The pinball
                                  # model trains on the RAW target. Cap computed on TRAIN-ERA
                                  # wells only so the holdout cannot set the cap.

# --- target (well-level pace; EUR is fit here, not sourced — see legs/context) ---
MIN_MONTHS = 6                    # a well needs >= this much production history to have a measured pace
DCA_MIN_MONTHS = 6               # minimum history to fit a decline curve for EUR
BOE_GAS_DIVISOR = 6.0            # gas->BOE convention (6 mcf ~= 1 boe); keep consistent everywhere

# --- splits ---
# With only 2022-01 onward available, the holdout boundary also decides how much history the test
# wells can possibly have. Wells first producing in 2025 carry <= ~19 months; pushing this later
# buys recency at the cost of a noisier target on exactly the wells being scored.
TEMPORAL_HOLDOUT = "2025-01-01"   # first_prod_date >= this -> test set (never touched in training)

# --- model / training ---
QUANTILES = (0.1, 0.5, 0.9)
SEED = 17
