"""Central configuration.

Every numerical constant used anywhere in the pipeline is defined here so that
the manuscript can cite one place, and so that sensitivity runs only need to
change this file (or pass overrides on the command line).
"""
from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class Config:
    # ---- data / sequence construction -------------------------------------
    year_min: int = 1990
    year_max: int = 2023
    seq_len: int = 8            # T: input window (years)
    horizon: int = 3            # H: forecast horizon (years ahead)
    trend_window: int = 5       # years used for the OLS trend baseline
    emission_floor_kt: float = 0.1   # floor before taking logs (kt CO2)

    # ---- clinker-ratio provenance ------------------------------------------
    # IPCC (2006) Vol.3 Ch.2 Tier 1 default clinker fraction when nothing is
    # known about the cement mix. NOTE: the original submission's code used
    # 0.70 while the text stated 0.75; 0.75 is used here and is configurable.
    default_clinker_ratio: float = 0.75
    # IPCC (2006) default emission factor for clinker (t CO2 / t clinker),
    # used only to infer a clinker ratio from emissions and cement production
    # when clinker production is not reported.
    clinker_emission_factor: float = 0.52
    clinker_bounds: Tuple[float, float] = (0.50, 0.95)
    q_reported: float = 1.0     # clinker production reported
    q_inferred: float = 0.5     # ratio inferred from emissions / cement production
    q_default: float = 0.1      # no production data: default ratio assigned
    # Provenance from the release's per-value source files ('..._source.csv'):
    # a production or clinker value whose source label matches this pattern is an
    # estimate (interpolated, extrapolated, modelled, ...) and does not count as
    # reported. 'auto' uses the source files when present, 'activity' ignores them.
    provenance_mode: str = "auto"
    estimate_label_regex: str = r"estimat|interpol|extrapol|assum|fill|proxy|model|regional|gnr|default|derived|calculat|scaled|copied|carried|trend|guess|average"

    # Emission-scale tiers (kt CO2, max over the input window). These are a
    # coarse proxy for industrial scale, NOT an official development class.
    tier_thresholds_kt: Tuple[float, ...] = (1000.0, 5000.0, 15000.0)

    # ---- evaluation protocol -----------------------------------------------
    # Information cut-off years C for rolling-origin evaluation. For each C:
    #   train : origins t0 with all targets < first validation target year
    #   val   : origins t0 in [C-H-val_span+1, C-H]  (targets <= C)
    #   test  : origins t0 in [C, C+test_span-1] (last fold: all t0 >= C)
    # Empty tuple = choose cut-offs automatically from the data (data.auto_cutoffs),
    # so that every fold has at least min_test_windows test windows. Fixed years
    # can be given on the command line with --cutoffs.
    cutoffs: Tuple[int, ...] = ()
    n_folds: int = 3
    min_test_windows: int = 15
    min_train_windows: int = 40
    min_calibration_windows: int = 30   # deployment: calibration set grows back in time until this size
    val_span: int = 3
    test_span: int = 3
    min_stratum_for_mondrian: int = 15
    conformal_alphas: Tuple[float, ...] = (0.5, 0.3, 0.2, 0.1, 0.05)

    # ---- training ----------------------------------------------------------
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    # Model at the centre of the analysis: significance tests, calibration,
    # cold start and deployment are reported for it against all others.
    focal_model: str = "N-BEATS"
    hidden: int = 32
    dropout: float = 0.3
    lr: float = 3e-3
    weight_decay: float = 1e-2
    batch_size: int = 16
    max_epochs: int = 300
    patience: int = 30
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    proto_momentum: float = 0.9

    # ---- statistics --------------------------------------------------------
    n_boot: int = 2000
    n_perm_repeats: int = 20
    denial_fractions: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    denial_trials: int = 10
    coldstart_folds: int = 4

    # optional CSV with columns ISO, region, income_group (World Bank) used to
    # build a static geographic/economic graph for the graph baseline
    country_meta_csv: str = ""

    def to_dict(self):
        return asdict(self)


# Feature groups (window-level features; all strictly causal, see data.py)
FEATURES: List[str] = [
    # emission history
    "log_level",        # log emissions (absolute scale context)
    "rel_level",        # log e_t - log e_{t0}  (window-relative level)
    "growth",           # log e_t - log e_{t-1}
    "volatility",       # std of growth over t-5..t-1 (lagged)
    # trend (rolling OLS over <= trend_window past years, ending at t)
    "trend_slope",
    "trend_r2",
    "trend_dev",
    # production
    "log_intensity",    # log(e_t / p_t)
    "prod_growth",      # log p_t - log p_{t-1}
    "prod_missing",     # 1 if cement production not reported
    # clinker / provenance
    "clinker_ratio",
    "quality",
    "quality_x_ratio",
    # calendar
    "year_norm",
]

FEATURE_GROUPS = {
    "emission history": ["log_level", "rel_level", "growth", "volatility"],
    "trend": ["trend_slope", "trend_r2", "trend_dev"],
    "production": ["log_intensity", "prod_growth", "prod_missing"],
    "clinker and provenance": ["clinker_ratio", "quality", "quality_x_ratio"],
    "calendar": ["year_norm"],
}
TREND_FEATURES = FEATURE_GROUPS["trend"]
QUALITY_FEATURES = FEATURE_GROUPS["clinker and provenance"]
