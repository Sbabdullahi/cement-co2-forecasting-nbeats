"""Data loading, provenance scoring, causal feature engineering, sequence
construction and purged temporal splits.

Leakage controls implemented here (each is unit-tested in selftest.py):
  L1  Every feature at year t uses observations from years <= t only.
  L2  No per-country statistic is computed over the full series (the original
      code used full-series country means, full-series z-scores, full-series
      tier assignment and backward-fill, all of which leak future values).
  L3  Targets are window-relative log changes, y_h = log e_{t0+h} - log e_{t0},
      so no scaling statistic is needed for the target at all.
  L4  Feature standardisation statistics are fitted on training sequences of
      the current fold only (FeatureScaler).
  L5  Splits are purged: the latest training target year precedes the earliest
      validation target year, and all validation targets are <= the
      information cut-off C, which precedes every test target.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config, FEATURES


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def _year_columns(df: pd.DataFrame, cfg: Config) -> Dict[int, object]:
    out = {}
    for c in df.columns:
        try:
            y = int(str(c).strip().split(".")[0])
        except ValueError:
            continue
        if cfg.year_min <= y <= cfg.year_max:
            out[y] = c
    return dict(sorted(out.items()))


def load_panel(source, cfg: Config, sheet: str = "X-Y") -> pd.DataFrame:
    """Read the wide Andrew (2024) style sheet into a long panel.

    Expected wide layout (as in the original script): columns ISO, Category,
    Activity and one column per year. Rows used:
      Category='Emissions',     Activity='Cement production'  -> emissions (kt CO2)
      Category='Activity data', Activity='Cement production'  -> cement production
      Category='Activity data', Activity='Clinker production' -> clinker production
    Returns columns: iso, year, emissions, production, clinker (NaN if missing).
    """
    df = source.copy() if isinstance(source, pd.DataFrame) else pd.read_excel(source, sheet_name=sheet)
    ycols = _year_columns(df, cfg)
    if not ycols:
        raise ValueError("No year columns between %d and %d found" % (cfg.year_min, cfg.year_max))

    def _row(sub, cat, act):
        r = sub[(sub["Category"] == cat) & (sub["Activity"] == act)]
        if r.empty:
            return np.full(len(ycols), np.nan)
        r = r.iloc[0]
        return pd.to_numeric(pd.Series([r[c] for c in ycols.values()]), errors="coerce").to_numpy(float)

    rows = []
    for iso, sub in df.groupby("ISO", sort=True):
        em = _row(sub, "Emissions", "Cement production")
        if np.all(np.isnan(em)):
            continue
        pr = _row(sub, "Activity data", "Cement production")
        ck = _row(sub, "Activity data", "Clinker production")
        rows.append(pd.DataFrame({"iso": str(iso), "year": list(ycols.keys()),
                                  "emissions": em, "production": pr, "clinker": ck}))
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["emissions"].notna() & (panel["emissions"] > 0)].reset_index(drop=True)
    return panel


_ISO = __import__("re").compile(r"^[A-Z]{3}$")


def _read_gcp_csv(path, cfg: Config, numeric: bool = True) -> pd.DataFrame:
    """Read one GCP-CEM CSV into long form (iso, year, value). Handles both
    layouts used in the release: countries in rows (ISO, UN code, Name, years...)
    and years in rows (Year, AFG, ALB, ...). Non-ISO columns (e.g. 'Kuwaiti Oil
    Fires') are ignored."""
    df = pd.read_csv(path)
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    if "ISO" in cols:
        ycols = {c: int(float(c)) for c in cols if c.replace(".0", "").isdigit()}
        long = df.melt(id_vars=["ISO"], value_vars=list(ycols), var_name="year", value_name="value")
        long["year"] = long["year"].map(ycols)
        long = long.rename(columns={"ISO": "iso"})
    elif cols[0].lower() == "year":
        iso_cols = [c for c in cols[1:] if _ISO.match(c)]
        long = df.melt(id_vars=[cols[0]], value_vars=iso_cols, var_name="iso", value_name="value")
        long = long.rename(columns={cols[0]: "year"})
    else:
        raise ValueError(f"Unrecognised layout in {path}: first columns {cols[:5]}")
    long["iso"] = long["iso"].astype(str).str.strip()
    long = long[long["iso"].str.match(_ISO)]
    long["year"] = pd.to_numeric(long["year"], errors="coerce")
    if numeric:
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
    else:
        long["value"] = long["value"].where(long["value"].notna(), None).astype(object)
        long["value"] = long["value"].map(lambda v: None if v is None or str(v).strip() in ("", "nan") else str(v).strip())
    long = long[(long["year"] >= cfg.year_min) & (long["year"] <= cfg.year_max)]
    return long.dropna(subset=["year"]).astype({"year": int})


def _unit_factor(ratio: pd.Series, lo: float, hi: float) -> float:
    """Power of ten that brings the median of a physically bounded ratio into [lo, hi]."""
    med = float(np.nanmedian(ratio.replace([np.inf, -np.inf], np.nan)))
    if not np.isfinite(med) or med <= 0:
        return 1.0
    for k in range(-9, 10):
        if lo <= med * 10.0 ** k <= hi:
            return 10.0 ** k
    return 1.0


def load_gcp_cem(folder: str, cfg: Config, verbose: bool = True) -> pd.DataFrame:
    """Load the Andrew GCP-CEM release from its CSV files in `folder`:
    '0. GCP-CEM.csv' (process emissions, kt CO2), '1. annual_cement_production.csv'
    and '2. annual_clinker_production.csv'. Production units are not stated in
    the files, so they are put on the emission scale by checking physical
    ratios: clinker emissions / clinker ~ 0.52 and clinker / cement ~ 0.5-0.95."""
    import glob
    def find(pattern, exclude="source", required=True):
        hits = [f for f in glob.glob(os.path.join(folder, "*")) if pattern.lower() in os.path.basename(f).lower()
                and (exclude is None or exclude not in os.path.basename(f).lower()) and f.lower().endswith(".csv")]
        if not hits:
            if required:
                raise FileNotFoundError(f"No file matching '*{pattern}*.csv' in {folder}")
            return None
        return sorted(hits)[0]
    e = _read_gcp_csv(find("GCP-CEM"), cfg).rename(columns={"value": "emissions"})
    p = _read_gcp_csv(find("cement_production"), cfg).rename(columns={"value": "production"})
    k = _read_gcp_csv(find("clinker_production"), cfg).rename(columns={"value": "clinker"})
    panel = e.merge(p, on=["iso", "year"], how="left").merge(k, on=["iso", "year"], how="left")
    both = panel.dropna(subset=["emissions", "clinker"])
    both = both[(both["emissions"] > 0) & (both["clinker"] > 0)]
    fk = _unit_factor(both["emissions"] / both["clinker"], 0.3, 1.0)          # E/K ~ 0.52
    panel["clinker"] = panel["clinker"] / fk
    bothp = panel.dropna(subset=["production", "clinker"])
    bothp = bothp[(bothp["production"] > 0) & (bothp["clinker"] > 0)]
    fp = _unit_factor(bothp["clinker"] / bothp["production"], 0.4, 1.2)       # K/P ~ 0.5-0.95
    panel["production"] = panel["production"] / fp
    if verbose:
        print(f"GCP-CEM loaded: {panel['iso'].nunique()} countries, {cfg.year_min}-{cfg.year_max}; "
              f"clinker divided by {fk:g}, cement production by {fp:g} to match emission units (kt).")
    cols = ["iso", "year", "emissions", "production", "clinker"]
    for name, pat in [("prod_src", "cement_production_source"), ("clk_src", "clinker_production_source")]:
        f = find(pat, exclude=None, required=False)
        if f:
            src = _read_gcp_csv(f, cfg, numeric=False).rename(columns={"value": name})
            panel = panel.merge(src, on=["iso", "year"], how="left")
            cols.append(name)
    if verbose and len(cols) > 5:
        print("Source files found: provenance will use the per-value source labels "
              "(run `diagnose` to review which labels count as estimates).")
    elif verbose:
        print("WARNING: no '*_source.csv' files found next to the data. Provenance falls back to the mere "
              "presence of production values, which marks nearly every year as reported. Add "
              "'1. annual_cement_production_source.csv' and '2. annual_clinker_production_source.csv'.")
    panel = panel[panel["emissions"].notna() & (panel["emissions"] > 0)]
    return panel[cols].sort_values(["iso", "year"]).reset_index(drop=True)


def load_any(source: str, cfg: Config, sheet: str = "X-Y") -> pd.DataFrame:
    """A folder with the GCP-CEM CSV files, or a wide Excel sheet (ISO/Category/Activity)."""
    if os.path.isdir(source):
        return load_gcp_cem(source, cfg)
    return load_panel(source, cfg, sheet)


def panel_hash(panel: pd.DataFrame) -> str:
    """SHA-256 of the panel contents, stored in the deployment artefact."""
    b = panel.sort_values(["iso", "year"]).to_csv(index=False).encode()
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# 2. Provenance (quality) scoring
# ---------------------------------------------------------------------------
def assign_provenance(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Clinker ratio and provenance score per country-year (row-wise, causal)."""
    p = panel.copy()
    lo, hi = cfg.clinker_bounds
    prod_ok = p["production"].notna() & (p["production"] > 0)
    clk_ok = p["clinker"].notna() & (p["clinker"] > 0)
    # values whose source label marks them as estimates do not count as reported
    if cfg.provenance_mode != "activity":
        import re
        rx = re.compile(cfg.estimate_label_regex, re.I)
        for col, mask_name in [("prod_src", "prod_ok"), ("clk_src", "clk_ok")]:
            if col in p:
                est = p[col].map(lambda s: bool(s) and bool(rx.search(str(s)))).astype(bool)
                if mask_name == "prod_ok":
                    prod_ok = prod_ok & ~est
                else:
                    clk_ok = clk_ok & ~est

    ratio = np.full(len(p), cfg.default_clinker_ratio)
    qual = np.full(len(p), cfg.q_default)
    src = np.array(["default"] * len(p), dtype=object)

    # q = 1.0 whenever clinker production is reported: the emission estimate then
    # rests on reported clinker activity data. The ratio itself needs cement
    # production too; without it the ratio is unknown and the default is used
    # (the year is flagged by prod_missing = 1).
    m_rep = clk_ok.to_numpy()
    both = (prod_ok & clk_ok).to_numpy()
    ratio[both] = np.clip(p.loc[both, "clinker"] / p.loc[both, "production"], lo, hi)
    qual[m_rep] = cfg.q_reported
    src[m_rep] = "reported"

    m_inf = (prod_ok & ~clk_ok).to_numpy()
    ratio[m_inf] = np.clip(p.loc[m_inf, "emissions"] /
                           (p.loc[m_inf, "production"] * cfg.clinker_emission_factor), lo, hi)
    qual[m_inf] = cfg.q_inferred
    src[m_inf] = "inferred"

    p["clinker_ratio"], p["quality"], p["source"] = ratio, qual, src
    return p


# ---------------------------------------------------------------------------
# 3. Causal per-country features
# ---------------------------------------------------------------------------
def _ols(y: np.ndarray):
    """Slope, intercept and R^2 of OLS on x = 0..n-1."""
    n = len(y)
    x = np.arange(n, dtype=float)
    if n < 2:
        return 0.0, float(y[-1]), 0.0
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - yhat) ** 2)) / ss_tot if ss_tot > 1e-12 else 0.0
    return float(b), float(a), float(np.clip(r2, 0.0, 1.0))


def consecutive_runs(years: np.ndarray) -> List[np.ndarray]:
    """Split sorted years into runs of consecutive years (indices)."""
    if len(years) == 0:
        return []
    breaks = np.where(np.diff(years) != 1)[0] + 1
    return np.split(np.arange(len(years)), breaks)


def country_features(dfc: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Features for one consecutive-year run of one country.

    Each row t is computed from rows <= t only (leakage control L1).
    'rel_level' is filled later per window because it depends on t0.
    """
    d = dfc.sort_values("year").reset_index(drop=True).copy()
    le = np.log(np.maximum(d["emissions"].to_numpy(float), cfg.emission_floor_kt))
    n = len(d)
    growth = np.zeros(n)
    growth[1:] = np.clip(np.diff(le), -1.0, 1.0)
    vol = np.zeros(n)
    for t in range(n):
        past = growth[max(1, t - 5):t]          # growth up to t-1 only
        vol[t] = float(np.std(past, ddof=1)) if len(past) >= 2 else 0.0
    slope, r2, dev = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        w = le[max(0, t - cfg.trend_window + 1):t + 1]
        b, a, r = _ols(w)
        slope[t], r2[t] = b, r
        dev[t] = le[t] - (a + b * (len(w) - 1))
    prod = d["production"].to_numpy(float)
    prod_ok = np.isfinite(prod) & (prod > 0)
    lp = np.where(prod_ok, np.log(np.where(prod_ok, prod, 1.0)), np.nan)
    log_int = np.where(prod_ok, le - lp, 0.0)
    pg = np.zeros(n)
    for t in range(1, n):
        if prod_ok[t] and prod_ok[t - 1]:
            pg[t] = np.clip(lp[t] - lp[t - 1], -1.0, 1.0)

    d["log_e"] = le
    d["log_level"] = le
    d["rel_level"] = 0.0
    d["growth"] = growth
    d["volatility"] = vol
    d["trend_slope"] = slope
    d["trend_r2"] = r2
    d["trend_dev"] = dev
    d["log_intensity"] = log_int
    d["prod_growth"] = pg
    d["prod_missing"] = (~prod_ok).astype(float)
    d["quality_x_ratio"] = d["clinker_ratio"] * d["quality"]
    d["year_norm"] = (d["year"] - cfg.year_min) / float(cfg.year_max - cfg.year_min)
    return d


def tier_of(max_kt: float, cfg: Config) -> int:
    return int(np.searchsorted(np.asarray(cfg.tier_thresholds_kt), max_kt, side="right"))


def trend_baseline(le_window: np.ndarray, cfg: Config) -> np.ndarray:
    """OLS on the last `trend_window` log emissions, extrapolated h=1..H and
    expressed relative to log e_{t0} (same space as the target)."""
    w = le_window[-cfg.trend_window:]
    b, a, _ = _ols(w)
    last = len(w) - 1
    return np.array([a + b * (last + h) for h in range(1, cfg.horizon + 1)]) - le_window[-1]


# ---------------------------------------------------------------------------
# 4. Sequences
# ---------------------------------------------------------------------------
_FIELDS = ["X", "y", "q", "trend", "tier", "iso", "t0", "e_t0", "e_future", "le_window", "scale_kt"]


@dataclass
class SequenceSet:
    X: np.ndarray            # (N,T,F) raw (unscaled) features
    y: np.ndarray            # (N,H)   log e_{t0+h} - log e_{t0}
    q: np.ndarray            # (N,T)   raw provenance scores
    trend: np.ndarray        # (N,H)   trend baseline in target space
    tier: np.ndarray         # (N,)
    iso: np.ndarray          # (N,)
    t0: np.ndarray           # (N,)   year of last input observation
    e_t0: np.ndarray         # (N,)   emissions at t0 (kt)
    e_future: np.ndarray     # (N,H)  observed future emissions (kt)
    le_window: np.ndarray    # (N,T)  log emissions in window
    scale_kt: np.ndarray     # (N,)   in-window naive one-step MAE (kt), MASE scale
    features: List[str] = field(default_factory=lambda: list(FEATURES))

    def __len__(self):
        return len(self.y)

    @property
    def high(self) -> np.ndarray:
        """True if the window contains at least one non-default year."""
        return (self.q > 0.1 + 1e-9).any(axis=1)

    def subset(self, idx) -> "SequenceSet":
        idx = np.asarray(idx)
        return SequenceSet(*(getattr(self, k)[idx] for k in _FIELDS), features=self.features)

    @staticmethod
    def concat(sets: List["SequenceSet"]) -> "SequenceSet":
        return SequenceSet(*[np.concatenate([getattr(s, k) for s in sets]) for k in _FIELDS])


def window_arrays(feat: pd.DataFrame, end: int, cfg: Config):
    """Build the (T,F) feature block for a window ending at row `end`."""
    T = cfg.seq_len
    blk = feat.iloc[end - T + 1:end + 1]
    le_w = blk["log_e"].to_numpy(float)
    X = blk[FEATURES].to_numpy(float).copy()
    X[:, FEATURES.index("rel_level")] = le_w - le_w[-1]
    q = blk["quality"].to_numpy(float).copy()
    e_w = blk["emissions"].to_numpy(float)
    tier = tier_of(float(e_w.max()), cfg)
    tb = trend_baseline(le_w, cfg)
    scale = float(np.mean(np.abs(np.diff(e_w))))
    scale = max(scale, 0.01 * float(np.mean(e_w)), 1e-6)   # floor for flat windows
    return X, q, le_w, tb, tier, scale


def build_sequences(panel: pd.DataFrame, cfg: Config, require_future: bool = True) -> SequenceSet:
    T, H = cfg.seq_len, cfg.horizon
    P = assign_provenance(panel, cfg) if "quality" not in panel else panel
    out = {k: [] for k in _FIELDS}
    for iso, dc in P.groupby("iso", sort=True):
        dc = dc.sort_values("year").reset_index(drop=True)
        for run in consecutive_runs(dc["year"].to_numpy()):
            seg = dc.iloc[run].reset_index(drop=True)
            if len(seg) < T + (H if require_future else 0):
                continue
            feat = country_features(seg, cfg)
            last_end = len(seg) - 1 - (H if require_future else 0)
            for end in range(T - 1, last_end + 1):
                X, q, le_w, tb, tier, scale = window_arrays(feat, end, cfg)
                if require_future:
                    fut = feat["log_e"].to_numpy(float)[end + 1:end + 1 + H]
                    y = fut - le_w[-1]
                    efut = feat["emissions"].to_numpy(float)[end + 1:end + 1 + H]
                else:
                    y = np.full(H, np.nan); efut = np.full(H, np.nan)
                if not (np.isfinite(X).all() and (np.isfinite(y).all() or not require_future)):
                    continue
                out["X"].append(X); out["y"].append(y); out["q"].append(q)
                out["trend"].append(tb); out["tier"].append(tier); out["iso"].append(iso)
                out["t0"].append(int(feat["year"].iloc[end])); out["e_t0"].append(float(feat["emissions"].iloc[end]))
                out["e_future"].append(efut); out["le_window"].append(le_w); out["scale_kt"].append(scale)
    if not out["X"]:
        raise ValueError("No sequences could be built (need >= T+H consecutive years).")
    return SequenceSet(
        X=np.asarray(out["X"], np.float32), y=np.asarray(out["y"], np.float32),
        q=np.asarray(out["q"], np.float32), trend=np.asarray(out["trend"], np.float32),
        tier=np.asarray(out["tier"], int), iso=np.asarray(out["iso"]), t0=np.asarray(out["t0"], int),
        e_t0=np.asarray(out["e_t0"], float), e_future=np.asarray(out["e_future"], float),
        le_window=np.asarray(out["le_window"], float), scale_kt=np.asarray(out["scale_kt"], float))


# ---------------------------------------------------------------------------
# 5. Purged temporal splits
# ---------------------------------------------------------------------------
def make_fold(S: SequenceSet, cutoff: int, cfg: Config, last_fold: bool) -> Dict[str, np.ndarray]:
    H, V = cfg.horizon, cfg.val_span
    val_lo, val_hi = cutoff - H - V + 1, cutoff - H
    first_val_target = val_lo + 1
    train = np.where(S.t0 + H < first_val_target)[0]
    val = np.where((S.t0 >= val_lo) & (S.t0 <= val_hi))[0]
    test_hi = S.t0.max() if last_fold else cutoff + cfg.test_span - 1
    test = np.where((S.t0 >= cutoff) & (S.t0 <= test_hi))[0]
    fold = {"train": train, "val": val, "test": test, "cutoff": cutoff}
    check_fold(S, fold, cfg)
    return fold


def check_fold(S: SequenceSet, fold, cfg: Config):
    """Hard assertions for leakage control L5."""
    H = cfg.horizon
    tr, va, te, C = fold["train"], fold["val"], fold["test"], fold["cutoff"]
    if len(tr) == 0 or len(va) == 0 or len(te) == 0:
        raise ValueError(f"Empty split at cutoff {C}: train={len(tr)} val={len(va)} test={len(te)}")
    assert (S.t0[tr] + H).max() < (S.t0[va] + 1).min(), "train/val target overlap"
    assert (S.t0[va] + H).max() <= C, "validation target beyond cut-off"
    assert (S.t0[te] + 1).min() > C, "test target not after cut-off"
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def _fold_sizes(S: SequenceSet, C: int, cfg: Config, last: bool):
    H, V = cfg.horizon, cfg.val_span
    val_lo = C - H - V + 1
    n_tr = int((S.t0 + H < val_lo + 1).sum())
    n_va = int(((S.t0 >= val_lo) & (S.t0 <= C - H)).sum())
    hi = S.t0.max() if last else C + cfg.test_span - 1
    n_te = int(((S.t0 >= C) & (S.t0 <= hi)).sum())
    return n_tr, n_va, n_te


def auto_cutoffs(S: SequenceSet, cfg: Config) -> tuple:
    """Choose up to cfg.n_folds cut-off years from the latest data backwards so
    that each fold has >= min_test_windows test windows and >= min_train_windows
    training windows. Depends only on the years covered (t0), never on targets
    or model results, so it can be fixed before any model is trained."""
    cuts = []
    C = int(S.t0.max())
    while C >= int(S.t0.min()) and len(cuts) < cfg.n_folds:
        last = not cuts
        n_tr, n_va, n_te = _fold_sizes(S, C, cfg, last)
        if n_tr < cfg.min_train_windows:
            break
        if n_te >= cfg.min_test_windows and n_va > 0:
            cuts.append(C)
            C -= cfg.test_span
        else:
            C -= 1
    if not cuts:
        raise ValueError("Not enough data for even one fold with the minimum sizes; "
                         "lower min_test_windows/min_train_windows or check the data coverage.")
    cuts = sorted(cuts)
    # all but the last fold use a fixed test span; make sure spans do not overlap
    return tuple(cuts)


def source_label_report(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Distinct source labels of production and clinker values (1990 onwards),
    with counts and whether the estimate pattern classifies them as estimates."""
    import re
    rx = re.compile(cfg.estimate_label_regex, re.I)
    rows = []
    for col, what in [("prod_src", "cement production"), ("clk_src", "clinker production")]:
        if col not in panel:
            continue
        vc = panel[col].dropna().value_counts()
        for lab, n in vc.items():
            rows.append({"series": what, "source_label": lab, "n_values": int(n),
                         "treated_as": "estimate" if rx.search(str(lab)) else "reported"})
    return pd.DataFrame(rows)


def coverage_report(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Per-country first/last year, gaps and longest consecutive run of positive
    emissions; explains why a country contributes few or no windows."""
    rows = []
    for iso, d in panel.groupby("iso"):
        yrs = np.sort(d["year"].to_numpy())
        runs = consecutive_runs(yrs)
        longest = max(runs, key=len)
        rows.append({"iso": iso, "first": int(yrs[0]), "last": int(yrs[-1]), "n_years": len(yrs),
                     "gaps": len(runs) - 1, "longest_run": f"{yrs[longest[0]]}-{yrs[longest[-1]]}",
                     "windows": sum(max(0, len(r) - cfg.seq_len - cfg.horizon + 1) for r in runs),
                     "pct_with_production": round(100 * float(d["production"].notna().mean()), 1),
                     "pct_with_clinker": round(100 * float(d["clinker"].notna().mean()), 1)})
    return pd.DataFrame(rows).sort_values("last")


# ---------------------------------------------------------------------------
# 6. Feature scaler fitted on training sequences only (L4)
# ---------------------------------------------------------------------------
class FeatureScaler:
    def fit(self, X: np.ndarray):
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(0)
        sd = flat.std(0)
        self.scale_ = np.where(sd > 1e-8, sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def state(self):
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_state(cls, st):
        s = cls(); s.mean_ = np.asarray(st["mean"]); s.scale_ = np.asarray(st["scale"]); return s


def describe(S: SequenceSet, panel: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    """Descriptive counts reported in the manuscript (all generated, never typed)."""
    P = assign_provenance(panel, cfg) if "quality" not in panel else panel
    used = set(S.iso.tolist())
    Pu = P[P["iso"].isin(used)]
    return {
        "n_countries": len(used),
        "n_sequences": len(S),
        "n_high": int(S.high.sum()),
        "n_low": int((~S.high).sum()),
        "pct_reported": 100 * float((Pu["source"] == "reported").mean()),
        "pct_inferred": 100 * float((Pu["source"] == "inferred").mean()),
        "pct_default": 100 * float((Pu["source"] == "default").mean()),
        "first_year": int(Pu["year"].min()),
        "last_year": int(Pu["year"].max()),
    }
