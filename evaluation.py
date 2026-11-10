"""Metrics, uncertainty calibration, significance testing and diagnostics."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import stats

from .config import Config, FEATURES, FEATURE_GROUPS
from .data import SequenceSet


# ---------------------------------------------------------------------------
# Point-forecast metrics (physical units and scale-free)
# ---------------------------------------------------------------------------
def to_kt(S: SequenceSet, pred: np.ndarray) -> np.ndarray:
    """Back-transform to kt CO2 (median forecast under a log-scale model)."""
    return S.e_t0[:, None] * np.exp(pred)


def per_sequence_mase(S: SequenceSet, pred: np.ndarray) -> np.ndarray:
    err = np.abs(to_kt(S, pred) - S.e_future)
    return (err / S.scale_kt[:, None]).mean(1)


def metrics(S: SequenceSet, pred: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    m = np.ones(len(S), bool) if mask is None else mask
    if m.sum() == 0:
        return {k: np.nan for k in ["n", "MASE", "sMAPE", "MAE_kt", "RMSE_kt", "MAE_log", "R2_log"]}
    ek, ef = to_kt(S, pred)[m], S.e_future[m]
    err = ek - ef
    y, p = S.y[m], pred[m]
    r2 = []
    for h in range(y.shape[1]):
        sst = np.sum((y[:, h] - y[:, h].mean()) ** 2)
        r2.append(1 - np.sum((y[:, h] - p[:, h]) ** 2) / sst if sst > 0 and m.sum() > 2 else np.nan)
    return {
        "n": int(m.sum()),
        "MASE": float((np.abs(err) / S.scale_kt[m][:, None]).mean()),
        "sMAPE": float((200 * np.abs(err) / (np.abs(ek) + np.abs(ef))).mean()),
        "MAE_kt": float(np.abs(err).mean()),
        "RMSE_kt": float(np.sqrt((err ** 2).mean())),
        "MAE_log": float(np.abs(y - p).mean()),
        "R2_log": float(np.nanmean(r2)),
    }


def metrics_by_stratum(S: SequenceSet, pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    return {"all": metrics(S, pred), "high": metrics(S, pred, S.high), "low": metrics(S, pred, ~S.high)}


# ---------------------------------------------------------------------------
# Split-conformal intervals, Mondrian by provenance stratum
# ---------------------------------------------------------------------------
def conformal_fit(S_cal: SequenceSet, pred_cal: np.ndarray, cfg) -> Dict:
    """Absolute-residual quantiles per (stratum, horizon, alpha), computed on
    the calibration (validation) set only. A stratum with fewer than
    cfg.min_stratum_for_mondrian calibration sequences falls back to pooled."""
    r = np.abs(S_cal.y - pred_cal)
    out = {}
    for name, mask in [("all", np.ones(len(S_cal), bool)), ("high", S_cal.high), ("low", ~S_cal.high)]:
        small = name != "all" and mask.sum() < cfg.min_stratum_for_mondrian
        use = np.ones(len(S_cal), bool) if small else mask
        n = int(use.sum())
        for h in range(r.shape[1]):
            srt = np.sort(r[use, h])
            for a in cfg.conformal_alphas:
                k = math.ceil((n + 1) * (1 - a))
                out[(name, h, a)] = float(srt[k - 1]) if k <= n else float("inf")
        out[(name, "n")] = n
        out[(name, "pooled_fallback")] = bool(small)
    return out


def conformal_interval(S: SequenceSet, pred: np.ndarray, qd: Dict, alpha: float, mondrian: bool = True):
    lo, hi = np.zeros_like(pred), np.zeros_like(pred)
    for i in range(len(S)):
        st = ("high" if S.high[i] else "low") if mondrian else "all"
        for h in range(pred.shape[1]):
            w = qd[(st, h, alpha)]
            lo[i, h], hi[i, h] = pred[i, h] - w, pred[i, h] + w
    return lo, hi


def gaussian_interval(pred, sd, alpha):
    z = stats.norm.ppf(1 - alpha / 2)
    return pred - z * sd, pred + z * sd


def interval_scores(S: SequenceSet, lo, hi, alpha, mask=None) -> Dict[str, float]:
    m = np.ones(len(S), bool) if mask is None else mask
    y, l, u = S.y[m], lo[m], hi[m]
    if len(y) == 0:
        return {"coverage": np.nan, "width_pct": np.nan, "interval_score": np.nan, "n": 0}
    cover = ((y >= l) & (y <= u)).mean()
    finite = np.isfinite(l) & np.isfinite(u)
    with np.errstate(over="ignore", invalid="ignore"):
        width_pct = 100 * np.median(np.exp(u) - np.exp(l))      # % of e_t0 (inf if any interval unbounded)
        isc = np.where(finite, (u - l) + (2 / alpha) * np.maximum(l - y, 0) + (2 / alpha) * np.maximum(y - u, 0), np.inf)
    return {"coverage": float(cover), "width_pct": float(width_pct),
            "interval_score": float(isc.mean()), "n": int(m.sum()),
            "pct_unbounded": float(100 * (~finite).mean())}


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------
def cluster_bootstrap_diff(loss_a: np.ndarray, loss_b: np.ndarray, clusters: np.ndarray,
                           n_boot: int, seed: int = 0) -> Dict[str, float]:
    """Mean of (loss_a - loss_b), with a percentile CI from resampling whole
    countries (clusters). Overlapping windows from one country are dependent,
    so sequences must not be resampled individually."""
    d = loss_a - loss_b
    groups = [np.where(clusters == c)[0] for c in np.unique(clusters)]
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, len(groups), len(groups))
        boots[b] = d[np.concatenate([groups[k] for k in pick])].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(min(1.0, max(p, 1.0 / n_boot)))}


def dm_hln(loss_a: np.ndarray, loss_b: np.ndarray, t0: np.ndarray, h: int) -> Dict[str, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold correction on the
    series of cross-sectional mean loss differentials indexed by origin year."""
    years = np.unique(t0)
    d = np.array([(loss_a[t0 == y] - loss_b[t0 == y]).mean() for y in years])
    n = len(d)
    if n < 4:
        return {"stat": np.nan, "p": np.nan, "n_origins": n}
    dc = d - d.mean()
    gam = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h)]
    var = gam[0] + 2 * sum(gam[1:])
    if var <= 0:
        return {"stat": np.nan, "p": np.nan, "n_origins": n}
    dm = d.mean() / math.sqrt(var / n)
    k = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    stat = k * dm
    return {"stat": float(stat), "p": float(2 * stats.t.sf(abs(stat), df=n - 1)), "n_origins": n}


def holm(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, float)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


# ---------------------------------------------------------------------------
# Interpretability and sensitivity
# ---------------------------------------------------------------------------
def permutation_importance(predict_fn: Callable, S: SequenceSet, Xs: np.ndarray, cfg: Config,
                           seed: int = 0, groups: Optional[Dict[str, List[str]]] = None) -> Dict[str, np.ndarray]:
    """Increase in MASE when a feature group is permuted across test sequences
    (whole T-step blocks are moved, preserving within-window structure).
    The provenance group also permutes the q vector used by the attention bias."""
    groups = groups or FEATURE_GROUPS
    base = per_sequence_mase(S, predict_fn(Xs, S.q)).mean()
    rng = np.random.RandomState(seed)
    out = {}
    for g, feats in groups.items():
        cols = [FEATURES.index(f) for f in feats]
        vals = []
        for _ in range(cfg.n_perm_repeats):
            perm = rng.permutation(len(S))
            Xp = Xs.copy()
            Xp[:, :, cols] = Xs[perm][:, :, cols]
            qp = S.q[perm] if "quality" in feats else S.q
            vals.append(per_sequence_mase(S, predict_fn(Xp, qp)).mean() - base)
        out[g] = np.asarray(vals)
    return out


def deny_provenance(S: SequenceSet, frac: float, cfg: Config, rng: np.random.RandomState):
    """Counterfactual input in which a fraction of the non-default years of each
    window lose their production/clinker information and are represented
    exactly as the pipeline represents default years. Emissions unchanged."""
    X, q = S.X.copy(), S.q.copy()
    ci = {f: FEATURES.index(f) for f in FEATURES}
    for i in range(len(S)):
        cand = np.where(q[i] > 0.1 + 1e-9)[0]
        if len(cand) == 0 or frac <= 0:
            continue
        k = int(round(frac * len(cand)))
        if k == 0:
            continue
        for t in rng.choice(cand, k, replace=False):
            q[i, t] = cfg.q_default
            X[i, t, ci["quality"]] = cfg.q_default
            X[i, t, ci["clinker_ratio"]] = cfg.default_clinker_ratio
            X[i, t, ci["quality_x_ratio"]] = cfg.default_clinker_ratio * cfg.q_default
            X[i, t, ci["prod_missing"]] = 1.0
            X[i, t, ci["log_intensity"]] = 0.0
            X[i, t, ci["prod_growth"]] = 0.0
            if t + 1 < X.shape[1]:
                X[i, t + 1, ci["prod_growth"]] = 0.0
    return X, q


def attention_by_provenance(attn: np.ndarray, S: SequenceSet) -> Dict[str, float]:
    """Mean attention weight on non-default vs default years, restricted to
    windows that contain both kinds (otherwise the comparison is undefined)."""
    nd, d = [], []
    for a, q in zip(attn, S.q):
        m = q > 0.1 + 1e-9
        if m.any() and (~m).any():
            nd.append(a[m].mean()); d.append(a[~m].mean())
    return {"n_mixed_windows": len(nd),
            "mean_attn_nondefault": float(np.mean(nd)) if nd else np.nan,
            "mean_attn_default": float(np.mean(d)) if d else np.nan,
            "mean_attn_by_position": attn.mean(0).tolist()}
