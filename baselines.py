"""Statistical and classical machine-learning baselines (target space:
y_h = log e_{t0+h} - log e_{t0}). None of them sees data beyond t0.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from .data import SequenceSet


def persistence(S: SequenceSet) -> np.ndarray:
    """Random walk: e_{t0+h} = e_{t0}."""
    return np.zeros_like(S.y, dtype=float)


def drift(S: SequenceSet) -> np.ndarray:
    """Random walk with drift estimated from the input window."""
    T = S.le_window.shape[1]
    d = (S.le_window[:, -1] - S.le_window[:, 0]) / (T - 1)
    return d[:, None] * np.arange(1, S.y.shape[1] + 1)[None, :]


def ols_trend(S: SequenceSet) -> np.ndarray:
    """The fixed OLS trend extrapolation used as TPN's skip connection."""
    return S.trend.astype(float).copy()


def _damped_holt_one(y, H, alpha, beta, phi):
    level, trend = y[0], y[1] - y[0]
    sse = 0.0
    for t in range(1, len(y)):
        f = level + phi * trend
        sse += (y[t] - f) ** 2
        new_level = alpha * y[t] + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    steps = np.cumsum(phi ** np.arange(1, H + 1))
    return sse, level + steps * trend


_GRID = list(itertools.product([0.2, 0.4, 0.6, 0.8, 1.0], [0.05, 0.1, 0.2, 0.4], [0.8, 0.9, 0.98]))


def damped_trend(S: SequenceSet) -> np.ndarray:
    """Additive damped-trend exponential smoothing (Gardner, 1985) fitted per
    window by grid search on one-step-ahead squared error."""
    H = S.y.shape[1]
    out = np.zeros((len(S), H))
    for i, w in enumerate(S.le_window):
        best = min((_damped_holt_one(w, H, a, b, p) for a, b, p in _GRID), key=lambda r: r[0])
        out[i] = best[1] - w[-1]
    return out


def _flat(S: SequenceSet, Xs: np.ndarray) -> np.ndarray:
    return np.concatenate([Xs.reshape(len(S), -1), S.trend], axis=1)


class RidgeResidual:
    """Pooled ridge regression on the flattened (scaled) window, predicting the
    residual from the OLS trend. alpha chosen on the validation set."""

    def fit(self, S_tr, X_tr, S_va, X_va):
        best = None
        for a in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            m = Ridge(alpha=a).fit(_flat(S_tr, X_tr), S_tr.y - S_tr.trend)
            mae = np.abs(m.predict(_flat(S_va, X_va)) + S_va.trend - S_va.y).mean()
            if best is None or mae < best[0]:
                best = (mae, a, m)
        self.alpha, self.model = best[1], best[2]
        return self

    def predict(self, S, X):
        return self.model.predict(_flat(S, X)) + S.trend


class GBRTResidual:
    """Gradient-boosted trees (one model per horizon), residual from the trend,
    number of boosting iterations chosen on the validation set."""

    def fit(self, S_tr, X_tr, S_va, X_va, seed=0):
        self.models = []
        Ftr, Fva = _flat(S_tr, X_tr), _flat(S_va, X_va)
        for h in range(S_tr.y.shape[1]):
            best = None
            for it in [50, 100, 200, 400]:
                m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=it,
                                                  min_samples_leaf=10, l2_regularization=1.0,
                                                  random_state=seed)
                m.fit(Ftr, S_tr.y[:, h] - S_tr.trend[:, h])
                mae = np.abs(m.predict(Fva) + S_va.trend[:, h] - S_va.y[:, h]).mean()
                if best is None or mae < best[0]:
                    best = (mae, m)
            self.models.append(best[1])
        return self

    def predict(self, S, X):
        F = _flat(S, X)
        return np.stack([m.predict(F) for m in self.models], axis=1) + S.trend


STAT_BASELINES = {
    "Persistence": persistence,
    "Drift": drift,
    "OLS trend": ols_trend,
    "Damped trend": damped_trend,
}
