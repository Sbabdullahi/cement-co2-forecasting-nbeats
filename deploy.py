"""Deployment: train the final ensemble, save a self-describing artefact, and
serve forecasts for one country's history with input checks and warnings."""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .config import Config, FEATURES
from .data import (FeatureScaler, assign_provenance, build_sequences, consecutive_runs,
                   country_features, panel_hash, window_arrays)
from .evaluation import conformal_fit
from .train import fit, make_model, predict, ABLATIONS

ARTEFACT_VERSION = "1.0"


def _build(model: str, variant: str, cfg: Config):
    if model == "TPN":
        return make_model("TPN", cfg, len(FEATURES), **ABLATIONS[variant])
    return make_model(model, cfg, len(FEATURES))


def train_final(panel_raw: pd.DataFrame, cfg: Config, out: str, backtest_json: Optional[str] = None,
                variant: str = "TPN (full)", model: str = "N-BEATS") -> str:
    """Train on all origins except the most recent `val_span` years, which are
    kept for early stopping and conformal calibration (purged as in backtests)."""
    os.makedirs(out, exist_ok=True)
    panel = assign_provenance(panel_raw, cfg)
    S = build_sequences(panel, cfg)
    H, V = cfg.horizon, cfg.val_span
    val_hi = int(S.t0.max()); val_lo = val_hi - V + 1
    # Countries' records end in different years, so the latest V origins can hold
    # very few windows. Move the calibration start back until it holds at least
    # min_calibration_windows windows (training stays purged before it).
    while (S.t0 >= val_lo).sum() < cfg.min_calibration_windows and val_lo > S.t0.min() + H + 1:
        val_lo -= 1
    va = np.where(S.t0 >= val_lo)[0]
    tr = np.where(S.t0 + H < val_lo + 1)[0]
    if len(tr) < cfg.min_train_windows:
        raise ValueError(f"Only {len(tr)} training windows before calibration start {val_lo}.")
    S_tr, S_va = S.subset(tr), S.subset(va)
    scaler = FeatureScaler().fit(S_tr.X)
    X_tr, X_va = scaler.transform(S_tr.X), scaler.transform(S_va.X)
    states, val_preds = [], []
    for seed in cfg.seeds:
        kw = ABLATIONS[variant] if model == "TPN" else {}
        m = _build(model, variant, cfg)
        m, _ = fit(m, S_tr, X_tr, S_va, X_va, cfg, seed,
                   quality_weight=(model == "TPN") and not kw.get("_no_qweight", False))
        states.append({k: v.cpu() for k, v in m.state_dict().items()})
        val_preds.append(predict(m, S_va, X_va)["mean"])
    qd = conformal_fit(S_va, np.mean(val_preds, 0), cfg)
    g = S_tr.X[:, :, FEATURES.index("growth")].ravel()
    lvl = S_tr.X[:, :, FEATURES.index("log_level")]
    art = {
        "artefact_version": ARTEFACT_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "config": cfg.to_dict(), "features": FEATURES, "scaler": scaler.state(),
        "states": states, "variant": variant, "model": model,
        "conformal": {"|".join(map(str, k)): v for k, v in qd.items()},
        "training_envelope": {
            "log_level_min": float(lvl.min()), "log_level_max": float(lvl.max()),
            "growth_p01": float(np.percentile(g, 1)), "growth_p99": float(np.percentile(g, 99)),
            "last_data_year": int(panel["year"].max()),
            "train_origins": [int(S_tr.t0.min()), int(S_tr.t0.max())],
            "calibration_origins": [val_lo, val_hi],
            "n_train": int(len(S_tr)), "n_calibration": int(len(S_va)),
            "calibration_high_pooled_fallback": bool(qd[("high", "pooled_fallback")]),
        },
        "data_sha256": panel_hash(panel_raw),
        "countries": sorted(set(S.iso.tolist())),
    }
    path = os.path.join(out, "tpn_artifact.pt")
    torch.save(art, path)
    write_model_card(art, out, backtest_json)
    return path


def write_model_card(art: Dict, out: str, backtest_json: Optional[str]):
    env = art["training_envelope"]
    bt = ""
    if backtest_json and os.path.exists(backtest_json):
        R = json.load(open(backtest_json))
        mk = art.get("model", "TPN")
        key = mk if mk in R["main"] else "TPN"
        t, p = R["main"][key]["all"], R["main"]["Persistence"]["all"]
        bt = (f"- Rolling-origin backtest (pooled test, n={t['n']}): {key} MASE {t['MASE']:.3f}, "
              f"sMAPE {t['sMAPE']:.1f}%, MAE {t['MAE_kt']:.0f} kt; persistence MASE {p['MASE']:.3f}, sMAPE {p['sMAPE']:.1f}%.\n")
        cal = [c for c in R["calibration"] if c["model"] == key and c["method"] == "conformal"
               and abs(c["nominal"] - 0.9) < 1e-6]
        if cal:
            bt += f"- Empirical coverage of 90% conformal intervals in backtest: {100*cal[0]['all_coverage']:.1f}%.\n"
    fb = " (high stratum used pooled residuals: too few calibration windows)" if env["calibration_high_pooled_fallback"] else ""
    txt = f"""# Model card: cement process-CO2 forecaster (artefact v{art['artefact_version']})

**Model.** {art.get('model', 'TPN')}{(' (' + art.get('variant') + ')') if art.get('model') == 'TPN' else ''}, seed ensemble of {len(art['states'])}.

**Intended use.** Screening-level forecasts (1-{art['config']['horizon']} years ahead) of national cement
process (calcination) CO2 emissions, with prediction intervals, for trend analysis and prioritisation.

**Not intended for.** Compliance reporting, carbon-credit accounting, or any use requiring
facility-level accuracy. Forecasts inherit the provenance of the input data; for countries whose
clinker ratios are IPCC defaults, the model cannot recover information that was never measured.

**Inputs.** At least {art['config']['seq_len']} consecutive annual observations of emissions (kt CO2);
optional cement and clinker production. Provenance is derived automatically
(reported clinker = 1.0, inferred = 0.5, default = 0.1).

**Training data.** {len(art['countries'])} countries; data SHA-256 `{art['data_sha256'][:16]}...`;
training origins {env['train_origins'][0]}-{env['train_origins'][1]};
calibration origins {env['calibration_origins'][0]}-{env['calibration_origins'][1]}
(n_train={env['n_train']}, n_cal={env['n_calibration']}).

**Evaluation.**
{bt or '- Run `python -m tpn_cement experiment` and pass --backtest results.json to include backtest metrics.'}

**Uncertainty.** Split-conformal intervals on log-change residuals, separate for windows with/without
non-default provenance{fb}. Coverage is guaranteed only under exchangeability; under temporal
shocks (e.g. 2020) coverage can fall below nominal.

**Automatic warnings.** Emission level outside the training range; last-year growth outside the
training 1-99th percentile; all-default provenance; data newer than the model by more than 5 years.

**Comparison references.** Every forecast is returned with the persistence and OLS-trend forecasts so
that users can see when the model departs from simple extrapolation.

**Created** {art['created_utc']} with torch {art['torch_version']}.
"""
    with open(os.path.join(out, "model_card.md"), "w") as fh:
        fh.write(txt)


class Forecaster:
    def __init__(self, art: Dict):
        self.art = art
        d = {k: tuple(v) if isinstance(v, list) else v for k, v in art["config"].items()}
        self.cfg = Config(**d)
        self.scaler = FeatureScaler.from_state(art["scaler"])
        self.models = []
        for st in art["states"]:
            m = _build(art.get("model", "TPN"), art.get("variant", "TPN (full)"), self.cfg)
            m.load_state_dict(st); m.eval()
            self.models.append(m)
        self.q = {}
        for k, v in art["conformal"].items():
            parts = k.split("|")
            if len(parts) == 3:
                self.q[(parts[0], int(parts[1]), float(parts[2]))] = v

    @classmethod
    def load(cls, path: str) -> "Forecaster":
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def predict(self, history: pd.DataFrame, iso: str = "XXX") -> pd.DataFrame:
        cfg, T, H = self.cfg, self.cfg.seq_len, self.cfg.horizon
        h = history.copy()
        for c in ["production", "clinker"]:
            if c not in h:
                h[c] = np.nan
        h = h[["year", "emissions", "production", "clinker"]].dropna(subset=["emissions"])
        h = h[h["emissions"] > 0].sort_values("year").reset_index(drop=True)
        runs = consecutive_runs(h["year"].to_numpy())
        if not runs or len(runs[-1]) < T:
            raise ValueError(f"Need at least {T} consecutive years of positive emissions ending at the latest year.")
        seg = h.iloc[runs[-1]].reset_index(drop=True)
        seg["iso"] = iso
        feat = country_features(assign_provenance(seg, cfg), cfg)
        X, q, le_w, tb, tier, _ = window_arrays(feat, len(feat) - 1, cfg)
        warnings: List[str] = []
        env = self.art["training_envelope"]
        if not (env["log_level_min"] <= le_w[-1] <= env["log_level_max"]):
            warnings.append("emission level outside training range (extrapolation)")
        if not (env["growth_p01"] <= X[-1, FEATURES.index("growth")] <= env["growth_p99"]):
            warnings.append("last-year growth outside training 1-99th percentile")
        if (q <= 0.1 + 1e-9).all():
            warnings.append("all input years carry default provenance")
        if int(seg["year"].iloc[-1]) > env["last_data_year"] + 5:
            warnings.append("data more than 5 years newer than model; retrain recommended")
        if int(seg["year"].iloc[-1]) < env["last_data_year"]:
            warnings.append(f"history ends in {int(seg['year'].iloc[-1])}, before the latest year in the "
                            f"training data ({env['last_data_year']}): targets may already be observed")
        high = bool((q > 0.1 + 1e-9).any())
        b = {"x": torch.as_tensor(self.scaler.transform(X[None]), dtype=torch.float32),
             "q": torch.as_tensor(np.array(q[None]), dtype=torch.float32),
             "tier": torch.as_tensor([tier]),
             "trend": torch.as_tensor(np.array(tb[None]), dtype=torch.float32),
             "rel": torch.as_tensor((le_w - le_w[-1])[None], dtype=torch.float32)}
        with torch.no_grad():
            pred = np.mean([m(b)["mean"].numpy()[0].astype(float) for m in self.models], 0)
        e0, t0 = float(np.exp(le_w[-1])), int(seg["year"].iloc[-1])
        st = "high" if high else "low"
        rows = []
        for k in range(H):
            row = {"iso": iso, "origin_year": t0, "target_year": t0 + k + 1, "horizon": k + 1,
                   "forecast_kt": e0 * np.exp(pred[k]), "ols_trend_kt": e0 * np.exp(float(tb[k])),
                   "persistence_kt": e0, "provenance_stratum": st}
            for a, lab in [(0.2, "80"), (0.1, "90")]:
                w = self.q.get((st, k, a), np.inf)
                if not np.isfinite(w):
                    w = self.q.get(("all", k, a), np.inf)
                ok = np.isfinite(w)
                row[f"lo{lab}_kt"] = e0 * np.exp(pred[k] - w) if ok else np.nan
                row[f"hi{lab}_kt"] = e0 * np.exp(pred[k] + w) if ok else np.nan
                if not ok and "interval unavailable" not in "; ".join(warnings):
                    warnings.append("interval unavailable: too few calibration windows")
            row["warnings"] = "; ".join(warnings)
            rows.append(row)
        return pd.DataFrame(rows)
