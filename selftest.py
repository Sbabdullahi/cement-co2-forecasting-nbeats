"""Synthetic data in the same wide schema as the real sheet, and automated
checks of every leakage control. Run:  python -m tpn_cement selftest
Synthetic results say nothing about real-data performance; they only show
that the pipeline runs and that the guarantees hold."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import Config, FEATURES
from .data import FeatureScaler, assign_provenance, build_sequences, load_panel, make_fold
from .train import make_model, predict, to_batch


def synthetic_wide(n_countries=30, seed=0, cfg: Config = None) -> pd.DataFrame:
    cfg = cfg or Config()
    rng = np.random.RandomState(seed)
    years = list(range(cfg.year_min, cfg.year_max + 1))
    rows = []
    for c in range(n_countries):
        iso = f"C{c:02d}"
        start = rng.choice([1990, 1990, 1990, 1995, 2000, 2005])
        g = rng.uniform(-0.01, 0.08)
        le = rng.uniform(4, 10) + np.cumsum(g + 0.05 * rng.randn(len(years))
                                            - 0.002 * np.arange(len(years)) * (g > 0.04))
        if rng.rand() < 0.5:
            le[years.index(2009):] -= rng.uniform(0, 0.15)
        le[years.index(2020)] -= rng.uniform(0, 0.12)
        e = np.exp(le)
        cr_true = np.clip(0.85 - 0.004 * (np.array(years) - 1990) + 0.02 * rng.randn(len(years)), 0.55, 0.95)
        prod = e / (cr_true * 0.52)
        regime = rng.choice(["reported", "partial", "none"], p=[0.15, 0.25, 0.6])
        P = np.full(len(years), np.nan) if regime == "none" else prod.copy()
        if regime == "partial":
            P = np.where(np.array(years) >= rng.choice([2000, 2008, 2014]), prod, np.nan)
        K = np.where((regime == "reported") & np.isfinite(P), prod * cr_true, np.nan)
        mask = np.array(years) < start
        e[mask] = np.nan; P[mask] = np.nan; K[mask] = np.nan
        for cat, act, vals in [("Emissions", "Cement production", e), ("Activity data", "Cement production", P),
                               ("Activity data", "Clinker production", K)]:
            rows.append({"ISO": iso, "Category": cat, "Activity": act, **dict(zip(years, vals))})
    return pd.DataFrame(rows)


def _ok(name, cond):
    print(("  PASS  " if cond else "  FAIL  ") + name, flush=True)
    if not cond:
        raise AssertionError(name)


def leakage_tests(cfg: Config) -> pd.DataFrame:
    print("Leakage and integrity tests")
    panel = load_panel(synthetic_wide(cfg=cfg), cfg)
    S = build_sequences(assign_provenance(panel, cfg), cfg)

    # L1/L2: changing every value after year Y must not change inputs of windows with t0 <= Y
    Y = 2010
    p2 = panel.copy()
    later = p2["year"] > Y
    p2.loc[later, "emissions"] *= np.exp(np.random.RandomState(1).randn(int(later.sum())))
    p2.loc[later, "production"] *= 3.0
    p2.loc[later, "clinker"] = np.nan
    S2 = build_sequences(assign_provenance(p2, cfg), cfg, require_future=False)
    key1 = {(c, t): i for i, (c, t) in enumerate(zip(S.iso, S.t0))}
    same, n_checked = True, 0
    for j, (c, t) in enumerate(zip(S2.iso, S2.t0)):
        if t <= Y and (c, t) in key1:
            i = key1[(c, t)]; n_checked += 1
            same &= bool(np.allclose(S.X[i], S2.X[j]) and np.allclose(S.q[i], S2.q[j])
                         and np.allclose(S.trend[i], S2.trend[j]) and S.tier[i] == S2.tier[j])
    _ok(f"features, provenance, trend baseline and tier are causal ({n_checked} windows checked against future edits)",
        same and n_checked > 0)

    from .data import auto_cutoffs
    cuts = cfg.cutoffs or auto_cutoffs(S, cfg)
    for k, C in enumerate(cuts):
        f = make_fold(S, C, cfg, last_fold=(k == len(cuts) - 1))   # asserts inside
        _ok(f"fold C={C}: max train target {(S.t0[f['train']] + cfg.horizon).max()} < "
            f"min val target {(S.t0[f['val']] + 1).min()}; val targets <= {C} < test targets", True)

    f = make_fold(S, cuts[-1], cfg, last_fold=True)
    sc1 = FeatureScaler().fit(S.X[f["train"]])
    Xmod = S.X.copy(); Xmod[f["test"]] *= 100.0
    sc2 = FeatureScaler().fit(Xmod[f["train"]])
    _ok("feature scaler unaffected by test data",
        np.allclose(sc1.mean_, sc2.mean_) and np.allclose(sc1.scale_, sc2.scale_))

    m = make_model("TPN", cfg, len(FEATURES))
    Xs = sc1.transform(S.X)
    m.train(); m(to_batch(S, Xs, f["train"][:32]))
    before = m.protos.clone()
    m.eval()
    with torch.no_grad():
        m(to_batch(S, Xs, f["test"]))
    _ok("prototype bank unchanged by evaluation forward pass", torch.equal(before, m.protos))

    p_all = predict(m, S.subset(f["test"]), Xs[f["test"]])["mean"]
    p_one = predict(m, S.subset(f["test"][:1]), Xs[f["test"][:1]])["mean"]
    _ok("per-sequence predictions independent of batch composition", np.allclose(p_all[0], p_one[0], atol=1e-5))

    npar = lambda mm: sum(p.numel() for p in mm.parameters())
    full = make_model("TPN", cfg, len(FEATURES))
    _ok("ablation variants are structurally different models",
        npar(full) != npar(make_model("TPN", cfg, len(FEATURES), use_quality_bias=False))
        and npar(full) != npar(make_model("TPN", cfg, len(FEATURES), use_prototype=False))
        and npar(full) != npar(make_model("TPN", cfg, len(FEATURES), use_swiglu=False)))
    return panel


def run_selftest(out="selftest_out", quick=True):
    cfg = Config()
    if quick:
        cfg.seeds = (0, 1)
        cfg.max_epochs = 40
        cfg.patience = 10
        cfg.n_boot = 200
        cfg.n_perm_repeats = 3
        cfg.denial_trials = 2
    panel = leakage_tests(cfg)
    from .experiment import run_experiment
    run_experiment(panel, cfg, out, do_ablation=True, do_coldstart=True)
    from .deploy import train_final, Forecaster
    path = train_final(panel, cfg, f"{out}/artefact", backtest_json=f"{out}/results.json")
    fc = Forecaster.load(path)
    iso = panel["iso"].iloc[0]
    hist = panel[panel["iso"] == iso][["year", "emissions", "production", "clinker"]]
    print(fc.predict(hist, iso=iso).round(1).to_string(index=False))
    print("Self-test complete (synthetic data; numbers are not results).")
