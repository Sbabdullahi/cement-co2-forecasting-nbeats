"""Full experimental protocol. All numbers reported in the manuscript are
produced here and written to LaTeX tables/macros, so the text and the code
cannot drift apart."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .baselines import STAT_BASELINES, RidgeResidual, GBRTResidual
from .config import Config, FEATURES, FEATURE_GROUPS
from .data import (SequenceSet, FeatureScaler, assign_provenance, build_sequences, describe,
                   make_fold, tier_of, auto_cutoffs)
from .evaluation import (metrics, metrics_by_stratum, per_sequence_mase, conformal_fit,
                         conformal_interval, gaussian_interval, interval_scores,
                         cluster_bootstrap_diff, dm_hln, holm, permutation_importance,
                         deny_provenance, attention_by_provenance)
from .models import n_params
from .train import NEURAL_MODELS, ABLATIONS, make_model, fit, predict

STAT_ORDER = list(STAT_BASELINES) + ["Ridge", "GBRT"]
MODEL_ORDER = STAT_ORDER + NEURAL_MODELS


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------
# Static country graph for the graph baseline (training-period data only)
# ---------------------------------------------------------------------------
def static_adjacency(panel: pd.DataFrame, countries: List[str], cutoff: int, cfg: Config, k: int = 4):
    P = panel[panel["year"] <= cutoff]
    n = len(countries)
    A = np.zeros((n, n))
    g = {}
    for c in countries:
        d = P[P["iso"] == c].sort_values("year")
        if len(d) > 1:
            g[c] = pd.Series(np.diff(np.log(np.maximum(d["emissions"].to_numpy(), cfg.emission_floor_kt))),
                             index=d["year"].to_numpy()[1:])
    for i, a in enumerate(countries):
        corr = []
        for j, b in enumerate(countries):
            if i == j or a not in g or b not in g:
                continue
            s = pd.concat([g[a], g[b]], axis=1, join="inner").dropna()
            if len(s) >= 8:
                r = s.corr().iloc[0, 1]
                if np.isfinite(r) and r > 0:
                    corr.append((r, j))
        for r, j in sorted(corr, reverse=True)[:k]:
            A[i, j] = A[j, i] = max(A[i, j], r)
    tiers = {c: tier_of(P[P["iso"] == c]["emissions"].max() if (P["iso"] == c).any() else 0.0, cfg)
             for c in countries}
    for i, a in enumerate(countries):
        for j, b in enumerate(countries):
            if i != j and tiers[a] == tiers[b]:
                A[i, j] += 0.25
    if cfg.country_meta_csv and os.path.exists(cfg.country_meta_csv):
        meta = pd.read_csv(cfg.country_meta_csv).set_index("ISO")
        for i, a in enumerate(countries):
            for j, b in enumerate(countries):
                if i == j or a not in meta.index or b not in meta.index:
                    continue
                if meta.loc[a, "region"] == meta.loc[b, "region"]:
                    A[i, j] += 0.5
                if "income_group" in meta.columns and meta.loc[a, "income_group"] == meta.loc[b, "income_group"]:
                    A[i, j] += 0.25
    return A


# ---------------------------------------------------------------------------
# One rolling-origin fold
# ---------------------------------------------------------------------------
def run_fold(S: SequenceSet, panel: pd.DataFrame, fold: Dict, cfg: Config, do_ablation: bool):
    S_tr, S_va, S_te = S.subset(fold["train"]), S.subset(fold["val"]), S.subset(fold["test"])
    scaler = FeatureScaler().fit(S_tr.X)                # fitted on training sequences only
    X_tr, X_va, X_te = (scaler.transform(Z.X) for Z in (S_tr, S_va, S_te))
    countries = sorted(set(S.iso.tolist()))
    node_index = {c: i for i, c in enumerate(countries)}
    A = static_adjacency(panel, countries, fold["cutoff"], cfg)
    nF = len(FEATURES)

    R = {"test": {}, "val": {}, "seed_test": defaultdict(list), "sd": {}, "params": {},
         "models": defaultdict(list), "beta": [], "S_te": S_te, "S_va": S_va, "X_te": X_te,
         "X_va": X_va, "scaler": scaler, "node_index": node_index, "cutoff": fold["cutoff"]}

    for name, fn in STAT_BASELINES.items():
        R["test"][name], R["val"][name] = fn(S_te), fn(S_va)
        R["seed_test"][name] = [R["test"][name]]
    rg = RidgeResidual().fit(S_tr, X_tr, S_va, X_va)
    R["test"]["Ridge"], R["val"]["Ridge"] = rg.predict(S_te, X_te), rg.predict(S_va, X_va)
    R["seed_test"]["Ridge"] = [R["test"]["Ridge"]]
    gb = GBRTResidual().fit(S_tr, X_tr, S_va, X_va)
    R["test"]["GBRT"], R["val"]["GBRT"] = gb.predict(S_te, X_te), gb.predict(S_va, X_va)
    R["seed_test"]["GBRT"] = [R["test"]["GBRT"]]

    jobs = [(m, {}) for m in NEURAL_MODELS]
    if do_ablation:
        jobs += [("ABL:" + k, v) for k, v in ABLATIONS.items() if k != "TPN (full)"]
    for name, kw in jobs:
        arch = "TPN" if name.startswith("ABL:") else name
        tv, vv, sdv = [], [], []
        for seed in cfg.seeds:
            m = make_model(arch, cfg, nF, n_nodes=len(countries), A_static=A, **kw)
            qw = (arch == "TPN") and not kw.get("_no_qweight", False)
            m, _ = fit(m, S_tr, X_tr, S_va, X_va, cfg, seed, node_index=node_index, quality_weight=qw)
            pt, pv = predict(m, S_te, X_te, node_index), predict(m, S_va, X_va, node_index)
            tv.append(pt["mean"]); vv.append(pv["mean"])
            if "sd" in pt:
                sdv.append(pt["sd"])
            if name in ("TPN", "LSTM"):
                R["models"][name].append(m)
            if name == "TPN":
                R["beta"].append(float(m.q_beta.detach()))
        R["params"][name] = n_params(m)
        R["test"][name], R["val"][name] = np.mean(tv, 0), np.mean(vv, 0)
        R["seed_test"][name] = tv
        if sdv:   # mixture sd of the seed ensemble
            R["sd"][name] = np.sqrt(np.mean(np.square(sdv), 0) + np.var(tv, 0))
        log(f"    fold {fold['cutoff']} {name:32s} test MASE={per_sequence_mase(S_te, R['test'][name]).mean():.3f}")
    return R


def run_coldstart(S: SequenceSet, fold: Dict, cfg: Config, seeds):
    """Leave-countries-out within the final temporal fold: test countries are
    absent from training and validation (tests transfer to unseen countries)."""
    rng = np.random.RandomState(123)
    test_c = sorted(set(S.iso[fold["test"]].tolist()))
    rng.shuffle(test_c)
    preds = defaultdict(list); idx_all = []
    for g in np.array_split(np.array(test_c), cfg.coldstart_folds):
        g = set(g.tolist())
        tr = np.array([i for i in fold["train"] if S.iso[i] not in g])
        va = np.array([i for i in fold["val"] if S.iso[i] not in g])
        te = np.array([i for i in fold["test"] if S.iso[i] in g])
        if len(te) == 0 or len(va) == 0:
            continue
        S_tr, S_va, S_te = S.subset(tr), S.subset(va), S.subset(te)
        sc = FeatureScaler().fit(S_tr.X)
        X_tr, X_va, X_te = sc.transform(S_tr.X), sc.transform(S_va.X), sc.transform(S_te.X)
        for name, fn in STAT_BASELINES.items():
            preds[name].append(fn(S_te))
        preds["Ridge"].append(RidgeResidual().fit(S_tr, X_tr, S_va, X_va).predict(S_te, X_te))
        preds["GBRT"].append(GBRTResidual().fit(S_tr, X_tr, S_va, X_va).predict(S_te, X_te))
        for name in dict.fromkeys([cfg.focal_model, "LSTM", "TPN"]):
            pv = []
            for seed in seeds:
                m = make_model(name, cfg, len(FEATURES))
                m, _ = fit(m, S_tr, X_tr, S_va, X_va, cfg, seed, quality_weight=(name == "TPN"))
                pv.append(predict(m, S_te, X_te)["mean"])
            preds[name].append(np.mean(pv, 0))
        idx_all.append(te)
    S_cs = S.subset(np.concatenate(idx_all))
    return S_cs, {k: np.concatenate(v) for k, v in preds.items()}


def _pooled(fold_res, key, seed=None):
    if seed is None:
        return np.concatenate([r["test"][key] for r in fold_res])
    return np.concatenate([r["seed_test"][key][seed] for r in fold_res])


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(panel_raw: pd.DataFrame, cfg: Config, out: str, do_ablation=True, do_coldstart=True):
    for sub in ["", "tables", "figures"]:
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    panel = assign_provenance(panel_raw, cfg)
    S = build_sequences(panel, cfg)
    desc = describe(S, panel, cfg)
    log(f"Data: {desc}")

    if not cfg.cutoffs:
        cfg.cutoffs = auto_cutoffs(S, cfg)
        log(f"Cut-offs chosen from data coverage (before any training): {cfg.cutoffs}")
    folds, fold_rows = [], []
    for k, C in enumerate(cfg.cutoffs):
        f = make_fold(S, C, cfg, last_fold=(k == len(cfg.cutoffs) - 1))
        folds.append(f)
        fold_rows.append({"cutoff": C, "n_train": len(f["train"]), "n_val": len(f["val"]),
                          "n_test": len(f["test"]), "test_high": int(S.high[f["test"]].sum()),
                          "test_countries": len(set(S.iso[f["test"]])),
                          "train_t0_max": int(S.t0[f["train"]].max()),
                          "val_t0": f"{S.t0[f['val']].min()}--{S.t0[f['val']].max()}",
                          "test_t0": f"{S.t0[f['test']].min()}--{S.t0[f['test']].max()}"})
        log(f"Fold C={C}: {fold_rows[-1]}")

    fold_res = []
    for f in folds:
        log(f"=== Fold cut-off {f['cutoff']} ===")
        fold_res.append(run_fold(S, panel, f, cfg, do_ablation))

    # ---- pooled test set across folds (distinct forecast origins) ----------
    S_te = SequenceSet.concat([r["S_te"] for r in fold_res])
    names = list(fold_res[0]["test"])
    P = {n: _pooled(fold_res, n) for n in names}
    main = {n: metrics_by_stratum(S_te, P[n]) for n in names}
    seed_sd = {}
    for n in names:
        ns = len(fold_res[0]["seed_test"][n])
        per_seed = [per_sequence_mase(S_te, _pooled(fold_res, n, s)).mean() for s in range(ns)]
        seed_sd[n] = float(np.std(per_seed, ddof=1)) if ns > 1 else 0.0
    per_fold = {n: [float(per_sequence_mase(r["S_te"], r["test"][n]).mean()) for r in fold_res] for n in names}

    # ---- significance: focal model vs every other model --------------------
    F = cfg.focal_model
    L = {n: per_sequence_mase(S_te, P[n]) for n in names}
    comps = [n for n in MODEL_ORDER if n != F and n in L]
    sig = []
    for n in comps:
        bs = cluster_bootstrap_diff(L[F], L[n], S_te.iso, cfg.n_boot, seed=1)
        dm = dm_hln(L[F], L[n], S_te.t0, cfg.horizon)
        sig.append({"model": n, **bs, "dm_stat": dm["stat"], "dm_p": dm["p"], "n_origins": dm["n_origins"]})
    for row, ph in zip(sig, holm([r["p"] for r in sig])):
        row["p_holm"] = ph
    best_base = min(comps, key=lambda n: main[n]["all"]["MASE"])

    # ---- TPN with components selected on validation data only -------------
    # For each fold, the TPN variant (full or one component removed) with the
    # lowest *validation* MASE is chosen; its test forecasts are then scored.
    # No test information enters the choice.
    val_selected = []
    if do_ablation:
        sel_parts = []
        for r in fold_res:
            keys = ["TPN"] + ["ABL:" + k for k in ABLATIONS if k != "TPN (full)"]
            vm = {k: float(per_sequence_mase(r["S_va"], r["val"][k]).mean()) for k in keys}
            best = min(vm, key=vm.get)
            val_selected.append({"cutoff": r["cutoff"], "selected": "TPN (full)" if best == "TPN" else best[4:],
                                 "val_MASE": vm[best]})
            sel_parts.append(r["test"][best])
        P["TPN (val-selected)"] = np.concatenate(sel_parts)
        main["TPN (val-selected)"] = metrics_by_stratum(S_te, P["TPN (val-selected)"])
        L["TPN (val-selected)"] = per_sequence_mase(S_te, P["TPN (val-selected)"])
        seed_sd["TPN (val-selected)"] = float("nan")
        per_fold["TPN (val-selected)"] = [float(per_sequence_mase(r["S_te"], p).mean())
                                          for r, p in zip(fold_res, sel_parts)]

    # ---- which model would validation data have chosen? -------------------
    # Per fold, every candidate is ranked by validation MASE. This shows whether
    # the focal model could have been chosen without looking at test data.
    val_model_choice = []
    for r in fold_res:
        vm = {n: float(per_sequence_mase(r["S_va"], r["val"][n]).mean()) for n in MODEL_ORDER if n in r["val"]}
        ranked = sorted(vm, key=vm.get)
        val_model_choice.append({"cutoff": r["cutoff"], "selected": ranked[0], "val_MASE": vm[ranked[0]],
                                 "focal_rank": ranked.index(F) + 1 if F in ranked else None,
                                 "focal_val_MASE": vm.get(F), "n_candidates": len(ranked)})
    P["Validation-selected model"] = np.concatenate(
        [r["test"][v["selected"]] for r, v in zip(fold_res, val_model_choice)])
    main["Validation-selected model"] = metrics_by_stratum(S_te, P["Validation-selected model"])
    L["Validation-selected model"] = per_sequence_mase(S_te, P["Validation-selected model"])

    # ---- heterogeneity: error by emission scale and by provenance class ---
    qmax = S_te.q.max(1)
    prov_class = np.where(qmax >= 1.0 - 1e-9, "reported clinker", np.where(qmax > 0.1 + 1e-9, "inferred only", "default only"))
    tier_names = ["< 1 Mt", "1-5 Mt", "5-15 Mt", "> 15 Mt"]
    hetero = []
    focus = [m for m in dict.fromkeys([F, "Persistence", "GBRT", "LSTM", "TPN"]) if m in L]
    for kind, labels, groups in [("scale", tier_names, [S_te.tier == k for k in range(4)]),
                                 ("provenance", ["reported clinker", "inferred only", "default only"],
                                  [prov_class == c for c in ["reported clinker", "inferred only", "default only"]])]:
        for lab, msk in zip(labels, groups):
            if msk.sum() == 0:
                continue
            row = {"kind": kind, "group": lab, "n": int(msk.sum()), "countries": int(len(set(S_te.iso[msk])))}
            for m_ in focus:
                row[m_] = float(L[m_][msk].mean())
            if msk.sum() >= 10 and len(set(S_te.iso[msk])) >= 5:
                d = cluster_bootstrap_diff(L[F][msk], L["Persistence"][msk], S_te.iso[msk], cfg.n_boot, seed=5)
                row.update({"focal_vs_persist": d["diff"], "lo": d["lo"], "hi": d["hi"]})
            hetero.append(row)

    # ---- every model vs persistence (does anything beat a random walk?) ---
    vs_persist = []
    for n in [m for m in MODEL_ORDER + ["TPN (val-selected)", "Validation-selected model"] if m in L and m != "Persistence"]:
        bs = cluster_bootstrap_diff(L[n], L["Persistence"], S_te.iso, cfg.n_boot, seed=4)
        vs_persist.append({"model": n, **bs})
    for row, ph in zip(vs_persist, holm([r["p"] for r in vs_persist])):
        row["p_holm"] = ph

    # ---- ablation -----------------------------------------------------------
    abl = []
    if do_ablation:
        for k in ABLATIONS:
            key = "TPN" if k == "TPN (full)" else "ABL:" + k
            per_seed = [per_sequence_mase(S_te, _pooled(fold_res, key, s)).mean() for s in range(len(cfg.seeds))]
            row = {"variant": k, "MASE_ens": float(L[key].mean()),
                   "MASE_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0}
            if key != "TPN":
                bs = cluster_bootstrap_diff(L[key], L["TPN"], S_te.iso, cfg.n_boot, seed=2)
                row.update({"delta": bs["diff"], "lo": bs["lo"], "hi": bs["hi"], "p": bs["p"]})
            abl.append(row)
        ab_rows = [r for r in abl if "p" in r]
        for r, ph in zip(ab_rows, holm([r["p"] for r in ab_rows])):
            r["p_holm"] = ph

    # ---- uncertainty: split conformal (Mondrian by provenance) ------------
    calib, curve = [], defaultdict(list)
    for n in dict.fromkeys([F, "Persistence", "TPN", "DeepAR-style", "LSTM"]):
        if n not in P:
            continue
        for method in ["conformal"] + (["Gaussian"] if n == "DeepAR-style" else []):
            for a in cfg.conformal_alphas:
                los, his = [], []
                for r in fold_res:
                    if method == "conformal":
                        qd = conformal_fit(r["S_va"], r["val"][n], cfg)
                        lo, hi = conformal_interval(r["S_te"], r["test"][n], qd, a)
                    else:
                        lo, hi = gaussian_interval(r["test"][n], r["sd"][n], a)
                    los.append(lo); his.append(hi)
                lo, hi = np.concatenate(los), np.concatenate(his)
                row = {"model": n, "method": method, "nominal": 1 - a}
                for st, m in [("all", None), ("high", S_te.high), ("low", ~S_te.high)]:
                    for kk, v in interval_scores(S_te, lo, hi, a, m).items():
                        row[f"{st}_{kk}"] = v
                # coverage of last fold separately (contains 2020 shock)
                lf = fold_res[-1]
                if method == "conformal":
                    qd = conformal_fit(lf["S_va"], lf["val"][n], cfg)
                    l2, h2 = conformal_interval(lf["S_te"], lf["test"][n], qd, a)
                else:
                    l2, h2 = gaussian_interval(lf["test"][n], lf["sd"][n], a)
                row["lastfold_coverage"] = interval_scores(lf["S_te"], l2, h2, a)["coverage"]
                calib.append(row)
                curve[f"{n} ({method})"].append((1 - a, row["all_coverage"]))
    fallback = {r["cutoff"]: conformal_fit(r["S_va"], r["val"][F], cfg)[("high", "pooled_fallback")]
                for r in fold_res}

    # ---- interpretability: grouped permutation importance (TPN) ------------
    fi, rank_mat = defaultdict(list), []
    for r in fold_res:
        for si, m in enumerate(r["models"]["TPN"]):
            fn = lambda Xs, q, m=m, Sx=r["S_te"]: predict(m, Sx, Xs, q_override=q)["mean"]
            imp = permutation_importance(fn, r["S_te"], r["X_te"], cfg, seed=si)
            for g, v in imp.items():
                fi[g].extend(v.tolist())
            if r is fold_res[-1]:
                rank_mat.append([imp[g].mean() for g in FEATURE_GROUPS])
    fi_rows = [{"group": g, "mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5))} for g, v in fi.items()]
    rho = [spearmanr(rank_mat[i], rank_mat[j])[0] for i in range(len(rank_mat)) for j in range(i + 1, len(rank_mat))]
    rank_stability = float(np.nanmean(rho)) if rho else float("nan")

    # ---- attention diagnostics --------------------------------------------
    att = np.concatenate([np.mean([predict(m, r["S_te"], r["X_te"])["attn"] for m in r["models"]["TPN"]], 0)
                          for r in fold_res])
    att_diag = attention_by_provenance(att, S_te)
    betas = [b for r in fold_res for b in r["beta"]]

    # ---- epistemic proxy: seed-ensemble spread by provenance stratum ------
    spread = np.concatenate([np.std(np.stack(r["seed_test"]["TPN"]), axis=0, ddof=1).mean(1) for r in fold_res]) \
        if len(cfg.seeds) > 1 else np.zeros(len(S_te))
    epistemic = {"high": float(spread[S_te.high].mean()) if S_te.high.any() else float("nan"),
                 "low": float(spread[~S_te.high].mean())}

    # ---- provenance-denial sensitivity -------------------------------------
    denial = defaultdict(list)
    for n in ["TPN", "LSTM"]:
        for frac in cfg.denial_fractions:
            deltas = []
            for trial in range(cfg.denial_trials if 0 < frac < 1 else 1):
                rng = np.random.RandomState(1000 + trial)
                d_all, base_all = [], []
                for r in fold_res:
                    Sh = r["S_te"].subset(np.where(r["S_te"].high)[0])
                    if len(Sh) == 0:
                        continue
                    Xd, qd = deny_provenance(Sh, frac, cfg, rng)
                    pd_ = np.mean([predict(m, Sh, r["scaler"].transform(Xd), q_override=qd)["mean"]
                                   for m in r["models"][n]], 0)
                    pb_ = np.mean([predict(m, Sh, r["scaler"].transform(Sh.X))["mean"] for m in r["models"][n]], 0)
                    d_all.append(per_sequence_mase(Sh, pd_)); base_all.append(per_sequence_mase(Sh, pb_))
                if d_all:
                    deltas.append(np.concatenate(d_all).mean() - np.concatenate(base_all).mean())
            denial[n].append((frac, float(np.mean(deltas)) if deltas else float("nan"),
                              float(np.std(deltas)) if len(deltas) > 1 else 0.0))

    # ---- cold start ----------------------------------------------------------
    cold = None
    if do_coldstart:
        log("=== Cold-start (leave-countries-out, final fold) ===")
        S_cs, Pcs = run_coldstart(S, folds[-1], cfg, cfg.seeds[:3])
        cold = {n: metrics(S_cs, p) for n, p in Pcs.items()}
        Lc = {n: per_sequence_mase(S_cs, p) for n, p in Pcs.items()}
        for n in cold:
            if n != F:
                cold[n]["vs_focal"] = cluster_bootstrap_diff(Lc[F], Lc[n], S_cs.iso, cfg.n_boot, seed=3)

    # ---- save everything ----------------------------------------------------
    pred_rows = []
    for r in fold_res:
        St = r["S_te"]
        lo, hi = conformal_interval(St, r["test"][F], conformal_fit(r["S_va"], r["val"][F], cfg), 0.1)
        for i in range(len(St)):
            for h in range(cfg.horizon):
                row = {"cutoff": r["cutoff"], "iso": St.iso[i], "origin": int(St.t0[i]),
                       "target_year": int(St.t0[i] + h + 1), "horizon": h + 1,
                       "stratum": "high" if St.high[i] else "low", "observed_kt": St.e_future[i, h],
                       f"{F}_lo90_kt": St.e_t0[i] * np.exp(lo[i, h]), f"{F}_hi90_kt": St.e_t0[i] * np.exp(hi[i, h])}
                for n in names:
                    if not n.startswith("ABL:"):
                        row[f"{n}_kt"] = St.e_t0[i] * np.exp(r["test"][n][i, h])
                pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(os.path.join(out, "test_predictions.csv"), index=False)

    results = {"config": cfg.to_dict(), "data": desc, "folds": fold_rows, "main": main,
               "seed_sd": seed_sd, "per_fold": per_fold, "significance": sig, "best_baseline": best_base,
               "ablation": abl, "val_selected": val_selected, "focal": F,
               "val_model_choice": val_model_choice, "heterogeneity": hetero, "vs_persistence": vs_persist, "calibration": calib, "calibration_curve": dict(curve),
               "mondrian_high_fallback": fallback, "feature_importance": fi_rows,
               "fi_rank_stability_spearman": rank_stability, "attention": att_diag,
               "attention_beta": {"mean": float(np.mean(betas)),
                                  "sd": float(np.std(betas, ddof=1)) if len(betas) > 1 else 0.0},
               "epistemic_seed_spread": epistemic, "denial": dict(denial), "coldstart": cold,
               "params": {n: fold_res[-1]["params"].get(n) for n in NEURAL_MODELS}}
    with open(os.path.join(out, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    from .report import write_latex, make_figures
    write_latex(results, out)
    make_figures(results, fold_res, cfg, out)
    log(f"Done. Outputs in {out}")
    return results
