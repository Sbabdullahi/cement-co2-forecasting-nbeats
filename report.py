"""Writes LaTeX tabulars, a macro file with every number quoted in the text,
and the figures. The manuscript \\input{}s these files, so reported numbers are
always the ones the code produced."""
from __future__ import annotations

import os

import numpy as np

from .evaluation import conformal_fit, conformal_interval

ORDER = ["Persistence", "Drift", "OLS trend", "Damped trend", "Ridge", "GBRT",
         "LSTM", "TCN", "Graph (static)", "Graph (learned)", "N-BEATS", "DeepAR-style", "TPN", "TPN (val-selected)",
         "Validation-selected model"]
MIDRULE = object()


def _f(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return f"{x:.{d}f}" if np.isfinite(x) else "--"


def _p(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def _tex(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def _tab(path, colspec, header, rows):
    with open(path, "w") as fh:
        fh.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n")
        for r in rows:
            fh.write("\\midrule\n" if r is MIDRULE else r + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def write_latex(R, out):
    T = os.path.join(out, "tables")
    os.makedirs(T, exist_ok=True)
    main, sd = R["main"], R["seed_sd"]

    rows = [f"{r['cutoff']} & {r['train_t0_max']} & {r['val_t0']} & {r['test_t0']} & "
            f"{r['n_train']} & {r['n_val']} & {r['n_test']} & {r['test_high']} & {r['test_countries']}"
            for r in R["folds"]]
    _tab(f"{T}/tab_folds.tex", "@{}ccccrrrrr@{}",
         "$C$ & Train $t_0\\le$ & Val $t_0$ & Test $t_0$ & $n_{\\mathrm{tr}}$ & $n_{\\mathrm{val}}$ & "
         "$n_{\\mathrm{test}}$ & High & Countries", rows)

    rows = []
    for n in ORDER:
        if n not in main:
            continue
        if n in ("Ridge", "LSTM", "TPN (val-selected)"):
            rows.append(MIDRULE)
        a, h, l = main[n]["all"], main[n]["high"], main[n]["low"]
        name = f"\\textbf{{{n}}}" if n == R.get("focal", "N-BEATS") else n
        sds = f" ({_f(sd.get(n, 0))})" if sd.get(n, 0) > 0 else ""
        rows.append(f"{name} & {_f(a['MASE'])}{sds} & {_f(a['sMAPE'], 1)} & {_f(a['MAE_kt'], 0)} & "
                    f"{_f(a['R2_log'], 2)} & {_f(h['MASE'])} & {_f(l['MASE'])}")
    _tab(f"{T}/tab_main.tex", "@{}lrrrrrr@{}",
         "Model & MASE (s.d.) & sMAPE (\\%) & MAE (kt) & $R^2_{\\Delta\\log}$ & MASE$_{\\mathrm{high}}$ & MASE$_{\\mathrm{low}}$",
         rows)

    cut = [r["cutoff"] for r in R["folds"]]
    rows = [f"{n} & " + " & ".join(_f(v) for v in R["per_fold"][n]) for n in ORDER if n in R["per_fold"]]
    _tab(f"{T}/tab_perfold.tex", "@{}l" + "r" * len(cut) + "@{}",
         "Model & " + " & ".join(f"$C={c}$" for c in cut), rows)

    rows = [f"{s['model']} & {_f(s['diff'])} & [{_f(s['lo'])}, {_f(s['hi'])}] & {_p(s['p'])} & "
            f"{_p(s['p_holm'])} & {_f(s['dm_stat'], 2)} & {_p(s['dm_p'])}" for s in R["significance"]]
    _tab(f"{T}/tab_significance.tex", "@{}lrcrrrr@{}",
         "Comparator & $\\Delta$MASE & 95\\% CI & $p_{\\mathrm{boot}}$ & $p_{\\mathrm{Holm}}$ & DM-HLN & $p_{\\mathrm{DM}}$",
         rows)

    F = R.get("focal", "N-BEATS")
    if R.get("heterogeneity"):
        cols = [c for c in [F, "Persistence", "GBRT", "LSTM", "TPN"] if c in R["heterogeneity"][0]]
        rows = []
        prev = None
        for h in R["heterogeneity"]:
            if prev and prev != h["kind"]:
                rows.append(MIDRULE)
            prev = h["kind"]
            d = f"{_f(h['focal_vs_persist'])} [{_f(h['lo'])}, {_f(h['hi'])}]" if "focal_vs_persist" in h else "--"
            rows.append(f"{h['group']} & {h['n']} ({h['countries']}) & " + " & ".join(_f(h[c]) for c in cols) + f" & {d}")
        _tab(f"{T}/tab_heterogeneity.tex", "@{}lr" + "r" * len(cols) + "c@{}",
             "Group & Windows (countries) & " + " & ".join(cols) + f" & {F} $-$ Persistence [95\\% CI]", rows)
    if R.get("val_model_choice"):
        rows = [f"{v['cutoff']} & {v['selected']} & {_f(v['val_MASE'])} & {v['focal_rank']} of {v['n_candidates']} & {_f(v['focal_val_MASE'])}"
                for v in R["val_model_choice"]]
        _tab(f"{T}/tab_val_model_choice.tex", "@{}clrcr@{}",
             f"$C$ & Lowest validation MASE & Its MASE & Rank of {F} & {F} MASE", rows)

    if R.get("vs_persistence"):
        rows = [f"{s['model']} & {_f(s['diff'])} & [{_f(s['lo'])}, {_f(s['hi'])}] & {_p(s['p'])} & {_p(s['p_holm'])}"
                for s in R["vs_persistence"]]
        _tab(f"{T}/tab_vs_persistence.tex", "@{}lrcrr@{}",
             "Model & $\\Delta$MASE vs persistence & 95\\% CI & $p_{\\mathrm{boot}}$ & $p_{\\mathrm{Holm}}$", rows)
    if R.get("val_selected"):
        rows = [f"{v['cutoff']} & {_tex(v['selected'])} & {_f(v['val_MASE'])}" for v in R["val_selected"]]
        _tab(f"{T}/tab_val_selected.tex", "@{}clr@{}", "$C$ & Variant selected on validation & Validation MASE", rows)

    if R["ablation"]:
        rows = []
        for a in R["ablation"]:
            head = f"{_tex(a['variant'])} & {_f(a['MASE_ens'])} ({_f(a['MASE_sd'])})"
            rows.append(head + (" & -- & -- & --" if a["variant"] == "TPN (full)" else
                                f" & {_f(a['delta'])} [{_f(a['lo'])}, {_f(a['hi'])}] & {_p(a['p'])} & {_p(a.get('p_holm'))}"))
        _tab(f"{T}/tab_ablation.tex", "@{}lrcrr@{}",
             "Variant & MASE (s.d.) & $\\Delta$MASE [95\\% CI] & $p_{\\mathrm{boot}}$ & $p_{\\mathrm{Holm}}$", rows)

    rows = []
    for c in R["calibration"]:
        if round(c["nominal"], 2) not in (0.8, 0.9):
            continue
        rows.append(f"{c['model']} ({c['method']}) & {int(round(100 * c['nominal']))} & "
                    f"{_f(100 * c['all_coverage'], 1)} & {_f(100 * c['high_coverage'], 1)} & "
                    f"{_f(100 * c['low_coverage'], 1)} & {_f(100 * c['lastfold_coverage'], 1)} & "
                    f"{_f(c['all_width_pct'], 1)} & {_f(c['all_interval_score'], 3)}")
    _tab(f"{T}/tab_calibration.tex", "@{}lrrrrrrr@{}",
         "Model (method) & Nominal & Cov.\\ all & Cov.\\ high & Cov.\\ low & Cov.\\ last fold & "
         "Width (\\% of $E_{t_0}$) & Int.\\ score", rows)

    rows = [f"{_tex(r['group'])} & {_f(r['mean'])} & [{_f(r['lo'])}, {_f(r['hi'])}]"
            for r in sorted(R["feature_importance"], key=lambda r: -r["mean"])]
    _tab(f"{T}/tab_importance.tex", "@{}lrc@{}", "Feature group & $\\Delta$MASE & 2.5--97.5\\% range", rows)

    if R["coldstart"]:
        rows = []
        for n in ORDER:
            if n in R["coldstart"]:
                c = R["coldstart"][n]
                vs = c.get("vs_focal")
                ci = f"{_f(vs['diff'])} [{_f(vs['lo'])}, {_f(vs['hi'])}]" if vs else "--"
                rows.append(f"{n} & {_f(c['MASE'])} & {_f(c['sMAPE'], 1)} & {ci}")
        _tab(f"{T}/tab_coldstart.tex", "@{}lrrc@{}",
             f"Model & MASE & sMAPE (\\%) & {R.get('focal', 'N-BEATS')} $-$ model [95\\% CI]", rows)

    write_macros(R, out)


def write_macros(R, out):
    m = {}
    d = R["data"]
    m.update(ResNcountries=d["n_countries"], ResNsequences=d["n_sequences"], ResNhigh=d["n_high"],
             ResNlow=d["n_low"], ResPctReported=_f(d["pct_reported"], 1), ResPctInferred=_f(d["pct_inferred"], 1),
             ResPctDefault=_f(d["pct_default"], 1), ResFirstYear=d["first_year"], ResLastYear=d["last_year"])
    m["ResNfolds"] = len(R["folds"])
    m["ResCutoffs"] = ", ".join(str(r["cutoff"]) for r in R["folds"])
    m["ResMinTest"] = R["config"]["min_test_windows"]; m["ResMinTrain"] = R["config"]["min_train_windows"]
    m["ResNseeds"] = len(R["config"]["seeds"])
    t = R["main"]["TPN"]
    m.update(ResNtest=t["all"]["n"], ResNtestHigh=t["high"]["n"], ResNtestLow=t["low"]["n"])
    fl = R["folds"][-1]
    m.update(ResFinalTrain=fl["n_train"], ResFinalVal=fl["n_val"], ResFinalTest=fl["n_test"])
    m.update(ResTPNmase=_f(t["all"]["MASE"]), ResTPNmaseSD=_f(R["seed_sd"]["TPN"]),
             ResTPNsmape=_f(t["all"]["sMAPE"], 1), ResTPNmaekt=_f(t["all"]["MAE_kt"], 0),
             ResTPNmaseHigh=_f(t["high"]["MASE"]), ResTPNmaseLow=_f(t["low"]["MASE"]))
    p = R["main"]["Persistence"]["all"]
    m.update(ResPersistMase=_f(p["MASE"]), ResPersistSmape=_f(p["sMAPE"], 1))
    bb = R["best_baseline"]
    m.update(ResBestBaseName=bb, ResBestBaseMase=_f(R["main"][bb]["all"]["MASE"]),
             ResBestBaseSmape=_f(R["main"][bb]["all"]["sMAPE"], 1))
    sig = {s["model"]: s for s in R["significance"]}

    def put(prefix, s):
        m[prefix] = _f(s["diff"]); m[prefix + "Lo"] = _f(s["lo"]); m[prefix + "Hi"] = _f(s["hi"])
        m[prefix + "P"] = _p(s["p"]); m[prefix + "PHolm"] = _p(s.get("p_holm"))
    put("ResDiffBest", sig[bb]); put("ResDiffPersist", sig["Persistence"]); put("ResDiffLSTM", sig["LSTM"])
    m["ResDMpBest"] = _p(sig[bb]["dm_p"]); m["ResNorigins"] = sig[bb]["n_origins"]
    abl = {a["variant"]: a for a in R["ablation"]}
    for key, pre in [("w/o trend skip", "ResAblSkip"), ("w/o provenance attention bias", "ResAblQbias"),
                     ("w/o prototype", "ResAblProto"), ("w/o trend features", "ResAblTrendFeat"),
                     ("w/o SwiGLU (SiLU MLP)", "ResAblSwiglu"), ("w/o tier embedding", "ResAblTier"),
                     ("w/o provenance loss weight", "ResAblQloss")]:
        if key in abl:
            m[pre] = _f(abl[key]["delta"]); m[pre + "Lo"] = _f(abl[key]["lo"]); m[pre + "Hi"] = _f(abl[key]["hi"])
            m[pre + "PHolm"] = _p(abl[key].get("p_holm"))
    cal = {(c["model"], c["method"], round(c["nominal"], 2)): c for c in R["calibration"]}
    c90, c80 = cal.get(("TPN", "conformal", 0.9)), cal.get(("TPN", "conformal", 0.8))
    if c90:
        m.update(ResCovNinety=_f(100 * c90["all_coverage"], 1), ResCovNinetyHigh=_f(100 * c90["high_coverage"], 1),
                 ResCovNinetyLow=_f(100 * c90["low_coverage"], 1), ResWidthNinety=_f(c90["all_width_pct"], 1),
                 ResCovNinetyLast=_f(100 * c90["lastfold_coverage"], 1))
    if c80:
        m["ResCovEighty"] = _f(100 * c80["all_coverage"], 1)
    g90 = cal.get(("DeepAR-style", "Gaussian", 0.9))
    if g90:
        m["ResDeepARGaussCovNinety"] = _f(100 * g90["all_coverage"], 1)
    a = R["attention"]
    m.update(ResAttnNonDefault=_f(a["mean_attn_nondefault"]), ResAttnDefault=_f(a["mean_attn_default"]),
             ResAttnNmixed=a["n_mixed_windows"])
    for n, pre in [("TPN", "ResDenialTPN"), ("LSTM", "ResDenialLSTM")]:
        full = [x for x in R["denial"].get(n, []) if x[0] == 1.0]
        if full:
            m[pre] = _f(full[0][1])
    fi = sorted(R["feature_importance"], key=lambda r: -r["mean"])
    m["ResFItop"] = fi[0]["group"] if fi else "--"
    m["ResFIrho"] = _f(R["fi_rank_stability_spearman"], 2)
    if R["coldstart"]:
        cs = R["coldstart"]
        m["ResColdTPN"] = _f(cs["TPN"]["MASE"])
        best = min([n for n in cs if n != "TPN"], key=lambda n: cs[n]["MASE"])
        m["ResColdBestName"], m["ResColdBestMase"] = best, _f(cs[best]["MASE"])
    m["ResEpiHigh"] = _f(R["epistemic_seed_spread"]["high"], 4)
    m["ResEpiLow"] = _f(R["epistemic_seed_spread"]["low"], 4)
    m["ResBetaMean"] = _f(R["attention_beta"]["mean"], 2)
    m["ResBetaSD"] = _f(R["attention_beta"]["sd"], 2)
    F = R.get("focal", "N-BEATS")
    fm = R["main"][F]
    m.update(ResFocalName=F, ResFocalMase=_f(fm["all"]["MASE"]), ResFocalSmape=_f(fm["all"]["sMAPE"], 1),
             ResFocalMaekt=_f(fm["all"]["MAE_kt"], 0), ResFocalRmsekt=_f(fm["all"]["RMSE_kt"], 0),
             ResFocalSD=_f(R["seed_sd"].get(F, float("nan"))),
             ResFocalMaseHigh=_f(fm["high"]["MASE"]), ResFocalMaseLow=_f(fm["low"]["MASE"]))
    def red(a, b):   # percentage reduction of a relative to b
        return _f(100 * (b - a) / b, 0) if b else "--"
    for other, tag in [("TPN", "TPN"), ("LSTM", "LSTM"), ("TCN", "TCN"), ("DeepAR-style", "DeepAR"),
                       ("Graph (learned)", "GraphL"), ("Graph (static)", "GraphS"), ("GBRT", "GBRT"),
                       ("Ridge", "Ridge"), ("OLS trend", "OLS"), ("Damped trend", "Damped"),
                       ("Drift", "Drift"), ("Persistence", "Persist")]:
        if other in R["main"]:
            o = R["main"][other]["all"]
            m[f"ResRedMase{tag}"] = red(fm["all"]["MASE"], o["MASE"])
            m[f"ResRedMae{tag}"] = red(fm["all"]["MAE_kt"], o["MAE_kt"])
            m[f"ResMase{tag}"] = _f(o["MASE"]); m[f"ResMae{tag}"] = _f(o["MAE_kt"], 0); m[f"ResSmape{tag}"] = _f(o["sMAPE"], 1)
    sigF = {s["model"]: s for s in R["significance"]}
    for other, tag in [("TPN", "TPN"), ("LSTM", "LSTM"), ("TCN", "TCN"), ("DeepAR-style", "DeepAR"),
                       ("Graph (learned)", "GraphL"), ("GBRT", "GBRT"), ("Persistence", "Persist"), ("Damped trend", "Damped")]:
        if other in sigF:
            s_ = sigF[other]
            m[f"ResF{tag}"] = _f(s_["diff"]); m[f"ResF{tag}Lo"] = _f(s_["lo"]); m[f"ResF{tag}Hi"] = _f(s_["hi"])
            m[f"ResF{tag}PHolm"] = _p(s_.get("p_holm"))
    nsig = [s_["model"] for s_ in R["significance"] if s_["hi"] < 0 and s_.get("p_holm", 1) < 0.05]
    m["ResNFocalBetter"] = len(nsig)
    m["ResNComparators"] = len(R["significance"])
    nl = [n for n in ["LSTM", "TCN", "Graph (static)", "Graph (learned)", "DeepAR-style", "TPN"] if n in nsig]
    m["ResNFocalBetterDeep"] = len(nl)
    vmc = R.get("val_model_choice", [])
    if vmc:
        m["ResValChoice"] = "; ".join(f"{v['cutoff']}: {v['selected']}" for v in vmc)
        m["ResValFocalRanks"] = ", ".join(str(v["focal_rank"]) for v in vmc)
        m["ResNValFocalFirst"] = sum(v["selected"] == F for v in vmc)
    if "Validation-selected model" in R["main"]:
        m["ResValSelMase"] = _f(R["main"]["Validation-selected model"]["all"]["MASE"])
    calF = {(c["model"], c["method"], round(c["nominal"], 2)): c for c in R["calibration"]}
    f90, f80 = calF.get((F, "conformal", 0.9)), calF.get((F, "conformal", 0.8))
    if f90:
        m.update(ResFocalCovNinety=_f(100 * f90["all_coverage"], 1), ResFocalCovNinetyHigh=_f(100 * f90["high_coverage"], 1),
                 ResFocalCovNinetyLow=_f(100 * f90["low_coverage"], 1), ResFocalCovNinetyLast=_f(100 * f90["lastfold_coverage"], 1),
                 ResFocalWidthNinety=_f(f90["all_width_pct"], 1), ResFocalISNinety=_f(f90["all_interval_score"], 3))
    if f80:
        m["ResFocalCovEighty"] = _f(100 * f80["all_coverage"], 1)
    p90 = calF.get(("Persistence", "conformal", 0.9))
    if p90:
        m["ResPersistWidthNinety"] = _f(p90["all_width_pct"], 1); m["ResPersistISNinety"] = _f(p90["all_interval_score"], 3)
        m["ResPersistCovNinety"] = _f(100 * p90["all_coverage"], 1)
    if R["coldstart"] and F in R["coldstart"]:
        cs = R["coldstart"]
        m["ResColdFocal"] = _f(cs[F]["MASE"]); m["ResColdFocalSmape"] = _f(cs[F]["sMAPE"], 1)
        bo = min([n for n in cs if n != F], key=lambda n: cs[n]["MASE"])
        m["ResColdOtherName"], m["ResColdOtherMase"] = bo, _f(cs[bo]["MASE"])
        for other, tag in [("TPN", "TPN"), ("LSTM", "LSTM"), ("Persistence", "Persist")]:
            if other in cs and "vs_focal" in cs[other]:
                v = cs[other]["vs_focal"]; m[f"ResColdF{tag}"] = _f(v["diff"]); m[f"ResColdF{tag}Lo"] = _f(v["lo"]); m[f"ResColdF{tag}Hi"] = _f(v["hi"])
    het = R.get("heterogeneity", [])
    sc = [h for h in het if h["kind"] == "scale" and "focal_vs_persist" in h]
    if sc:
        small = sc[0]
        m["ResHetSmallN"] = small["n"]; m["ResHetSmallFocal"] = _f(small[F]); m["ResHetSmallPersist"] = _f(small["Persistence"])
    pc = [h for h in het if h["kind"] == "provenance"]
    for h in pc:
        tag = {"reported clinker": "Rep", "inferred only": "Inf", "default only": "Def"}[h["group"]]
        m[f"ResHet{tag}N"] = h["n"]; m[f"ResHet{tag}Focal"] = _f(h[F]); m[f"ResHet{tag}Persist"] = _f(h["Persistence"])
        m[f"ResHet{tag}Countries"] = h["countries"]
        if "focal_vs_persist" in h:
            m[f"ResHet{tag}Diff"] = _f(h["focal_vs_persist"]); m[f"ResHet{tag}Lo"] = _f(h["lo"]); m[f"ResHet{tag}Hi"] = _f(h["hi"])
            m[f"ResHet{tag}Red"] = _f(100 * (h["Persistence"] - h[F]) / h["Persistence"], 1)
        if "TPN" in h:
            m[f"ResHet{tag}TPN"] = _f(h["TPN"])
    vp = {s["model"]: s for s in R.get("vs_persistence", [])}
    beaters = [n for n, s in vp.items() if s["hi"] < 0 and s.get("p_holm", 1) < 0.05]
    m["ResNbeatPersist"] = len(beaters)
    m["ResBeatPersistList"] = ", ".join(beaters) if beaters else "none"
    if "TPN (val-selected)" in R["main"]:
        m["ResTPNselMase"] = _f(R["main"]["TPN (val-selected)"]["all"]["MASE"])
        s = vp.get("TPN (val-selected)")
        if s:
            m["ResTPNselVsPersist"] = _f(s["diff"]); m["ResTPNselVsPersistLo"] = _f(s["lo"]); m["ResTPNselVsPersistHi"] = _f(s["hi"])
        m["ResTPNselVariants"] = "; ".join(f"{v['cutoff']}: {v['selected']}" for v in R.get("val_selected", []))
    if "N-BEATS" in vp:
        s = vp["N-BEATS"]; m["ResNbeatsVsPersist"] = _f(s["diff"]); m["ResNbeatsVsPersistLo"] = _f(s["lo"]); m["ResNbeatsVsPersistHi"] = _f(s["hi"])
    if "N-BEATS" in R["main"]:
        m["ResNbeatsMase"] = _f(R["main"]["N-BEATS"]["all"]["MASE"])
    if "GBRT" in vp:
        s2 = vp["GBRT"]; m["ResGbrtVsPersist"] = _f(s2["diff"]); m["ResGbrtVsPersistLo"] = _f(s2["lo"]); m["ResGbrtVsPersistHi"] = _f(s2["hi"])
    if "TPN" in vp:
        s2 = vp["TPN"]; m["ResTPNvsPersist"] = _f(s2["diff"]); m["ResTPNvsPersistLo"] = _f(s2["lo"]); m["ResTPNvsPersistHi"] = _f(s2["hi"])
    ab = {a_["variant"]: a_ for a_ in R.get("ablation", [])}
    if "w/o trend skip" in ab:
        m["ResAblSkipSD"] = _f(ab["w/o trend skip"]["MASE_sd"]); m["ResAblSkipMase"] = _f(ab["w/o trend skip"]["MASE_ens"])
    m["ResPctDefaultWindows"] = _f(100 * R["data"]["n_low"] / max(R["data"]["n_sequences"], 1), 1)
    m["ResNbaselines"] = len([n for n in R["main"] if n != "TPN" and not n.startswith("ABL:")
                              and n not in ("TPN (val-selected)", "Validation-selected model")])
    m["ResNmodels"] = len([n for n in R["main"] if not n.startswith("ABL:")
                           and n not in ("TPN (val-selected)", "Validation-selected model")])
    m["ResFocalParams"] = f"{R['params'].get(R.get('focal', 'N-BEATS'), 0):,}".replace(",", "{,}")
    m["ResTPNparams"] = f"{R['params'].get('TPN', 0):,}".replace(",", "{,}")
    m["ResLSTMparams"] = f"{R['params'].get('LSTM', 0):,}".replace(",", "{,}")
    with open(os.path.join(out, "results_macros.tex"), "w") as fh:
        fh.write("% Auto-generated by tpn_cement.report -- do not edit by hand.\n")
        for k, v in m.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{_tex(str(v)) if isinstance(v, str) and '{' not in v and '$' not in v else v}}}\n")


def make_figures(R, fold_res, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = os.path.join(out, "figures")
    os.makedirs(F, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(f"{F}/{name}.pdf"); fig.savefig(f"{F}/{name}.png", dpi=200)
        plt.close(fig)

    names = [n for n in ORDER if n in R["per_fold"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    cut = [r["cutoff"] for r in R["folds"]]
    w = 0.8 / len(cut)
    for k, c in enumerate(cut):
        ax.bar(np.arange(len(names)) + k * w, [R["per_fold"][n][k] for n in names], w, label=f"C={c}")
    ax.set_xticks(np.arange(len(names)) + 0.4 - w / 2)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MASE (test)"); ax.legend(fontsize=8)
    save(fig, "fig_perfold")

    fig, ax = plt.subplots(figsize=(4.2, 4))
    for k, v in R["calibration_curve"].items():
        v = sorted(v)
        ax.plot([a for a, _ in v], [c for _, c in v], "-o", ms=3, label=k)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0.4, 1); ax.set_ylim(0.4, 1)
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage (pooled test)")
    ax.legend(fontsize=7)
    save(fig, "fig_calibration")

    if R["ablation"]:
        ab = [a for a in R["ablation"] if a["variant"] != "TPN (full)"]
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        y = np.arange(len(ab))
        ax.errorbar([a["delta"] for a in ab], y,
                    xerr=[[a["delta"] - a["lo"] for a in ab], [a["hi"] - a["delta"] for a in ab]], fmt="o")
        ax.axvline(0, color="k", lw=1); ax.set_yticks(y); ax.set_yticklabels([a["variant"] for a in ab], fontsize=8)
        ax.set_xlabel("$\\Delta$MASE vs full TPN (positive = component helps)")
        save(fig, "fig_ablation")

    fi = sorted(R["feature_importance"], key=lambda r: r["mean"])
    fig, ax = plt.subplots(figsize=(5, 2.8))
    y = np.arange(len(fi))
    ax.errorbar([r["mean"] for r in fi], y,
                xerr=[[r["mean"] - r["lo"] for r in fi], [r["hi"] - r["mean"] for r in fi]], fmt="s")
    ax.axvline(0, color="k", lw=1); ax.set_yticks(y); ax.set_yticklabels([r["group"] for r in fi], fontsize=8)
    ax.set_xlabel("Increase in MASE when group is permuted")
    save(fig, "fig_importance")

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for n, v in R["denial"].items():
        ax.errorbar([a for a, _, _ in v], [b for _, b, _ in v], yerr=[c for _, _, c in v], fmt="-o", ms=3, label=n)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Fraction of non-default years denied"); ax.set_ylabel("$\\Delta$MASE (high stratum)")
    ax.legend(fontsize=8)
    save(fig, "fig_denial")

    if fold_res:  # example forecasts need the fitted models; skipped when re-plotting from results.json
        r = fold_res[-1]; S = r["S_te"]
        Fm = R.get("focal", "N-BEATS")
        lo, hi = conformal_interval(S, r["test"][Fm], conformal_fit(r["S_va"], r["val"][Fm], cfg), 0.1)
        pool = sorted(set(S.iso[S.high])) or sorted(set(S.iso))
        cand = sorted(pool, key=lambda c: -S.e_t0[S.iso == c].max())[:4]
        fig, axes = plt.subplots(1, len(cand), figsize=(3 * len(cand), 2.8), squeeze=False)
        for ax, c in zip(axes[0], cand):
            i = np.where(S.iso == c)[0]; i = i[np.argmin(S.t0[i])]
            T, H = S.le_window.shape[1], S.y.shape[1]
            yrs_in = np.arange(S.t0[i] - T + 1, S.t0[i] + 1)
            yrs_out = S.t0[i] + np.arange(1, H + 1)
            ax.plot(yrs_in, np.exp(S.le_window[i]), "k-", lw=1.2, label="observed (input)")
            ax.plot(yrs_out, S.e_future[i], "ko", ms=3, label="observed (target)")
            ax.plot(yrs_out, S.e_t0[i] * np.exp(r["test"][Fm][i]), "C3-", label=Fm)
            if "TPN" in r["test"] and Fm != "TPN":
                ax.plot(yrs_out, S.e_t0[i] * np.exp(r["test"]["TPN"][i]), "C2:", lw=1, label="TPN")
            ax.fill_between(yrs_out, S.e_t0[i] * np.exp(lo[i]), S.e_t0[i] * np.exp(hi[i]), color="C3",
                            alpha=0.2, label="90% conformal")
            ax.plot(yrs_out, np.full(H, S.e_t0[i]), "C0--", lw=1, label="persistence")
            ax.set_title(c, fontsize=9); ax.tick_params(labelsize=7)
        axes[0][0].set_ylabel("kt CO$_2$"); axes[0][0].legend(fontsize=6)
        save(fig, "fig_examples")

    # forest plot: every model against persistence
    vp = R.get("vs_persistence", [])
    if vp:
        Fm = R.get("focal", "N-BEATS")
        rows = sorted([v for v in vp if v["model"] not in ("Validation-selected model",)], key=lambda v: v["diff"])
        fig, ax = plt.subplots(figsize=(5.6, 0.32 * len(rows) + 1.0))
        y = np.arange(len(rows))
        cols = ["C3" if v["model"] == Fm else "0.35" for v in rows]
        for yi, v, c in zip(y, rows, cols):
            ax.errorbar(v["diff"], yi, xerr=[[v["diff"] - v["lo"]], [v["hi"] - v["diff"]]], fmt="o", color=c, ms=4, capsize=2)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(y); ax.set_yticklabels([v["model"] for v in rows], fontsize=8)
        ax.set_xlabel("MASE difference from persistence (left = more accurate)")
        save(fig, "fig_forest")
    # error by emission scale and provenance
    het = R.get("heterogeneity", [])
    if het:
        Fm = R.get("focal", "N-BEATS")
        models = [c for c in [Fm, "Persistence", "GBRT", "LSTM", "TPN"] if c in het[0]]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [4, 3]})
        for ax, kind, title in [(axes[0], "scale", "By maximum annual emissions"), (axes[1], "provenance", "By activity-data provenance")]:
            hh = [h for h in het if h["kind"] == kind]
            x = np.arange(len(hh)); w = 0.8 / len(models)
            palette = ["C3", "C7", "C2", "C0", "C4"]
            for k, mname in enumerate(models):
                ax.bar(x + k * w, [h[mname] for h in hh], w, label=mname, color=palette[k % len(palette)])
            ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels([f"{h['group']}\n(n={h['n']})" for h in hh], fontsize=7)
            ax.set_title(title, fontsize=9); ax.set_ylabel("MASE")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(models), fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(f"{F}/fig_heterogeneity.pdf"); fig.savefig(f"{F}/fig_heterogeneity.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 2.8))
    pos = R["attention"]["mean_attn_by_position"]
    ax.bar(np.arange(-len(pos) + 1, 1), pos)
    ax.set_xlabel("Position relative to $t_0$"); ax.set_ylabel("Mean attention weight")
    save(fig, "fig_attention")


def overview_figure(panel, cfg, out):
    """Figure 1: national process-emission trajectories by emission scale, and the
    share of country-years by activity-data provenance over time (GCP-CEM)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from .data import assign_provenance
    P = assign_provenance(panel, cfg)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    peak = P.groupby("iso")["emissions"].max()
    edges = [0] + list(cfg.tier_thresholds_kt) + [np.inf]
    names = ["< 1 Mt", "1-5 Mt", "5-15 Mt", "> 15 Mt"]
    cls = pd.cut(peak, edges, labels=names, right=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    for k, nm in enumerate(names):
        for iso in cls[cls == nm].index:
            d = P[P["iso"] == iso].sort_values("year")
            ax.plot(d["year"], d["emissions"], color=f"C{k}", lw=0.5, alpha=0.35)
        ax.plot([], [], color=f"C{k}", lw=2, label=f"{nm} (n={int((cls == nm).sum())})")
    ax.set_yscale("log"); ax.set_ylabel("Process emissions (kt CO$_2$, log scale)"); ax.set_xlabel("Year")
    ax.legend(fontsize=7, title="Peak annual emissions", title_fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    ax = axes[1]
    sh = (P.groupby(["year", "source"]).size().unstack(fill_value=0))
    sh = sh.div(sh.sum(1), axis=0) * 100
    order = [c for c in ["reported", "inferred", "default"] if c in sh]
    lab = {"reported": "reported clinker", "inferred": "cement production only", "default": "neither"}
    ax.stackplot(sh.index, *[sh[c] for c in order], labels=[lab[c] for c in order], colors=["C2", "C1", "C7"], alpha=0.85)
    ax.set_ylim(0, 100); ax.set_ylabel("Share of country-years (%)"); ax.set_xlabel("Year")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig_overview.pdf")); fig.savefig(os.path.join(out, "figures", "fig_overview.png"), dpi=200)
    plt.close(fig)
    return {"n_countries": int(len(peak)), "by_class": {nm: int((cls == nm).sum()) for nm in names}}
