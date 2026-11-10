"""Command-line interface.

  python -m tpn_cement selftest [--full] [--out selftest_out]
  python -m tpn_cement experiment  --data Carbon_Emission.xlsx [--sheet X-Y] [--out results]
                                   [--seeds 0 1 2 3 4] [--no-ablation] [--no-coldstart]
                                   [--country-meta meta.csv] [--default-clinker 0.75]
  python -m tpn_cement train-final --data Carbon_Emission.xlsx --out artefact [--backtest results/results.json]
  python -m tpn_cement forecast    --artefact artefact/tpn_artifact.pt --history country.csv [--iso XXX]
"""
import argparse

from .config import Config


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tpn_cement")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--full", action="store_true"); s.add_argument("--out", default="selftest_out")
    e = sub.add_parser("experiment")
    e.add_argument("--data", required=True, help="Excel file, or folder with the GCP-CEM CSV files"); e.add_argument("--sheet", default="X-Y")
    e.add_argument("--year-max", type=int, default=None)
    e.add_argument("--out", default="results"); e.add_argument("--seeds", type=int, nargs="*")
    e.add_argument("--no-ablation", action="store_true"); e.add_argument("--no-coldstart", action="store_true")
    e.add_argument("--country-meta", default=""); e.add_argument("--default-clinker", type=float, default=None)
    e.add_argument("--provenance", choices=["auto", "activity"], default="auto")
    e.add_argument("--cutoffs", type=int, nargs="*", help="fixed cut-off years; default: chosen from data coverage")
    o = sub.add_parser("overview")
    o.add_argument("--data", required=True); o.add_argument("--sheet", default="X-Y"); o.add_argument("--out", default="results")
    g = sub.add_parser("diagnose")
    g.add_argument("--provenance", choices=["auto", "activity"], default="auto")
    g.add_argument("--data", required=True, help="Excel file, or folder with the GCP-CEM CSV files"); g.add_argument("--sheet", default="X-Y")
    g.add_argument("--year-max", type=int, default=None)
    t = sub.add_parser("train-final")
    t.add_argument("--data", required=True, help="Excel file, or folder with the GCP-CEM CSV files"); t.add_argument("--sheet", default="X-Y")
    t.add_argument("--year-max", type=int, default=None)
    t.add_argument("--out", default="artefact"); t.add_argument("--backtest", default=None)
    t.add_argument("--seeds", type=int, nargs="*")
    t.add_argument("--provenance", choices=["auto", "activity"], default="auto")
    t.add_argument("--variant", default="TPN (full)", help="TPN variant to deploy (an ablation name)")
    t.add_argument("--model", default="N-BEATS", help="model to deploy: N-BEATS (default), TPN, LSTM, ...")
    f = sub.add_parser("forecast")
    f.add_argument("--artefact", required=True); f.add_argument("--history", required=True)
    f.add_argument("--iso", default="XXX"); f.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        from .selftest import run_selftest
        run_selftest(a.out, quick=not a.full)
        return
    cfg = Config()
    if getattr(a, "provenance", None):
        cfg.provenance_mode = a.provenance
    if getattr(a, "year_max", None):
        cfg.year_max = a.year_max
    if getattr(a, "seeds", None):
        cfg.seeds = tuple(a.seeds)
    if a.cmd == "overview":
        from .data import load_any
        from .report import overview_figure
        print(overview_figure(load_any(a.data, cfg, a.sheet), cfg, a.out))
        return
    if a.cmd == "diagnose":
        import pandas as pd
        from .data import load_any, build_sequences, assign_provenance, coverage_report, auto_cutoffs, _fold_sizes
        panel = load_any(a.data, cfg, a.sheet)
        pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
        rep = coverage_report(panel, cfg)
        print(rep.to_string(index=False))
        from .data import source_label_report
        lab = source_label_report(panel, cfg)
        if len(lab):
            print("\nSource labels (production/clinker values, 1990 onwards):")
            print(lab.to_string(index=False, max_colwidth=70))
        P_ = assign_provenance(panel, cfg)
        print("\nProvenance shares of country-years:", (100 * P_["source"].value_counts(normalize=True)).round(1).to_dict())
        S = build_sequences(assign_provenance(panel, cfg), cfg)
        print(f"\n{len(S)} windows; forecast origins t0 by year:")
        print(pd.Series(S.t0).value_counts().sort_index().to_string())
        cuts = auto_cutoffs(S, cfg)
        print(f"\nSuggested cut-offs: {cuts}")
        for k, C in enumerate(cuts):
            print(f"  C={C}: train/val/test windows = {_fold_sizes(S, C, cfg, k == len(cuts) - 1)}")
        return
    if a.cmd == "experiment":
        if a.cutoffs:
            cfg.cutoffs = tuple(a.cutoffs)
        if a.country_meta:
            cfg.country_meta_csv = a.country_meta
        if a.default_clinker is not None:
            cfg.default_clinker_ratio = a.default_clinker
        from .data import load_any
        from .experiment import run_experiment
        run_experiment(load_any(a.data, cfg, a.sheet), cfg, a.out,
                       do_ablation=not a.no_ablation, do_coldstart=not a.no_coldstart)
    elif a.cmd == "train-final":
        from .data import load_any
        from .deploy import train_final
        print(train_final(load_any(a.data, cfg, a.sheet), cfg, a.out, a.backtest, variant=a.variant, model=a.model))
    elif a.cmd == "forecast":
        import pandas as pd
        from .deploy import Forecaster
        res = Forecaster.load(a.artefact).predict(pd.read_csv(a.history), iso=a.iso)
        print(res.to_string(index=False))
        if a.out:
            res.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
