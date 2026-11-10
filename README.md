# cement-co2-forecasting-nbeats

**Cement emission forecasting under data scarcity and heterogeneity**

Code and results for the paper:

> Abdullahi, S. B. *Neural Basis Expansion for Cement Emission Forecasting Under Data Scarcity and
> Heterogeneity.* (under review)

The repository contains a leakage-controlled framework for forecasting national cement process
(calcination) CO2 emissions one to three years ahead. Its core is a global neural basis expansion
model (N-BEATS) that uses only each country's emission history, with prediction intervals calibrated
by activity-data provenance. Twelve comparison models are included: persistence, drift, OLS trend,
damped trend, ridge regression, gradient boosting, LSTM, TCN, static- and learned-graph networks, a
DeepAR-style network, and a provenance-aware temporal prototype network (TPN).

The Python package is named `tpn_cement` for historical reasons; it contains all models, not only the TPN.

## Quick start

```bash
git clone https://github.com/Sbabdullahi/cement-co2-forecasting-nbeats.git
cd cement-co2-forecasting-nbeats
pip install -r requirements.txt

python -m tpn_cement selftest              # leakage tests + full pipeline on synthetic data (~3 min)
python scripts/download_data.py            # fetch the GCP-CEM data from Zenodo into data/gcp_cem
```

A Google Colab notebook that runs every step is in `notebooks/`.

## Reproduce the paper

```bash
python -m tpn_cement diagnose   --data data/gcp_cem                 # coverage, source labels, cut-offs
python -m tpn_cement experiment --data data/gcp_cem --out results    # all tables and figures (~1-1.5 h CPU)
python -m tpn_cement overview   --data data/gcp_cem --out results    # Figure 1
```

`experiment` writes `results.json` (every number, with the configuration), `results_macros.tex` and
`tables/*.tex` (read directly by the manuscript), `figures/`, and `test_predictions.csv` (every test
forecast in kt with the framework's 90% intervals and all comparison models).

The outputs of the run reported in the paper are included in `results/`. Neural network training on a
different machine or PyTorch version can change results in the third decimal place.

## Use the trained forecaster

```bash
python -m tpn_cement train-final --data data/gcp_cem --out artefact --backtest results/results.json
python -m tpn_cement forecast --artefact artefact/tpn_artifact.pt --history my_country.csv --iso MEX
```

`my_country.csv` needs the columns `year, emissions` (kt CO2) and optionally `production, clinker`.
The output gives 1-3 year forecasts in kt with 80% and 90% intervals, the persistence and OLS-trend
forecasts for comparison, and warnings when inputs fall outside the training range.
`artefact/model_card.md` states the intended use: screening and trend analysis, not compliance or
carbon-credit accounting.

## Leakage controls (tested by `selftest`)

1. Every feature at year t uses years <= t only (no full-series means, z-scores, classes or back-fill).
2. Targets are window-relative log changes, so no target scaling statistics exist.
3. Feature scalers are fitted on each fold's training windows only.
4. Purged rolling-origin splits: training targets < validation targets <= cut-off < test targets.
5. Model state at evaluation time depends on training data only.
6. Per-sequence predictions do not depend on batch composition.

The causality test fails, as it should, when a full-series country mean is inserted as a feature.

## Repository layout

| path | content |
|---|---|
| `tpn_cement/config.py` | every constant used in the paper |
| `tpn_cement/data.py` | GCP-CEM loader, provenance scoring, causal features, purged folds |
| `tpn_cement/models.py` | N-BEATS, TPN (with ablation switches), LSTM, TCN, graph, DeepAR-style |
| `tpn_cement/baselines.py` | persistence, drift, OLS trend, damped trend, ridge, gradient boosting |
| `tpn_cement/evaluation.py` | MASE/sMAPE/kt metrics, conformal intervals, cluster bootstrap, DM-HLN, Holm |
| `tpn_cement/experiment.py` | full protocol |
| `tpn_cement/report.py` | LaTeX tables, macros, figures |
| `tpn_cement/deploy.py` | final training, artefact, model card, forecaster |
| `tpn_cement/selftest.py` | synthetic data and leakage tests |
| `scripts/download_data.py` | downloads the data from Zenodo |
| `notebooks/` | Google Colab notebook |
| `results/` | outputs of the run reported in the paper |

## Data

Andrew, R. M. Global CO2 emissions from cement production (GCP-CEM), release 260526.
Zenodo. https://doi.org/10.5281/zenodo.20397304 (CC BY 4.0). The data are downloaded, not redistributed.

## Citation

Please cite the paper and this software (see `CITATION.cff`, or the DOI badge once archived on Zenodo).

## Licence

Code: MIT (see `LICENSE`).
