"""Download the five GCP-CEM files used in the paper from Zenodo.

Usage:  python scripts/download_data.py [--record 20397304] [--out data/gcp_cem]

Dataset: Andrew, R. M., Global CO2 emissions from cement production (GCP-CEM),
release 260526, https://doi.org/10.5281/zenodo.20397304 (CC BY 4.0).
The data are not redistributed in this repository; this script fetches them.
"""
import argparse
import os
import urllib.parse
import urllib.request

FILES = [
    "0. GCP-CEM.csv",
    "1. annual_cement_production.csv",
    "2. annual_clinker_production.csv",
    "1. annual_cement_production_source.csv",
    "2. annual_clinker_production_source.csv",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="20397304", help="Zenodo record id of the GCP-CEM release")
    ap.add_argument("--out", default="data/gcp_cem")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in FILES:
        url = f"https://zenodo.org/records/{a.record}/files/{urllib.parse.quote(f)}?download=1"
        dest = os.path.join(a.out, f)
        print(f"downloading {f} ...", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"  {os.path.getsize(dest) / 1e3:.0f} kB")
    print(f"Done. Use:  python -m tpn_cement experiment --data {a.out}")


if __name__ == "__main__":
    main()
