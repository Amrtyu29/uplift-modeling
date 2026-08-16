"""Fetch the raw randomized-experiment datasets.

    python scripts/00_download_data.py --dataset hillstrom
    python scripts/00_download_data.py --dataset criteo    # ~460 MB
"""

import argparse
import sys
import urllib.request

import _bootstrap  # noqa: F401

from uplift.config import DATA_RAW, get_spec

URLS = {
    "hillstrom": (
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ),
    "criteo": "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
}


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100 * done / total_size
    sys.stdout.write(f"\r  {done/1e6:8.1f} / {total_size/1e6:.1f} MB ({pct:5.1f}%)")
    sys.stdout.flush()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="hillstrom", choices=sorted(URLS))
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = p.parse_args()

    spec = get_spec(args.dataset)
    dest = DATA_RAW / spec.filename
    if dest.exists() and not args.force:
        print(f"{dest} already present ({dest.stat().st_size/1e6:.1f} MB); use --force to re-download.")
        return

    print(f"Downloading {args.dataset} -> {dest}")
    urllib.request.urlretrieve(URLS[args.dataset], dest, reporthook=_progress)
    print(f"\nDone: {dest.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
