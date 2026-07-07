"""Generate CIGALE filter .dat files from the CHANCES custom continuum filter
table and register them in the pcigale filter database.

Source of truth: photextra_pipeline/data/chances_continuum_filters.csv
(columns: name, lambda_min, lambda_max, enabled, notes -- Angstrom, tophat).

Filter type is registered as 'photon' (T weighted by wavelength on import,
CIGALE then stores it internally as energy-type), matching xpectrafit's own
synthetic-photometry convention in xpectrafit/filters/integrate.py
(photon-counting: <f> = int f*T*lam dlam / int T*lam dlam). Using the same
convention in both places means the pipeline's synthetic fluxes and CIGALE's
model fluxes are computed the same way for these bands.

Names are registered with a 'chances.' prefix (e.g. 'chances.M3992') to avoid
colliding with a stale pre-revision filter set found already registered under
the bare names (M3992, M4542, ..., 09265) in an orphaned pcigale database on
this machine.

Usage:
    conda activate cigale
    python make_cigale_filters.py [--csv PATH] [--outdir PATH] [--no-register]
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

PREFIX = "chances"
STEP_AA = 1.0       # sampling step inside the tophat (Angstrom)
PAD_AA = 5.0         # a few zero-transmission points just outside the edges


def read_filter_table(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "name": row["name"].strip(),
                "lambda_min": float(row["lambda_min"]),
                "lambda_max": float(row["lambda_max"]),
                "enabled": bool(int(row.get("enabled", "1"))),
                "notes": row.get("notes", "").strip(),
            })
    return rows


def write_filter_dat(row, outdir):
    lo, hi = row["lambda_min"], row["lambda_max"]
    name = f"{PREFIX}.{row['name']}"
    wl_in = np.arange(lo, hi + STEP_AA, STEP_AA)
    tr_in = np.ones_like(wl_in)
    wl = np.concatenate(([lo - PAD_AA], wl_in, [hi + PAD_AA]))
    tr = np.concatenate(([0.0], tr_in, [0.0]))

    path = outdir / f"{row['name']}.dat"
    with open(path, "w") as f:
        f.write(f"# {name}\n")
        f.write("# photon\n")
        f.write(f"# CHANCES continuum tophat {lo:.0f}-{hi:.0f}A"
                f"{' -- ' + row['notes'] if row['notes'] else ''}\n")
        for w, t in zip(wl, tr):
            f.write(f"{w:.2f} {t:.6f}\n")
    return path, name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(
        Path(__file__).resolve().parent.parent / "photextra_pipeline" / "data"
        / "chances_continuum_filters.csv"))
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parent / "cigale_filters_out"))
    ap.add_argument("--include-disabled", action="store_true",
                     help="also generate/register filters with enabled=0 "
                          "(e.g. O9265)")
    ap.add_argument("--no-register", action="store_true",
                     help="only write the .dat files, skip 'pcigale-filters add'")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_filter_table(args.csv)
    written = []
    for row in rows:
        if not row["enabled"] and not args.include_disabled:
            print(f"skip (disabled): {row['name']}")
            continue
        path, name = write_filter_dat(row, outdir)
        written.append(path)
        print(f"wrote {path} -> registers as '{name}'")

    if not written:
        print("nothing to register")
        return

    if args.no_register:
        print("skipping registration (--no-register)")
        return

    cmd = ["pcigale-filters", "add"] + [str(p) for p in written]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
