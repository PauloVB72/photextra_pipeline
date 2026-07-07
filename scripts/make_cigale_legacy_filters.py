"""Generate CIGALE filter .dat files from the real DECam / Legacy Survey g/r/z
transmission curves and register them in the pcigale filter database.

Source of truth: the DECam system-throughput curves shipped with ``hostphot``
(``hostphot/filters/LegacySurvey/DECAM_{g,r,z}.dat``). These are the SAME curves
the photometry itself is implicitly defined against, since photextra_pipeline's
downloader fetches Legacy Survey cutouts via hostphot -- so they are the
scientifically correct instrument responses for these bands (not a generic DECam
curve from elsewhere).

The hostphot files are two-column ASCII, wavelength[Å] / total system
throughput (dimensionless, including atmosphere at airmass 1.4; peak ~0.36/0.51/
0.55 for g/r/z). They are NOT in the header layout CIGALE's add_filters() wants
(the g file has free-text comment lines, r/z have none), so we rewrite them with
the exact 3-line header CIGALE expects:  ``# name`` / ``# energy`` / ``# desc``.

Filter type is registered as 'energy' -- the correct convention for a REAL
instrument throughput curve integrated against F_lambda with energy weighting,
matching the other real broadband filters already in the CIGALE default database
(galex.FUV, wise.W1, sdss.*, ...). This is deliberately DIFFERENT from the
custom chances.* tophats, which use 'photon' to match xpectrafit's own
T*lambda synthetic-photometry convention for reconstructing fluxes FROM a
spectrum -- a different physical setup than modelling a photon-counting
detector's real response.

Names are registered with a 'chances_legacy.' prefix
(chances_legacy.DECam_g / _r / _z) to namespace them and avoid colliding with
anything in the default database (there is no native DECam/Legacy filter in this
CIGALE build -- verified with `pcigale-filters list | grep -i decam`).

Usage:
    conda activate cigale
    python make_cigale_legacy_filters.py [--srcdir PATH] [--outdir PATH]
                                         [--no-register]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

PREFIX = "chances_legacy"

# Default location of the hostphot DECam curves (photextra conda env).
DEFAULT_SRCDIR = Path(
    "/home/polo/miniconda3/envs/photextra/lib/python3.14/site-packages/"
    "hostphot/filters/LegacySurvey"
)

# hostphot filename -> registered CIGALE band suffix
BANDS = {
    "DECAM_g.dat": "DECam_g",
    "DECAM_r.dat": "DECam_r",
    "DECAM_z.dat": "DECam_z",
}


def read_hostphot_curve(path):
    """Read a two-column wavelength[Å]/transmission hostphot .dat, skipping any
    ``#`` comment lines. Returns (wl, tr) as float arrays, sanity-checked."""
    wl, tr = np.genfromtxt(path, comments="#", unpack=True)
    if np.any(~np.isfinite(wl)) or np.any(~np.isfinite(tr)):
        raise ValueError(f"{path}: non-finite values in curve")
    if np.any(tr < 0):
        raise ValueError(f"{path}: negative transmission values")
    if wl[0] > wl[-1]:  # ensure ascending wavelength
        wl, tr = wl[::-1], tr[::-1]
    return wl, tr


def write_filter_dat(src_path, suffix, outdir):
    name = f"{PREFIX}.{suffix}"
    wl, tr = read_hostphot_curve(src_path)
    pivot = np.sqrt(np.trapz(tr, wl) / np.trapz(tr / wl**2, wl))
    desc = (f"DECam/Legacy Survey {suffix.split('_')[-1]}-band total system "
            f"throughput (hostphot DECAM curve, airmass 1.4); pivot "
            f"{pivot:.0f}A")

    path = outdir / f"{suffix}.dat"
    with open(path, "w") as f:
        f.write(f"# {name}\n")
        f.write("# energy\n")
        f.write(f"# {desc}\n")
        for w, t in zip(wl, tr):
            f.write(f"{w:.2f} {t:.6e}\n")
    return path, name, pivot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--srcdir", default=str(DEFAULT_SRCDIR))
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parent / "cigale_legacy_filters_out"))
    ap.add_argument("--no-register", action="store_true",
                    help="only write the .dat files, skip 'pcigale-filters add'")
    args = ap.parse_args()

    srcdir = Path(args.srcdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = []
    for fname, suffix in BANDS.items():
        src = srcdir / fname
        if not src.exists():
            print(f"MISSING source: {src}", file=sys.stderr)
            sys.exit(2)
        path, name, pivot = write_filter_dat(src, suffix, outdir)
        written.append(path)
        print(f"wrote {path} -> registers as '{name}' (pivot {pivot:.0f} A)")

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
