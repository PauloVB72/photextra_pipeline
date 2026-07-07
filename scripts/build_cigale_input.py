#!/usr/bin/env python3
"""Build CIGALE inputs for N MKW8 targets, both methods:
  - "filters": 10 custom continuum filters + broadband (chances_legacy) + 4 emission lines
  - "spectro": native use_spectro=True with full DESI spectrum (prism FITS) + broadband

Mirrors the recipe already validated in cigale_pilot_3targets_v2_lines and
cigale_pilot_3targets_desi_spectro (units/columns verified against those dirs).
"""
import csv
import os
import sys
import importlib.util

import numpy as np
from astropy.io import fits

sys.path.insert(0, "/home/polo/Escritorio/PHD/code/xpectrafit")
sys.path.insert(0, "/home/polo/Escritorio/PHD/code/photextra_pipeline")
from xpectrafit.filters.custom import CustomFilter
from xpectrafit.filters.integrate import integrate_filters

# shared spectro-method helpers now live in the pipeline package (also used
# by Pipeline both_method="cigale"); the filters method stays local here.
from photextra_pipeline.cigale_integration import (
    BROADBAND_MAP, ERR_FLOOR, broadband_cells,
    convert_desi_to_prism_fits as convert_desi_fits,
    build_spectro_row as _build_spectro_row_shared,
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_specnorm = _load_module(
    "specnorm",
    "/home/polo/Escritorio/PHD/code/photextra_pipeline/"
    "photextra_pipeline/spec_normalization.py")
collect_anchors = _specnorm.collect_anchors
fit_spec_normalization = _specnorm.fit_spec_normalization

SURVEY_DEFAULTS = {
    "Legacy_g": {"lambda_eff": 0.472},
    "Legacy_r": {"lambda_eff": 0.641},
    "Legacy_z": {"lambda_eff": 0.906},
}
FLUX_UNIT = 1e-17
FILTER_CSV = ("/home/polo/Escritorio/PHD/code/photextra_pipeline/"
              "photextra_pipeline/data/chances_continuum_filters.csv")
RUN_DIR = "/home/polo/Escritorio/PHD/CHANCES/MKW8_full_run"

ANCHOR_SURVEYS = ["Legacy_g", "Legacy_r", "Legacy_z"]

LINE_MAP = {
    "Halpha":  "line.H-alpha",
    "Hbeta":   "line.H-beta",
    "OIII5007": "line.OIII-500.8",
    "NII6584":  "line.NII-658.5",
}
LINE_UNIT = 1e-20  # combined.csv line_*_scaled units (1e-17 erg/s/cm2) -> W/m2

C_AA_S = 2.99792458e18


def load_enabled_filters():
    filters = []
    with open(FILTER_CSV) as f:
        for row in csv.DictReader(f):
            if str(row.get("enabled", "1")).strip() not in ("1", "true", "True"):
                continue
            name = "chances." + row["name"].strip()
            filters.append(CustomFilter(name=name,
                                        lambda_min=float(row["lambda_min"]),
                                        lambda_max=float(row["lambda_max"])))
    return filters


def read_row(path):
    with open(path) as f:
        return next(csv.DictReader(f))


def to_float(v):
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return np.nan


def build_filters_row(tid, combined, filters, lambda_eff_um):
    z = to_float(combined.get("z"))
    npz = np.load(os.path.join(RUN_DIR, tid, "spectroscopy", f"desi_{tid}.npz"))
    wave = np.asarray(npz["wave"], float)
    flux = np.asarray(npz["flux"], float) * FLUX_UNIT
    ivar = np.asarray(npz["ivar"], float)
    with np.errstate(divide="ignore"):
        err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan) * FLUX_UNIT
    wmin, wmax = float(np.nanmin(wave)), float(np.nanmax(wave))

    ff = integrate_filters(wave, flux, filters, err)
    anchors = collect_anchors(ANCHOR_SURVEYS, lambda_eff_um, combined,
                              combined, wmin, wmax, "phot_")
    spec_norm, meta = fit_spec_normalization(anchors)

    row = {"id": tid, "redshift": f"{z:.6f}"}
    for name, fx in ff.items():
        f_mjy = fx.flux_fnu_mJy
        e_mjy = fx.flux_fnu_mJy_err
        if not (np.isfinite(f_mjy) and f_mjy > 0):
            row[name] = "nan"; row[name + "_err"] = "nan"
            continue
        piv = fx.lambda_pivot
        s = float(spec_norm.scale_at(piv)) if spec_norm is not None else 1.0
        f_tot = f_mjy * s
        e_tot = (e_mjy * s) if np.isfinite(e_mjy) else np.nan
        e_floor = np.sqrt((e_tot if np.isfinite(e_tot) else 0.0) ** 2
                          + (ERR_FLOOR * f_tot) ** 2)
        row[name] = f"{f_tot:.6e}"
        row[name + "_err"] = f"{e_floor:.6e}"

    row.update(broadband_cells(combined))

    for src, cig in LINE_MAP.items():
        fv = to_float(combined.get(f"line_{src}_flux_scaled"))
        ev = to_float(combined.get(f"line_{src}_flux_err_scaled"))
        if not np.isfinite(fv):
            row[cig] = "nan"; row[cig + "_err"] = "nan"
            continue
        row[cig] = f"{fv * LINE_UNIT:.6e}"
        row[cig + "_err"] = f"{ev * LINE_UNIT:.6e}" if np.isfinite(ev) else "nan"

    return row


def build_spectro_row(tid, combined, spec_dir):
    npz_path = os.path.join(RUN_DIR, tid, "spectroscopy", f"desi_{tid}.npz")
    return _build_spectro_row_shared(tid, combined, npz_path, spec_dir)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out_root = sys.argv[2] if len(sys.argv) > 2 else RUN_DIR

    cands = []
    for tid in sorted(os.listdir(RUN_DIR)):
        tdir = os.path.join(RUN_DIR, tid)
        if not os.path.isdir(tdir):
            continue
        combined = os.path.join(tdir, "products", f"{tid}_combined.csv")
        npz = os.path.join(tdir, "spectroscopy", f"desi_{tid}.npz")
        if os.path.exists(combined) and os.path.exists(npz):
            cands.append(tid)
    targets = cands[:n]
    print(f"Using {len(targets)} targets (requested {n}, {len(cands)} available)")

    filters = load_enabled_filters()
    lambda_eff_um = {s: SURVEY_DEFAULTS.get(s, {}).get("lambda_eff")
                     for s in ANCHOR_SURVEYS}

    filt_dir = os.path.join(out_root, f"cigale_N{n}_filters")
    spec_dir = os.path.join(out_root, f"cigale_N{n}_spectro")
    os.makedirs(filt_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)

    filt_rows, spec_rows = [], []
    for tid in targets:
        combined = read_row(os.path.join(RUN_DIR, tid, "products", f"{tid}_combined.csv"))
        try:
            filt_rows.append(build_filters_row(tid, combined, filters, lambda_eff_um))
        except Exception as e:
            print(f"  [filters] SKIP {tid}: {e}")
        try:
            spec_rows.append(build_spectro_row(tid, combined, spec_dir))
        except Exception as e:
            print(f"  [spectro] SKIP {tid}: {e}")

    # --- filters+broadband+lines data file ---
    bb_cols = list(BROADBAND_MAP.values())
    band_cols = [f.name for f in filters]
    line_cols = list(LINE_MAP.values())
    cols = ["id", "redshift"]
    for c in bb_cols + band_cols + line_cols:
        cols += [c, c + "_err"]
    with open(os.path.join(filt_dir, "cigale_input.txt"), "w") as f:
        f.write("# " + " ".join(cols) + "\n")
        for row in filt_rows:
            f.write(" ".join(str(row.get(c, "nan")) for c in cols) + "\n")
    print(f"Wrote {filt_dir}/cigale_input.txt ({len(filt_rows)} rows)")

    # --- native spectro data file ---
    cols2 = ["id", "redshift"]
    for c in bb_cols:
        cols2 += [c, c + "_err"]
    cols2 += ["spectrum", "mode", "norm"]
    with open(os.path.join(spec_dir, "cigale_input.txt"), "w") as f:
        f.write("# " + " ".join(cols2) + "\n")
        for row in spec_rows:
            f.write(" ".join(str(row.get(c, "nan")) for c in cols2) + "\n")
    print(f"Wrote {spec_dir}/cigale_input.txt ({len(spec_rows)} rows)")


if __name__ == "__main__":
    main()
