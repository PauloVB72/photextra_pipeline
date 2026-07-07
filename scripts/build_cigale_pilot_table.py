#!/usr/bin/env python3
"""Build the CIGALE input table for the 3-target MKW8 pilot.

For each target:
  * integrate the cached DESI spectrum through the 10 enabled CHANCES custom
    continuum filters (fiber-aperture synthetic photometry), using xpectrafit's
    own filter machinery;
  * fit the pipeline's wavelength-dependent SpecNorm from Legacy g/r/z
    (total-aperture imaging vs synthetic fluxes) and aperture-correct the
    custom-band fluxes to total-light scale;
  * assemble broadband (GALEX, Legacy->SDSS, WISE) + custom bands into one
    pcigale data_file, with a 10% error floor in quadrature.

Read-only w.r.t. the pipeline; writes only into the pilot working dir.
"""
import csv
import json
import os
import sys

import importlib.util

import numpy as np

# --- imports from the two production packages (read-only use) --------------
# xpectrafit imports cleanly; photextra's package __init__ pulls heavy deps
# (reproject, ...) absent from the cigale env, so load spec_normalization.py
# directly by file path (it needs only numpy).
sys.path.insert(0, "/home/polo/Escritorio/PHD/code/xpectrafit")
from xpectrafit.filters.custom import CustomFilter                 # noqa: E402
from xpectrafit.filters.integrate import integrate_filters         # noqa: E402


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

# lambda_eff (micron) from photextra_pipeline.downloader.SURVEY_DEFAULTS
SURVEY_DEFAULTS = {
    "Legacy_g": {"lambda_eff": 0.472},
    "Legacy_r": {"lambda_eff": 0.641},
    "Legacy_z": {"lambda_eff": 0.906},
}

FLUX_UNIT = 1e-17          # DESI/SDSS spectra stored in 1e-17 erg/s/cm2/A
FILTER_CSV = ("/home/polo/Escritorio/PHD/code/photextra_pipeline/"
              "photextra_pipeline/data/chances_continuum_filters.csv")
RUN_DIR = "/home/polo/Escritorio/PHD/CHANCES/MKW8_full_run"
OUT_DIR = os.path.join(RUN_DIR, "cigale_pilot_3targets")

TARGETS = ["2842599849197568", "2706266447151104", "2842599874363392"]

# Broadband survey -> CIGALE filter name (mapped by pivot wavelength).
# Legacy (DECaLS grz) mapped to closest available SDSS filters; DECam curves
# are not registered in this CIGALE build. Legacy_r (641nm) vs sdss.rp (617nm)
# is the largest mismatch; acceptable for a smoke test.
BROADBAND_MAP = {
    "GALEX_FUV": "galex.FUV",
    "GALEX_NUV": "galex.NUV",
    "Legacy_g":  "sdss.gp",
    "Legacy_r":  "sdss.rp",
    "Legacy_z":  "sdss.zp",
    "WISE_W1":   "WISE1",
    "WISE_W2":   "WISE2",
}
ANCHOR_SURVEYS = ["Legacy_g", "Legacy_r", "Legacy_z"]
ERR_FLOOR = 0.10           # 10% flux error floor added in quadrature


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
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    filters = load_enabled_filters()
    lambda_eff_um = {s: SURVEY_DEFAULTS.get(s, {}).get("lambda_eff")
                     for s in ANCHOR_SURVEYS}

    rows = []          # cigale table rows
    report = {}        # per-target diagnostics

    for tid in TARGETS:
        tdir = os.path.join(RUN_DIR, tid, "products")
        combined = read_row(os.path.join(tdir, f"{tid}_combined.csv"))
        z = to_float(combined.get("z"))

        npz = np.load(os.path.join(RUN_DIR, tid, "spectroscopy", f"desi_{tid}.npz"))
        wave = np.asarray(npz["wave"], float)          # observed-frame Angstrom
        flux = np.asarray(npz["flux"], float) * FLUX_UNIT   # -> erg/s/cm2/A
        ivar = np.asarray(npz["ivar"], float)
        with np.errstate(divide="ignore"):
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan) * FLUX_UNIT

        wmin, wmax = float(np.nanmin(wave)), float(np.nanmax(wave))

        # --- Step 1: synthetic photometry through custom bands (fiber) ------
        ff = integrate_filters(wave, flux, filters, err)

        # --- Step 2: fit SpecNorm (total-aperture aperture correction) ------
        anchors = collect_anchors(ANCHOR_SURVEYS, lambda_eff_um, combined,
                                  combined, wmin, wmax, "phot_")
        spec_norm, meta = fit_spec_normalization(anchors)

        # --- assemble one cigale row ---------------------------------------
        row = {"id": tid, "redshift": f"{z:.6f}"}
        usable = {}
        scales = {}
        for name, fx in ff.items():
            f_mjy = fx.flux_fnu_mJy
            e_mjy = fx.flux_fnu_mJy_err
            if not (np.isfinite(f_mjy) and f_mjy > 0):
                row[name] = "nan"; row[name + "_err"] = "nan"
                usable[name] = False
                continue
            piv = fx.lambda_pivot
            s = float(spec_norm.scale_at(piv)) if spec_norm is not None else 1.0
            f_tot = f_mjy * s
            e_tot = (e_mjy * s) if np.isfinite(e_mjy) else np.nan
            # 10% error floor in quadrature
            e_floor = np.sqrt((e_tot if np.isfinite(e_tot) else 0.0) ** 2
                              + (ERR_FLOOR * f_tot) ** 2)
            row[name] = f"{f_tot:.6e}"
            row[name + "_err"] = f"{e_floor:.6e}"
            usable[name] = True
            scales[name] = (float(piv), s, f_tot, e_floor)

        # --- broadband totals ----------------------------------------------
        for survey, cig in BROADBAND_MAP.items():
            f_mjy = to_float(combined.get(f"phot_{survey}_flux_mjy"))
            e_mjy = to_float(combined.get(f"phot_{survey}_flux_err_mjy"))
            # mask non-detections (<=0 flux) as missing rather than feeding
            # CIGALE a negative measurement
            if not (np.isfinite(f_mjy) and f_mjy > 0):
                row[cig] = "nan"; row[cig + "_err"] = "nan"
                continue
            e = e_mjy if np.isfinite(e_mjy) else 0.0
            e_floor = np.sqrt(e ** 2 + (ERR_FLOOR * abs(f_mjy)) ** 2)
            row[cig] = f"{f_mjy:.6e}"
            row[cig + "_err"] = f"{e_floor:.6e}"

        rows.append(row)
        report[tid] = {
            "z": z,
            "norm_degree": meta["degree"],
            "norm_coeffs": meta["coeffs"],
            "norm_lam0": meta["lam0"],
            "norm_n_anchors": meta["n_anchors"],
            "norm_names_kept": meta["names_kept"],
            "norm_factor_at_lam0": (float(spec_norm.scale_at(meta["lam0"]))
                                    if spec_norm is not None else None),
            "usable_bands": [k for k, v in usable.items() if v],
            "unusable_bands": [k for k, v in usable.items() if not v],
            "per_band_scale": {k: {"pivot_A": v[0], "scale": v[1],
                                   "flux_tot_mjy": v[2], "err_mjy": v[3]}
                               for k, v in scales.items()},
        }

    # --- write cigale data_file (space-separated ASCII) --------------------
    band_cols = [f.name for f in filters]
    bb_cols = list(BROADBAND_MAP.values())
    cols = ["id", "redshift"]
    for c in bb_cols + band_cols:
        cols += [c, c + "_err"]

    data_path = os.path.join(OUT_DIR, "cigale_input.txt")
    with open(data_path, "w") as f:
        f.write("# " + " ".join(cols) + "\n")
        for row in rows:
            f.write(" ".join(str(row.get(c, "nan")) for c in cols) + "\n")

    with open(os.path.join(OUT_DIR, "pilot_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # console summary
    print("Wrote", data_path)
    print("Columns:", len(cols), "| broadband:", bb_cols)
    print("Custom bands:", band_cols)
    for tid, r in report.items():
        print(f"\n=== {tid}  z={r['z']:.4f} ===")
        print(f"  SpecNorm degree={r['norm_degree']} coeffs={r['norm_coeffs']} "
              f"lam0={r['norm_lam0']:.1f} anchors={r['norm_names_kept']}")
        print(f"  factor@lam0={r['norm_factor_at_lam0']:.3f}")
        print(f"  usable custom bands ({len(r['usable_bands'])}): {r['usable_bands']}")
        if r["unusable_bands"]:
            print(f"  UNUSABLE: {r['unusable_bands']}")
        for b, s in r["per_band_scale"].items():
            print(f"    {b}: pivot={s['pivot_A']:.0f}A scale={s['scale']:.3f} "
                  f"F_tot={s['flux_tot_mjy']:.4g} mJy")


if __name__ == "__main__":
    main()
