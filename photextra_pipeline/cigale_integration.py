"""CIGALE native-spectroscopy SED fitting for the pipeline (cigale_run=True).

Wires the validated "spectro" recipe (docs/cigale_tutorial/TUTORIAL.md) into
the Pipeline: for each target with a combined product and a cached DESI
spectrum, it

1. converts the cached ``desi_{tid}.npz`` spectrum to the prism-style FITS
   CIGALE expects (WAVELENGTH micron / FLUX mJy / FLUX_ERROR mJy),
2. writes a ``cigale_input.txt`` row: broadband photometry (mJy, 10% error
   floor) + the ``spectrum``/``mode``/``norm`` columns of the cigale-spec
   fork (use_spectro=True),
3. copies the VALIDATED ``pcigale.ini``/``pcigale.ini.spec``/``desi_disp.txt``
   templates packaged under ``photextra_pipeline/data/cigale/`` (never
   regenerated with ``pcigale genconf`` — that resets the SED-module grid,
   see the tutorial's documented gotcha),
4. runs ``pcigale run`` once for the whole batch (CIGALE fits every row of
   one input file in a single run — much faster than per-target calls) in
   the ``cigale`` conda environment,
5. parses ``out/results.txt`` back into per-target ``cigale_*`` columns
   (stellar mass, SFR, chi2, ...) that the Pipeline appends to each
   ``{tid}_combined`` product.

The same helpers are imported by ``scripts/build_cigale_input.py`` so the
standalone script and the pipeline share one implementation.

Besides the combined ("both") method, :func:`run_cigale_for_targets` also
supports mode="spectroscopy" (spectrum-only rows: fixed z, NO broadband
columns — less constrained than a combined fit) and mode="photometry"
(broadband-only rows: NO spectrum and the ``redshift`` column left NaN, so
CIGALE scans the redshifting module's z grid per object — its native
chi2 photometric redshift — and the best-fit z comes back as
``cigale_z_phot``; see ``Z_PHOT_RESULT_COLUMNS``).

pcigale.ini customization: the packaged templates under
``photextra_pipeline/data/cigale/`` (``pcigale.ini``, ``pcigale.ini.spec``,
``desi_disp.txt``) are the single source of truth and are copied AS-IS into
every work dir. NO pipeline code path writes or regenerates their
SED-module parameters (and ``pcigale genconf`` must never be used — it
resets the validated grid). To change ANY CIGALE parameter — e.g. give the
``redshifting`` module a LIST of redshifts for photo-z fitting in
mode="photometry", or set ``use_spectro = False`` for photometry-only
input files — edit that packaged ``pcigale.ini`` BY HAND.
"""

import csv
import logging
import os
import shutil
import subprocess

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

C_AA_S = 2.99792458e18  # speed of light in Angstrom/s

# broadband column mapping: pipeline survey name -> CIGALE filter name
# (chances_legacy.* are the real DECam curves registered by
# scripts/make_cigale_legacy_filters.py)
BROADBAND_MAP = {
    "GALEX_FUV": "galex.FUV",
    "GALEX_NUV": "galex.NUV",
    "Legacy_g":  "chances_legacy.DECam_g",
    "Legacy_r":  "chances_legacy.DECam_r",
    "Legacy_z":  "chances_legacy.DECam_z",
    "WISE_W1":   "WISE1",
    "WISE_W2":   "WISE2",
}
ERR_FLOOR = 0.10  # 10% error floor added in quadrature (validated recipe)

# packaged validated templates (copied from the MKW8 cigale_N446_spectro run)
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "data", "cigale")

# results.txt column -> combined-product column (clearly cigale_-prefixed so
# they are never confused with the own-method columns)
RESULT_COLUMNS = {
    "bayes.stellar.m_star": "cigale_stellar_mass",
    "bayes.stellar.m_star_err": "cigale_stellar_mass_err",
    "bayes.sfh.sfr": "cigale_sfr",
    "bayes.sfh.sfr_err": "cigale_sfr_err",
    "bayes.sfh.sfr100Myrs": "cigale_sfr100Myrs",
    "bayes.sfh.sfr100Myrs_err": "cigale_sfr100Myrs_err",
    "bayes.sfh.age": "cigale_age",
    "bayes.sfh.age_err": "cigale_age_err",
    "bayes.attenuation.E_BVs": "cigale_E_BVs",
    "bayes.attenuation.E_BVs_err": "cigale_E_BVs_err",
    "bayes.dust.luminosity": "cigale_dust_luminosity",
    "bayes.dust.luminosity_err": "cigale_dust_luminosity_err",
    "best.chi_square": "cigale_chi2",
    "best.reduced_chi_square": "cigale_chi2_red",
}

# extra columns parsed ONLY in mode="photometry" (photo-z): CIGALE's
# best-fit redshift from the chi2 scan over the redshifting module's grid.
# Column names verified against a real cigale-spec results.txt
# (best.universe.redshift / bayes.universe.redshift[_err]).
Z_PHOT_RESULT_COLUMNS = {
    "best.universe.redshift": "cigale_z_phot",
    "bayes.universe.redshift": "cigale_z_phot_bayes",
    "bayes.universe.redshift_err": "cigale_z_phot_bayes_err",
}


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def convert_desi_to_prism_fits(npz_path, out_path):
    """Cached DESI ``.npz`` (wave A, flux 1e-17 erg/s/cm2/A, ivar) -> prism
    FITS (WAVELENGTH micron, FLUX mJy, FLUX_ERROR mJy) for CIGALE's
    ``read_prism()``. Physics: F_lambda -> F_nu with
    ``conv = wave**2 / c * 1e26``."""
    d = np.load(npz_path)
    wave = np.asarray(d["wave"], float)
    flux = np.asarray(d["flux"], float)
    ivar = np.asarray(d["ivar"], float)
    order = np.argsort(wave)
    wave, flux, ivar = wave[order], flux[order], ivar[order]
    flam = flux * 1e-17
    with np.errstate(divide="ignore"):
        flam_err = np.where(ivar > 0, 1.0 / np.sqrt(ivar) * 1e-17, np.nan)
    conv = wave ** 2 / C_AA_S * 1e26
    fnu = flam * conv
    fnu_err = flam_err * conv
    good = np.isfinite(fnu) & np.isfinite(fnu_err) & (fnu_err > 0)
    wave, fnu, fnu_err = wave[good], fnu[good], fnu_err[good]
    col_wave = fits.Column(name="WAVELENGTH", format="D", array=wave / 1e4)
    col_flux = fits.Column(name="FLUX", format="D", array=fnu)
    col_ferr = fits.Column(name="FLUX_ERROR", format="D", array=fnu_err)
    hdu = fits.BinTableHDU.from_columns([col_wave, col_flux, col_ferr])
    hdu.header["TUNIT1"] = "micron"
    hdu.header["TUNIT2"] = "mJy"
    hdu.header["TUNIT3"] = "mJy"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(out_path, overwrite=True)
    return out_path


def broadband_cells(combined):
    """Broadband flux cells (mJy, 10% error floor) from a combined-row dict."""
    cells = {}
    for survey, cig in BROADBAND_MAP.items():
        f_mjy = _to_float(combined.get(f"phot_{survey}_flux_mjy"))
        e_mjy = _to_float(combined.get(f"phot_{survey}_flux_err_mjy"))
        if not (np.isfinite(f_mjy) and f_mjy > 0):
            cells[cig] = "nan"
            cells[cig + "_err"] = "nan"
            continue
        e = e_mjy if np.isfinite(e_mjy) else 0.0
        e_floor = np.sqrt(e ** 2 + (ERR_FLOOR * abs(f_mjy)) ** 2)
        cells[cig] = f"{f_mjy:.6e}"
        cells[cig + "_err"] = f"{e_floor:.6e}"
    return cells


def build_spectro_row(tid, combined, npz_path, work_dir, mode="both"):
    """One cigale_input.txt row for a target, shaped by the pipeline mode.

    combined: dict of the target's product row (strings or floats).
    npz_path: cached DESI spectrum (ignored for mode="photometry"). The
    prism FITS is written into work_dir.

    mode="both" (default): fixed z + broadband photometry + spectrum
      (the validated native-spectroscopy method).
    mode="spectroscopy": fixed z + spectrum, NO broadband columns (they
      are written as ``nan`` and CIGALE skips non-finite fluxes per
      object). Spectrum-only fits are less constrained than combined
      photometry+spectrum fits.
    mode="photometry": broadband photometry only, NO spectrum, and the
      ``redshift`` cell left as ``nan`` — CIGALE treats an unknown
      (non-finite/negative) redshift by chi2-scanning the full z grid of
      the ``redshifting`` module (photo-z). This REQUIRES the packaged
      ``pcigale.ini`` to be hand-edited with a LIST of redshifts (and
      ``use_spectro = False``); the pipeline never writes that file.
    """
    row = {"id": str(tid)}
    if mode == "photometry":
        # unknown z -> NaN; CIGALE's Observation treats any redshift that
        # is not >= 0 as "unknown" and fits the full redshifting grid.
        row["redshift"] = "nan"
    else:
        z = _to_float(combined.get("z"))
        if not np.isfinite(z):
            raise ValueError(f"{tid}: no finite redshift in product row")
        row["redshift"] = f"{z:.6f}"
    if mode != "spectroscopy":
        row.update(broadband_cells(combined))
    if mode != "photometry":
        if npz_path is None or not os.path.exists(npz_path):
            raise FileNotFoundError(f"no cached DESI spectrum {npz_path}")
        fits_path = os.path.join(work_dir, f"desi_spec_{tid}.fits")
        convert_desi_to_prism_fits(npz_path, fits_path)
        row["spectrum"] = fits_path
        row["mode"] = "desi"     # cigale-spec fork column names, NOT the
        row["norm"] = "wave"     # official spec_name/disperser/norm_method
    return row


def write_cigale_input(rows, work_dir):
    """Write cigale_input.txt (spectro format) into work_dir.

    The spectrum/mode/norm columns are only written when at least one row
    carries a spectrum (photometry-only inputs must NOT have a ``spectrum``
    column: with ``use_spectro = True`` CIGALE checks every row's spectrum
    file exists).
    """
    cols = ["id", "redshift"]
    for c in BROADBAND_MAP.values():
        cols += [c, c + "_err"]
    if any("spectrum" in row for row in rows):
        cols += ["spectrum", "mode", "norm"]
    path = os.path.join(work_dir, "cigale_input.txt")
    with open(path, "w") as fh:
        fh.write("# " + " ".join(cols) + "\n")
        for row in rows:
            fh.write(" ".join(str(row.get(c, "nan")) for c in cols) + "\n")
    return path


def prepare_work_dir(work_dir, mode="both"):
    """Copy the validated pcigale.ini/.spec + desi_disp.txt templates.

    mode="photometry" copies ``pcigale_photoz.ini[.spec]`` instead of the
    default ``pcigale.ini[.spec]``: that variant sets an EXPLICIT redshift
    grid (every input row has redshift=NaN, so CIGALE chi2-scans the grid
    per object -- photo-z) and ``use_spectro = False`` (no spectrum column
    in photometry-only rows). The default template's ``redshift = `` stays
    EMPTY so mode="both"/"spectroscopy" keep using each target's EXACT
    known z (auto-built from the input file) -- do NOT point those modes
    at the photo-z template, it would snap known redshifts to the grid and
    degrade derived stellar mass/SFR (see config_xmask.yaml cigale_run
    comment).

    NEVER regenerate either ini with ``pcigale genconf``: it resets every
    [sed_modules_params] grid value to its default (tutorial section 3.5).
    """
    os.makedirs(work_dir, exist_ok=True)
    ini_stem = "pcigale_photoz" if mode == "photometry" else "pcigale"
    files = [(f"{ini_stem}.ini", "pcigale.ini"),
             (f"{ini_stem}.ini.spec", "pcigale.ini.spec"),
             ("desi_disp.txt", "desi_disp.txt")]
    for src_name, dst_name in files:
        src = os.path.join(_TEMPLATE_DIR, src_name)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"CIGALE template {src} missing — reinstall/copy the "
                f"validated baseline from the MKW8 cigale_N446_spectro run")
        shutil.copy(src, os.path.join(work_dir, dst_name))


def run_pcigale(work_dir, conda_env="cigale", timeout_s=7200):
    """Run ``pcigale run`` in work_dir inside the given conda environment."""
    cmd = ["conda", "run", "-n", conda_env, "--no-capture-output",
           "pcigale", "run"]
    logger.info("running CIGALE in %s: %s", work_dir, " ".join(cmd))
    # CIGALE refuses to overwrite an existing out/; move it aside
    out_dir = os.path.join(work_dir, "out")
    if os.path.isdir(out_dir):
        stamp = 1
        while os.path.isdir(f"{out_dir}.{stamp}"):
            stamp += 1
        os.rename(out_dir, f"{out_dir}.{stamp}")
    proc = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True,
                          timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"pcigale run failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    return os.path.join(work_dir, "out", "results.txt")


def parse_cigale_results(results_path, columns=None):
    """Parse out/results.txt -> {target_id: {cigale_*: value}}.

    columns: {results.txt column: product column} mapping; defaults to
    ``RESULT_COLUMNS``. Missing columns parse to NaN.
    """
    if columns is None:
        columns = RESULT_COLUMNS
    with open(results_path) as fh:
        header = fh.readline().split()
        out = {}
        for line in fh:
            vals = line.split()
            if len(vals) != len(header):
                continue
            rec = dict(zip(header, vals))
            tid = str(rec.get("id"))
            out[tid] = {new: _to_float(rec.get(old))
                        for old, new in columns.items()}
    return out


def read_combined_row(combined_csv):
    """First row of a {tid}_combined.csv as a plain dict of strings."""
    with open(combined_csv) as fh:
        return next(csv.DictReader(fh))


def run_cigale_for_targets(target_specs, work_dir, conda_env="cigale",
                           mode="both"):
    """Full batch: build inputs, run pcigale once, return parsed results.

    target_specs: list of (tid, product_row_dict, npz_path) tuples
    (npz_path may be None for mode="photometry"). Rows that cannot be
    built (missing spectrum/z) are skipped with a warning.
    mode: "both" (default), "spectroscopy" or "photometry"; see
    :func:`build_spectro_row` for the row shape per mode. In
    mode="photometry" the photo-z columns (``cigale_z_phot`` etc.,
    ``Z_PHOT_RESULT_COLUMNS``) are parsed in addition to the standard
    ``RESULT_COLUMNS``.
    Returns {tid: {cigale_*: value}} for the targets CIGALE fitted.
    """
    prepare_work_dir(work_dir, mode=mode)
    rows = []
    for tid, combined, npz_path in target_specs:
        try:
            rows.append(build_spectro_row(tid, combined, npz_path, work_dir,
                                          mode=mode))
        except Exception as exc:
            logger.warning("cigale: skipping %s: %s", tid, exc)
    if not rows:
        logger.warning("cigale: no usable targets; nothing to run")
        return {}
    write_cigale_input(rows, work_dir)
    results_path = run_pcigale(work_dir, conda_env=conda_env)
    columns = dict(RESULT_COLUMNS)
    if mode == "photometry":
        columns.update(Z_PHOT_RESULT_COLUMNS)
    return parse_cigale_results(results_path, columns)
