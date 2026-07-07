"""Helpers for the example notebook: N synthetic tophat continuum filters
sampling the DESI observed wavelength range, fed to CIGALE's narrowband
"filters" method (docs/cigale_tutorial/TUTORIAL.md, 'Metodo filtros').

This generalizes scripts/make_cigale_filters.py (which registers the 10
hand-designed CHANCES continuum tophats) to an arbitrary N of evenly spaced
tophats, and reuses the exact validated input-building recipe of
scripts/build_cigale_input.py (integrate_filters + spec-normalization
anchored on Legacy g/r/z + 10% error floor + broadband cells).
"""
import csv
import os
import shutil
import subprocess

import numpy as np

PIPE_DIR = "/home/polo/Escritorio/PHD/code/photextra_pipeline"
RUN_DIR = "/home/polo/Escritorio/PHD/CHANCES/MKW8_full_run"
PILOT_INI_DIR = os.path.join(RUN_DIR, "cigale_pilot_3targets_v2_lines")

# same conventions as scripts/make_cigale_filters.py
STEP_AA = 1.0   # sampling step inside each tophat (Angstrom)
PAD_AA = 5.0    # zero-transmission padding just outside the edges

# observed-frame windows to AVOID at low N (telluric bands, per the notes in
# photextra_pipeline/data/chances_continuum_filters.csv, plus the strong
# emission-line complexes redshifted to the MKW8 cluster z ~ 0.028)
_TELLURIC = [(6850.0, 6950.0),    # O2-B
             (7580.0, 7700.0),    # O2-A
             (8100.0, 8350.0),    # H2O moderate
             (8900.0, 9700.0)]    # H2O strong
_REST_LINES = [3727.0, 4861.0, 4959.0, 5007.0, 6548.0, 6563.0, 6584.0,
               6717.0, 6731.0]
_Z_CLUSTER = 0.028
_LINE_HALFWIDTH = 35.0  # Angstrom, observed frame

AVOID_BANDS = list(_TELLURIC) + [
    (l * (1 + _Z_CLUSTER) - _LINE_HALFWIDTH,
     l * (1 + _Z_CLUSTER) + _LINE_HALFWIDTH) for l in _REST_LINES]


def make_synthetic_filters(n, wmin=3650.0, wmax=9800.0, avoid_at_n=30):
    """N evenly spaced tophat (lo, hi) windows spanning [wmin, wmax].

    For n <= avoid_at_n the windows are trimmed against AVOID_BANDS
    (telluric + strong emission-line complexes at the cluster redshift),
    keeping the largest clean sub-window -- mimicking how the 10 official
    CHANCES continuum filters were designed by hand. At high n the windows
    are so narrow that a few contaminated ones do not matter, and even
    spacing is used as-is.

    Returns a list of dicts: {name, lambda_min, lambda_max}.
    """
    edges = np.linspace(wmin, wmax, n + 1)
    gap = min(2.0, 0.05 * (edges[1] - edges[0]))
    out = []
    for i in range(n):
        lo, hi = edges[i] + gap, edges[i + 1] - gap
        if n <= avoid_at_n:
            segs = [(lo, hi)]
            for alo, ahi in AVOID_BANDS:
                nxt = []
                for slo, shi in segs:
                    if ahi <= slo or alo >= shi:   # no overlap
                        nxt.append((slo, shi))
                        continue
                    if slo < alo:
                        nxt.append((slo, min(alo, shi)))
                    if shi > ahi:
                        nxt.append((max(ahi, slo), shi))
                segs = nxt
            segs = [(a, b) for a, b in segs if (b - a) >= 30.0]
            if not segs:
                continue  # entirely inside an avoid window: drop it
            lo, hi = max(segs, key=lambda s: s[1] - s[0])
        out.append({"name": f"nbdemo{n}.T{i:03d}",
                    "lambda_min": float(lo), "lambda_max": float(hi)})
    return out


def write_filter_dats(filters, outdir):
    """Write CIGALE .dat files (photon type, tophat) -- same format as
    scripts/make_cigale_filters.py."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for f in filters:
        lo, hi = f["lambda_min"], f["lambda_max"]
        wl_in = np.arange(lo, hi + STEP_AA, STEP_AA)
        wl = np.concatenate(([lo - PAD_AA], wl_in, [hi + PAD_AA]))
        tr = np.concatenate(([0.0], np.ones_like(wl_in), [0.0]))
        path = os.path.join(outdir, f["name"].replace(".", "_") + ".dat")
        with open(path, "w") as fh:
            fh.write(f"# {f['name']}\n# photon\n")
            fh.write(f"# demo synthetic tophat {lo:.0f}-{hi:.0f}A\n")
            for w, t in zip(wl, tr):
                fh.write(f"{w:.2f} {t:.6f}\n")
        paths.append(path)
    return paths


def register_filters(dat_paths, conda_env="cigale", cwd=PILOT_INI_DIR):
    """pcigale-filters add (idempotent: re-adding overwrites/skips).

    pcigale-filters insists on finding a pcigale.ini in its cwd, so we run
    it from the validated pilot directory.
    """
    cmd = ["conda", "run", "-n", conda_env, "pcigale-filters", "add"] + \
        [str(p) for p in dat_paths]
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"pcigale-filters add failed:\n{proc.stderr[-2000:]}")
    return proc.stdout


def registered_filter_names(conda_env="cigale", cwd=PILOT_INI_DIR):
    proc = subprocess.run(["conda", "run", "-n", conda_env,
                           "pcigale-filters", "list"],
                          cwd=cwd, capture_output=True, text=True)
    return set(proc.stdout.split())


# ---------------------------------------------------------------------------
# input table: same recipe as scripts/build_cigale_input.py::build_filters_row
# ---------------------------------------------------------------------------

def _load_specnorm():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "specnorm", os.path.join(PIPE_DIR, "photextra_pipeline",
                                 "spec_normalization.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_input_table(targets, filters, work_dir):
    """cigale_input.txt for CIGALE's filters method: broadband (mJy, 10%
    error floor) + the N synthetic tophat fluxes integrated from the cached
    DESI spectrum and rescaled fiber->total with the pipeline's
    spec-normalization (anchored on Legacy g/r/z), exactly as
    scripts/build_cigale_input.py does for the 10 CHANCES filters.

    Emission-line columns are deliberately EXCLUDED here so the comparison
    across N is purely about continuum sampling resolution.

    targets: list of (tid, combined_row_dict).
    """
    from xpectrafit.filters.custom import CustomFilter
    from xpectrafit.filters.integrate import integrate_filters
    from photextra_pipeline.cigale_integration import (
        BROADBAND_MAP, ERR_FLOOR, broadband_cells)
    specnorm = _load_specnorm()

    cfilters = [CustomFilter(name=f["name"], lambda_min=f["lambda_min"],
                             lambda_max=f["lambda_max"]) for f in filters]
    anchor_surveys = ["Legacy_g", "Legacy_r", "Legacy_z"]
    lambda_eff_um = {"Legacy_g": 0.472, "Legacy_r": 0.641, "Legacy_z": 0.906}
    flux_unit = 1e-17

    rows = []
    for tid, combined in targets:
        z = float(combined["z"])
        npz = np.load(os.path.join(RUN_DIR, tid, "spectroscopy",
                                   f"desi_{tid}.npz"))
        wave = np.asarray(npz["wave"], float)
        flux = np.asarray(npz["flux"], float) * flux_unit
        ivar = np.asarray(npz["ivar"], float)
        with np.errstate(divide="ignore"):
            err = np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan) * flux_unit

        ff = integrate_filters(wave, flux, cfilters, err)
        anchors = specnorm.collect_anchors(
            anchor_surveys, lambda_eff_um, combined, combined,
            float(np.nanmin(wave)), float(np.nanmax(wave)), "phot_")
        spec_norm, _meta = specnorm.fit_spec_normalization(anchors)

        row = {"id": tid, "redshift": f"{z:.6f}"}
        for name, fx in ff.items():
            f_mjy, e_mjy = fx.flux_fnu_mJy, fx.flux_fnu_mJy_err
            if not (np.isfinite(f_mjy) and f_mjy > 0):
                row[name] = "nan"
                row[name + "_err"] = "nan"
                continue
            s = (float(spec_norm.scale_at(fx.lambda_pivot))
                 if spec_norm is not None else 1.0)
            f_tot = f_mjy * s
            e_tot = e_mjy * s if np.isfinite(e_mjy) else 0.0
            e_floor = np.sqrt(e_tot ** 2 + (ERR_FLOOR * f_tot) ** 2)
            row[name] = f"{f_tot:.6e}"
            row[name + "_err"] = f"{e_floor:.6e}"
        row.update(broadband_cells(combined))
        rows.append(row)

    bb_cols = list(BROADBAND_MAP.values())
    band_cols = [f["name"] for f in filters]
    cols = ["id", "redshift"]
    for c in bb_cols + band_cols:
        cols += [c, c + "_err"]

    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, "cigale_input.txt")
    with open(path, "w") as fh:
        fh.write("# " + " ".join(cols) + "\n")
        for row in rows:
            fh.write(" ".join(str(row.get(c, "nan")) for c in cols) + "\n")
    return path, bb_cols, band_cols


def write_ini(work_dir, bb_cols, band_cols, cores=12):
    """pcigale.ini for the run: the VALIDATED pilot ini (same SED-module
    grid, 17280 models) with only the bands lists swapped for the N
    synthetic filters. Never regenerated with 'pcigale genconf' (that would
    reset the grid)."""
    src = os.path.join(PILOT_INI_DIR, "pcigale.ini")
    with open(src) as fh:
        lines = fh.readlines()

    phot = bb_cols + band_cols
    bands_fit = ", ".join(sum([[c, c + "_err"] for c in phot], []))
    bands_analysis = ", ".join(phot)

    out = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("bands =") and not ln.startswith("  "):
            out.append(f"bands = {bands_fit}\n")
        elif stripped.startswith("bands =") and ln.startswith("  "):
            out.append(f"  bands = {bands_analysis}\n")
        elif stripped.startswith("cores ="):
            out.append(f"cores = {cores}\n")
        else:
            out.append(ln)
    with open(os.path.join(work_dir, "pcigale.ini"), "w") as fh:
        fh.writelines(out)
    shutil.copy(os.path.join(PILOT_INI_DIR, "pcigale.ini.spec"),
                os.path.join(work_dir, "pcigale.ini.spec"))


def run_pcigale(work_dir, conda_env="cigale", timeout_s=3600):
    out_dir = os.path.join(work_dir, "out")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    proc = subprocess.run(["conda", "run", "-n", conda_env,
                           "--no-capture-output", "pcigale", "run"],
                          cwd=work_dir, capture_output=True, text=True,
                          timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"pcigale run failed:\n{proc.stdout[-1500:]}\n"
                           f"{proc.stderr[-1500:]}")
    return os.path.join(out_dir, "results.txt")


def parse_results(results_path):
    """results.txt -> {target_id: {chi2_red, mstar, mstar_err, sfr, sfr_err}}"""
    with open(results_path) as fh:
        header = fh.readline().split()
        if header[0] == "#":
            header = header[1:]
        rows = [dict(zip(header, ln.split())) for ln in fh if ln.strip()]
    out = {}
    for r in rows:
        out[r["id"]] = {
            "chi2_red": float(r["best.reduced_chi_square"]),
            "mstar": float(r["bayes.stellar.m_star"]),
            "mstar_err": float(r["bayes.stellar.m_star_err"]),
            "sfr": float(r["bayes.sfh.sfr"]),
            "sfr_err": float(r["bayes.sfh.sfr_err"]),
        }
    return out


def run_n_filter_experiment(n, targets, work_root, conda_env="cigale",
                            cores=12):
    """Full pipeline for one N: generate + register N tophats, build input,
    write ini, run pcigale, parse. Returns (filters, results_dict)."""
    work_dir = os.path.join(work_root, f"cigale_nb{n}")
    filters = make_synthetic_filters(n)
    dats = write_filter_dats(filters, os.path.join(work_dir, "filters"))
    have = registered_filter_names(conda_env=conda_env)
    missing = [p for p, f in zip(dats, filters) if f["name"] not in have]
    if missing:
        register_filters(missing, conda_env=conda_env)
    _, bb_cols, band_cols = build_input_table(targets, filters, work_dir)
    write_ini(work_dir, bb_cols, band_cols, cores=cores)
    results_path = run_pcigale(work_dir, conda_env=conda_env)
    return filters, parse_results(results_path)
