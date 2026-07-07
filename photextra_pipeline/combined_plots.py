"""Diagnostic plot for the combined photometry+spectroscopy product (mode="both").

One figure per target showing, on a single log-log SED panel:

- the real imaging photometry (``phot_*_flux_mjy`` from the combined dict),
- the spectrum's synthetic broadband fluxes BEFORE normalization
  (``synth_*_flux_mjy``, open markers),
- the same points AFTER the Zou et al. (2024)-style aperture normalization
  (``scaled_*_flux_mjy``, filled markers), plus an annotation of the
  ``norm_factor``/``norm_band`` actually used,
- and, when the raw spectrum file is accessible (DESI ``.npz`` cache or an
  SDSS FITS), the observed spectrum itself converted to mJy — both raw and
  scaled — as thin lines, so the whole combination can be sanity-checked
  visually.

Style matches validation_plots.py (dark background, mJy vs micron).
"""

import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .downloader import SURVEY_DEFAULTS

logger = logging.getLogger(__name__)

# effective wavelengths (micron) for synthetic bands not covered by
# SURVEY_DEFAULTS (the spectral fit can also produce SDSS synthetic fluxes)
_EXTRA_LAMBDA_EFF = {
    "SDSS_u": 0.3551, "SDSS_g": 0.4686, "SDSS_r": 0.6166,
    "SDSS_i": 0.7480, "SDSS_z": 0.8932,
}

_C_A_S = 2.99792458e18  # speed of light in Angstrom/s

# DESI/SDSS spectra are stored in units of 1e-17 erg s^-1 cm^-2 A^-1
_FLAM_UNIT = 1e-17


def _band_lambda_um(band):
    meta = SURVEY_DEFAULTS.get(band)
    if meta is not None:
        return meta.get("lambda_eff")
    return _EXTRA_LAMBDA_EFF.get(band)


def _flam_to_mjy(wave_A, flux_flam):
    """f_lambda [1e-17 erg/s/cm2/A] at wave [A] -> f_nu [mJy]."""
    fnu_cgs = flux_flam * _FLAM_UNIT * wave_A ** 2 / _C_A_S  # erg/s/cm2/Hz
    return fnu_cgs * 1e26  # 1 mJy = 1e-26 erg/s/cm2/Hz


def _load_spectrum_mjy(spectrum_path):
    """Best-effort load of the observed spectrum -> (wave_um, flux_mJy).

    Supports the pipeline's DESI ``.npz`` cache directly and SDSS-style FITS
    via the XpectraFit loader. Returns ``None`` on any failure (the plot then
    simply omits the spectrum trace).
    """
    if not spectrum_path:
        return None
    try:
        if spectrum_path.endswith(".npz"):
            data = np.load(spectrum_path)
            wave, flux = np.asarray(data["wave"], float), np.asarray(data["flux"], float)
        else:
            from xpectrafit.io.loaders import load_sdss
            spec = load_sdss(spectrum_path)
            wave, flux = np.asarray(spec.wave, float), np.asarray(spec.flux, float)
        good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
        if good.sum() < 10:
            return None
        wave, flux = wave[good], flux[good]
        return wave / 1e4, _flam_to_mjy(wave, flux)
    except Exception as exc:
        logger.warning("could not load spectrum for combined plot (%s): %s",
                       spectrum_path, exc)
        return None


def _smooth(y, width=15):
    """Running median for display only (keeps the log-y trace readable)."""
    if y.size < width * 2:
        return y
    from scipy.ndimage import median_filter
    return median_filter(y, size=width, mode="nearest")


def _collect_points(combined, prefix):
    """Extract (band, lambda_um, flux, err) rows for a given key prefix."""
    rows = []
    for key, val in combined.items():
        if not (key.startswith(prefix) and key.endswith("_flux_mjy")):
            continue
        if key.endswith("_flux_err_mjy"):
            continue
        band = key[len(prefix):-len("_flux_mjy")]
        lam = _band_lambda_um(band)
        if lam is None:
            continue
        try:
            f = float(val)
        except (TypeError, ValueError):
            continue
        e = combined.get(f"{prefix}{band}_flux_err_mjy", np.nan)
        try:
            e = float(e)
        except (TypeError, ValueError):
            e = np.nan
        if np.isfinite(f) and f > 0:
            rows.append((band, lam, f, e))
    rows.sort(key=lambda r: r[1])
    return rows


def plot_combined(target_id, combined_result, out_path, spectrum_path=None):
    """Photometry-vs-spectroscopy normalization diagnostic (mode="both")."""
    plt.style.use("dark_background")
    try:
        _plot_combined_impl(target_id, combined_result, out_path, spectrum_path)
    finally:
        plt.style.use("default")


def _plot_combined_impl(target_id, combined, out_path, spectrum_path):
    # NOTE: "phot_fiber_*" keys also match the "phot_" prefix, but their band
    # then reads "fiber_<band>", has no lambda_eff, and is dropped — so
    # collecting the two prefixes separately is safe.
    phot = _collect_points(combined, "phot_")
    fiber = _collect_points(combined, "phot_fiber_")
    synth = _collect_points(combined, "synth_")
    scaled = _collect_points(combined, "scaled_")

    norm = combined.get("norm_factor", np.nan)
    try:
        norm = float(norm)
    except (TypeError, ValueError):
        norm = np.nan
    norm_band = combined.get("norm_band", "") or "none"

    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    # --- observed spectrum (thin lines: raw + scaled) ----------------------
    spec = _load_spectrum_mjy(spectrum_path)
    if spec is not None:
        wum, fmjy = spec
        fsm = _smooth(fmjy)
        pos = fsm > 0
        ax.plot(wum[pos], fsm[pos], color="#9a9a9a", lw=0.7, alpha=0.75,
                zorder=2, label="observed spectrum (fibre)")
        if np.isfinite(norm):
            ax.plot(wum[pos], fsm[pos] * norm, color="#5db4ff", lw=0.7,
                    alpha=0.85, zorder=2,
                    label=f"spectrum x norm ({norm:.2f})")

    # --- imaging photometry (absolute reference) ----------------------------
    if phot:
        lam = np.array([r[1] for r in phot])
        f = np.array([r[2] for r in phot])
        e = np.array([r[3] for r in phot])
        ax.errorbar(lam, f, yerr=np.where(np.isfinite(e), e, 0.0), fmt="o",
                    color="#ffd24d", ms=8, mec="black", mew=0.5, capsize=4,
                    lw=0, elinewidth=1.4, ecolor="#ffd24d", zorder=5,
                    label="imaging photometry (total)")

    # --- fiber-matched imaging photometry (normalization reference) ---------
    fiber_diam = combined.get("fiber_diam_arcsec", np.nan)
    try:
        fiber_diam = float(fiber_diam)
    except (TypeError, ValueError):
        fiber_diam = np.nan
    if fiber:
        lam = np.array([r[1] for r in fiber])
        f = np.array([r[2] for r in fiber])
        e = np.array([r[3] for r in fiber])
        lbl = ("imaging photometry (fiber "
               + (f"{fiber_diam:.1f}\"" if np.isfinite(fiber_diam) else "")
               + ")")
        ax.errorbar(lam, f, yerr=np.where(np.isfinite(e), e, 0.0), fmt="^",
                    color="#7dff9b", ms=9, mec="black", mew=0.6, capsize=4,
                    lw=0, elinewidth=1.2, ecolor="#7dff9b", zorder=6,
                    label=lbl)

    # --- synthetic photometry from the spectrum -----------------------------
    if synth:
        lam = np.array([r[1] for r in synth])
        f = np.array([r[2] for r in synth])
        ax.scatter(lam, f, facecolors="none", edgecolors="#ff5d5d", s=95,
                   marker="s", linewidths=1.8, zorder=6,
                   label="synthetic phot (pre-norm)")

    if scaled:
        lam = np.array([r[1] for r in scaled])
        f = np.array([r[2] for r in scaled])
        ax.scatter(lam, f, color="#5db4ff", s=70, marker="D",
                   edgecolors="black", linewidths=0.6, zorder=7,
                   label="synthetic phot x norm")

    # arrow showing the normalization jump at the reference band, pointing
    # at whichever aperture flux was actually used for the normalization
    norm_aperture = combined.get("norm_aperture", "") or ""
    # new labels: "total" (primary) / "fiber_fallback"; old CSVs may still
    # carry "fiber" / "total_fallback"
    ref_points = fiber if norm_aperture.startswith("fiber") else phot
    ref_lam = _band_lambda_um(norm_band)
    if ref_lam is not None and np.isfinite(norm):
        fs = next((r[2] for r in synth if r[0] == norm_band), None)
        fp = next((r[2] for r in ref_points if r[0] == norm_band), None)
        if fs is not None and fp is not None and fs > 0 and fp > 0:
            ax.annotate("", xy=(ref_lam, fp), xytext=(ref_lam, fs),
                        arrowprops=dict(arrowstyle="-|>", color="white",
                                        lw=1.4, alpha=0.85), zorder=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=12)
    ax.set_ylabel("flux density [mJy]", fontsize=12)
    ax.set_title(f"{target_id} — photometry + spectroscopy combination",
                 fontsize=13)
    ax.grid(True, which="both", alpha=0.18)

    # sensible y-limits from the photometric points (the raw spectrum can be
    # noisy at the ends and would otherwise stretch the axis)
    fluxes = [r[2] for r in phot + fiber + synth + scaled]
    if fluxes:
        ax.set_ylim(min(fluxes) * 0.2, max(fluxes) * 8.0)

    # annotation box: the normalization actually applied + fit context
    z = combined.get("z")
    lines = [
        f"norm band   : {norm_band}",
        f"norm factor : {norm:.3f}" if np.isfinite(norm) else "norm factor : n/a",
        f"norm apert. : {norm_aperture or 'n/a'}",
        (f"fiber diam  : {fiber_diam:.1f}\" "
         f"({combined.get('spectrum_source') or 'n/a'})")
        if np.isfinite(fiber_diam) else "fiber diam  : n/a",
        f"z           : {float(z):.4f}" if z is not None else "z           : n/a",
        f"spec status : {combined.get('spec_status', 'n/a')}",
        f"synth source: {combined.get('synth_photometry_source', 'n/a')}",
    ]
    ax.text(0.985, 0.03, "\n".join(lines), transform=ax.transAxes,
            va="bottom", ha="right", multialignment="left",
            family="monospace", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#111111",
                      edgecolor="#555555", alpha=0.85))

    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("combined plot -> %s", out_path)
