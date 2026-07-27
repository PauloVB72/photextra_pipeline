"""Optional spectral-fitting stage using XpectraFit.

If a target dict carries ``spectrum_path`` (SDSS/DESI-style FITS) and ``z``,
the spectrum is fit with :class:`xpectrafit.XpectraFitter` and a flat dict of
the key results is returned. Targets without spectroscopy (the common case)
are untouched: :func:`run_spectral_fit` returns ``None`` immediately.

NOTE: the fibre spectrum covers the TOTAL light of the merging system, not an
individual deblended component, so the spectral result is associated with the
"total" aggregate — never with a single entry of ``masks_native``.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# Every narrow emission line XpectraFit measures is copied into the flat
# result dict (22 lines as of xpectrafit.core.lines.EMISSION_LINES: MgII2800,
# [NeV]3426, [OII]3727/3729, [NeIII]3869/3968, H10/H9/H8/Hepsilon, Hdelta,
# Hgamma, Hbeta, [OIII]4959/5007, HeI5876, [OI]6300, [NII]6548, Halpha,
# [NII]6584, [SII]6716/6731), each with flux, flux_err, EW, EW_err,
# sigma (km/s), velocity offset and S/N — see _flatten_emission_lines.
# The historical 4-line subset below is kept ONLY as a fallback when the
# fitted result carries no emission_lines dict at all.
_LINES_OF_INTEREST = ["Halpha", "Hbeta", "[OIII]5007", "[NII]6584"]


def _line_key(name):
    """Column-safe line name: strip brackets ([OIII]5007 -> OIII5007)."""
    return name.replace("[", "").replace("]", "")


def _flatten_emission_lines(result):
    """Flatten ALL per-line quantities XpectraFit measured into flat columns.

    XpectraFit's ``LineMeasurement`` (narrow lines) carries: flux, flux_err,
    ew, ew_err, sigma_kms, v_kms, snr. Its ``BroadLineMeasurement`` (broad
    AGN components) carries: flux, flux_err, fwhm_kms, fwhm_err, snr.
    Everything available is exported; note that a per-line sigma ERROR is
    not computed by XpectraFit (only the value), so no sigma_err column
    exists per line — the global stellar ``sigma``/``sigma_err`` columns
    are separate (stellar velocity dispersion from pPXF).

    Column scheme (MANGA-DAP style, one block per line):
      flux_<Name>, flux_err_<Name>            (kept for backward compat)
      line_<Name>_ew, line_<Name>_ew_err      (Angstrom, observed frame)
      line_<Name>_sigma                       (km/s velocity dispersion)
      line_<Name>_v_kms                       (velocity offset, km/s)
      line_<Name>_snr
      broadline_<Name>_flux/_flux_err/_fwhm_kms/_fwhm_err/_snr
    """
    out = {}
    narrow = getattr(result, "emission_lines", None) or {}
    for name, line in narrow.items():
        key = _line_key(name)
        out[f"flux_{key}"] = getattr(line, "flux", np.nan)
        out[f"flux_err_{key}"] = getattr(line, "flux_err", np.nan)
        out[f"line_{key}_ew"] = getattr(line, "ew", np.nan)
        out[f"line_{key}_ew_err"] = getattr(line, "ew_err", np.nan)
        out[f"line_{key}_sigma"] = getattr(line, "sigma_kms", np.nan)
        out[f"line_{key}_v_kms"] = getattr(line, "v_kms", np.nan)
        out[f"line_{key}_snr"] = getattr(line, "snr", np.nan)
    broad = getattr(result, "broad_lines", None) or {}
    for name, line in broad.items():
        key = _line_key(name)
        out[f"broadline_{key}_flux"] = getattr(line, "flux", np.nan)
        out[f"broadline_{key}_flux_err"] = getattr(line, "flux_err", np.nan)
        out[f"broadline_{key}_fwhm_kms"] = getattr(line, "fwhm_kms", np.nan)
        out[f"broadline_{key}_fwhm_err"] = getattr(line, "fwhm_err", np.nan)
        out[f"broadline_{key}_snr"] = getattr(line, "snr", np.nan)
    return out


def _bin_spectrum(wave, flux, ivar, factor):
    """Average adjacent pixels by ``factor`` (boxcar), combining ivar properly.

    DESI spectra are sampled at 0.8 Angstrom on a linear grid, i.e. ~2 pixels
    per resolution element (FWHM ~1.5-1.8 Angstrom). Adjacent pixels are
    therefore strongly correlated, which violates pPXF's independent-noise
    assumption: at native sampling the kinematic fit is over-sensitive to the
    (correlated) fine structure and lands on a spurious broad + velocity-shifted
    LOSVD (velocity dispersion inflated to ~350-500 km/s and a ~400-700 km/s
    velocity offset, even though the absorption lines actually sit at the same
    observed wavelengths as the SDSS spectrum of the same object). Averaging to
    ~Nyquist (1.6 Angstrom / ~1 pixel per FWHM) decorrelates the pixels and
    recovers velocity dispersions that agree with the matching SDSS fit.

    NB: interpolating onto a coarser log grid (e.g. via ppxf ``log_rebin`` at a
    larger velscale) does NOT fix this — only true pixel *averaging* does.
    """
    wave = np.asarray(wave, float)
    flux = np.asarray(flux, float)
    ivar = np.asarray(ivar, float)
    n = (len(wave) // factor) * factor
    if n < factor:
        return wave, flux, ivar
    w = wave[:n].reshape(-1, factor)
    f = flux[:n].reshape(-1, factor)
    iv = ivar[:n].reshape(-1, factor)
    wave_b = w.mean(1)
    flux_b = f.mean(1)
    # Combine as the (variance-)mean, with bad pixels (ivar<=0, e.g. masked sky
    # lines) carrying an enormous variance. A bin that contains one therefore
    # ends up with a near-zero ivar_b: it stays a valid pixel but with
    # negligible weight, rather than either being dropped or (worse) inheriting
    # a moderate weight that would re-inject sky-contaminated flux and re-bias
    # the kinematics. Empirically this variant recovers velocity dispersions
    # that match the SDSS fit of the same object; small changes to this
    # bad-pixel handling do not (the native-resolution fit sits in a biased
    # global chi2 minimum, so the combination rule matters).
    var = 1.0 / np.clip(iv, 1e-30, None)
    ivar_b = 1.0 / np.clip(var.mean(1), 1e-30, None)
    return wave_b, flux_b, ivar_b


def _bin_lsf(wave, lsf_fwhm, factor):
    """Bin a per-pixel FWHM array like :func:`_bin_spectrum` bins the flux.

    Averaging ``factor`` pixels is an extra boxcar convolution; the widening it
    adds on top of the boxcar already contained in the native LSF is
    ``sqrt((factor**2 - 1)/12) * dlam`` in sigma, added in quadrature.
    """
    wave = np.asarray(wave, float)
    lsf = np.asarray(lsf_fwhm, float)
    n = (len(lsf) // factor) * factor
    if n < factor:
        return lsf
    dlam = np.abs(np.median(np.diff(wave))) if len(wave) > 1 else 0.0
    sig = lsf[:n].reshape(-1, factor).mean(1) / 2.3548200
    extra = np.sqrt((factor ** 2 - 1.0) / 12.0) * dlam
    return 2.3548200 * np.sqrt(sig ** 2 + extra ** 2)


def _load_spectrum(path, z, target_id, desi_bin_factor=2):
    """Load a spectrum: DESI ``.npz`` cache (see spectrum_acquisition.py),
    else an SDSS-style FITS (tried first), else a generic FITS.

    DESI spectra (``.npz``) are averaged down by ``desi_bin_factor`` (default 2:
    0.8 -> 1.6 Angstrom) before fitting; see :func:`_bin_spectrum` for why this
    is required for unbiased pPXF kinematics. Set ``desi_bin_factor=1`` to
    disable."""
    if path.endswith(".npz"):
        from xpectrafit.core.spectrum import Spectrum
        data = np.load(path)
        wave, flux, ivar = data["wave"], data["flux"], data["ivar"]
        # Per-pixel instrumental FWHM, only in caches written after the LSF
        # support landed; absent -> scalar fwhm_gal fallback downstream.
        lsf = np.asarray(data["lsf_fwhm"], float) if "lsf_fwhm" in data.files else None
        if desi_bin_factor and desi_bin_factor > 1:
            n_in = len(wave)
            wave, flux, ivar = _bin_spectrum(wave, flux, ivar, desi_bin_factor)
            if lsf is not None and len(lsf) == n_in:
                lsf = _bin_lsf(np.asarray(data["wave"], float), lsf,
                               desi_bin_factor)
        # DESI wavelengths are vacuum; Spectrum converts them to air.
        return Spectrum.from_ivar(wave, flux, ivar, z=z, target_id=target_id,
                                  vacuum=True, lsf_fwhm=lsf)

    from xpectrafit.io.loaders import load_sdss, load_generic_fits
    try:
        return load_sdss(path, z=z, target_id=target_id)
    except Exception as exc:
        logger.info("load_sdss failed (%s); trying generic FITS loader", exc)
        return load_generic_fits(path, z=z, target_id=target_id)


def _extract_synthetic_photometry(result):
    """Flatten XpectraResult synthetic filter fluxes into synth_* columns.

    ``XpectraFitter`` (when given ``survey_filters``) integrates three
    versions of the spectrum through the filter curves:

    - ``filter_fluxes_obs``   : the observed spectrum itself (preferred here,
      since we want to compare *observed* fibre light to imaging photometry);
    - ``filter_fluxes_total`` : best-fit stars+gas+AGN model (fallback);
    - ``filter_fluxes``       : stars-only continuum (last resort).

    Fluxes come back as ``FilterFlux`` dataclasses with ``flux_fnu_mJy`` —
    the same mJy unit the photometric pipeline uses, so they are directly
    comparable. Returns e.g. {'synth_Legacy_r_flux_mjy': ..., ...}.
    """
    out = {}
    for attr, tag in (("filter_fluxes_obs", "obs"),
                      ("filter_fluxes_total", "model"),
                      ("filter_fluxes", "stars")):
        fluxes = getattr(result, attr, None) or {}
        if fluxes:
            out["synth_photometry_source"] = tag
            for name, ff in fluxes.items():
                out[f"synth_{name}_flux_mjy"] = getattr(ff, "flux_fnu_mJy", np.nan)
                out[f"synth_{name}_flux_err_mjy"] = getattr(ff, "flux_fnu_mJy_err", np.nan)
                out[f"synth_{name}_mag_AB"] = getattr(ff, "mag_AB", np.nan)
            break
    return out


def run_spectral_fit(target, survey_filters=None, fit_agn=True,
                     spectrum_fwhm=2.5):
    """Run XpectraFit on the target's spectrum, if it has one.

    Parameters
    ----------
    target : dict
        Pipeline target dict. Spectroscopy is triggered only when both
        ``spectrum_path`` and ``z`` are present.
    survey_filters : list of str, optional
        Survey families (e.g. ``['Legacy', 'SDSS']``) forwarded to
        :class:`xpectrafit.XpectraFitter`; synthetic photometry of the
        spectrum through those bands is added to the output dict as
        ``synth_<band>_flux_mjy`` / ``synth_<band>_mag_AB`` columns.
    fit_agn : bool
        Include an AGN component in the fit (config key
        ``spectroscopy.fit_agn``).
    spectrum_fwhm : float
        Instrumental FWHM (Angstrom) used when the target dict does not
        carry its own ``spectrum_fwhm`` (config key
        ``spectroscopy.spectrum_fwhm``). Only a fallback: when the DESI
        ``.npz`` cache carries a per-pixel ``lsf_fwhm`` array it is used
        instead, so pPXF convolves with the real instrumental resolution.

    Returns
    -------
    dict or None
        Flat dict of spectral-fit results (applies to the TOTAL system light),
        or ``None`` when the target has no spectrum or the fit fails. Never
        raises: the photometric pipeline must not break on spectral failures.
    """
    spectrum_path = target.get("spectrum_path")
    z = target.get("z")
    if spectrum_path is None or z is None:
        return None

    target_id = target.get("id")
    try:
        if not os.path.exists(spectrum_path):
            logger.warning("spectrum file not found for %s: %s",
                           target_id, spectrum_path)
            return None

        from xpectrafit import XpectraFitter

        spec = _load_spectrum(spectrum_path, float(z), target_id)
        result = XpectraFitter(
            spec.wave, spec.flux, spec.flux_err, z=float(z),
            ra=target.get("ra"), dec=target.get("dec"),
            target_id=target_id,
            fwhm_gal=(spec.lsf_fwhm if getattr(spec, "lsf_fwhm", None) is not None
                      else target.get("spectrum_fwhm", spectrum_fwhm)),
            fit_agn=bool(fit_agn),
            survey_filters=survey_filters,
        ).fit()

        if getattr(result, "status", "ok") != "ok":
            logger.warning("spectral fit for %s finished with status=%s",
                           target_id, result.status)

        # Flatten: these numbers describe the total (fibre) light of the
        # merging system, not any single deblended component.
        out = {
            "component": "total",
            "spectrum_path": spectrum_path,
            "z": float(z),
            "status": getattr(result, "status", "ok"),
            "fAGN": result.fAGN,
            "spectral_class": result.spectral_class,
            "spec_class": result.spec_class,
            "bpt_class": result.bpt_class,
            "whan_class": result.whan_class,
            "Dn4000": result.Dn4000,
            "stellar_mass": result.stellar_mass,
            "stellar_mass_aperture": result.stellar_mass_aperture,
            "log_MBH_Halpha": result.log_MBH_Halpha,
            "log_MBH_Hbeta": result.log_MBH_Hbeta,
            "sigma": result.sigma,
            "sigma_err": result.sigma_err,
            "sigma_unresolved": bool(getattr(result, "sigma_unresolved", False)),
            "agn_feii_fwhm": getattr(result, "agn_feii_fwhm", np.nan),
            "chi2_reduced": result.chi2_reduced,
            "agn_dominated": bool(result.agn_dominated),
        }
        # observed-frame spectral coverage: lets _combine_phot_spec restrict
        # normalization anchors to bands whose effective wavelength is
        # actually inside the spectrum (see spec_normalization.collect_anchors)
        try:
            wave = np.asarray(spec.wave, float)
            finite = wave[np.isfinite(wave)]
            if finite.size:
                out["spec_wave_min_aa"] = float(finite.min())
                out["spec_wave_max_aa"] = float(finite.max())
        except Exception:
            pass
        # every line + per-line quantity XpectraFit measured (see
        # _flatten_emission_lines); previously only 4 lines x flux/flux_err
        # were extracted here.
        line_cols = _flatten_emission_lines(result)
        if line_cols:
            out.update(line_cols)
        else:  # fit produced no emission_lines dict: keep the schema stable
            for name in _LINES_OF_INTEREST:
                key = _line_key(name)
                out[f"flux_{key}"] = np.nan
                out[f"flux_err_{key}"] = np.nan

        # synthetic broadband photometry from the spectrum (mJy + AB mags)
        if survey_filters:
            out.update(_extract_synthetic_photometry(result))
        return out

    except Exception as exc:
        logger.error("spectral fit failed for %s: %s", target_id, exc)
        return None
