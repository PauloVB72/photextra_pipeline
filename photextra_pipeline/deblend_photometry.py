"""Two independent deblending methods for merger photometry in blended bands.

When a band's PSF FWHM >= the nuclear separation (``blend_flag=True``), the
per-component fluxes measured by mask are unreliable because the PSF mixes light
from both components. These methods re-estimate the individual fluxes using
Legacy_r (PSF 1.3", 0.262"/px) as the reliable non-blended reference:

- ``deblend_ratio_prior``  : split the (reliable) total flux by the Legacy_r ratio.
- ``deblend_tphot``        : simplified T-PHOT — Legacy_r morphological templates
                             convolved + reprojected to each band, amplitudes by
                             non-negative least squares.
- ``deblend_combined``     : tphot when well-behaved, else ratio_prior fallback.

All methods return ``{}`` on any failure and only emit entries for bands where
``blend_flag=True``.
"""

import logging

import numpy as np

from .photometry import estimate_background, counts_to_mjy
from .convolution import matching_kernel, TARGET_SURVEY
from .reprojection import reproject_masks as _reproject_masks
from .downloader import SURVEY_DEFAULTS

logger = logging.getLogger(__name__)


def _component_names(d):
    """Component keys (gal1, gal2, ...) excluding bookkeeping/aggregate keys."""
    skip = {"total", "circular", "blend_flag", "psf_fwhm_arcsec", "n_pix"}
    return [k for k in d.keys() if k not in skip]


# ---------------------------------------------------------------------------
# 1. ratio prior
# ---------------------------------------------------------------------------
def deblend_ratio_prior(photometry, ref_survey="Legacy_r"):
    """Split the total flux in blended bands by the Legacy_r component ratio."""
    out = {}
    ref = photometry.get(ref_survey)
    if not isinstance(ref, dict):
        return out

    ref_total = ref.get("total", {})
    f_tot_ref = ref_total.get("flux_mjy", np.nan)
    s_tot_ref = ref_total.get("flux_err_mjy", np.nan)
    s_tot_ref = s_tot_ref if np.isfinite(s_tot_ref) else 0.0
    if not (np.isfinite(f_tot_ref) and f_tot_ref > 0):
        return out

    # reference component fluxes/errors
    ref_comp = {}
    for c in _component_names(ref):
        m = ref.get(c, {})
        f = m.get("flux_mjy", np.nan)
        s = m.get("flux_err_mjy", np.nan)
        if np.isfinite(f):
            ref_comp[c] = (f, s if np.isfinite(s) else 0.0)
    if not ref_comp:
        return out

    for survey, phot in photometry.items():
        if not isinstance(phot, dict) or not phot.get("blend_flag", False):
            continue
        tot = phot.get("total", {})
        f_tot = tot.get("flux_mjy", np.nan)
        s_tot = tot.get("flux_err_mjy", np.nan)
        if not np.isfinite(f_tot):
            continue
        s_tot = s_tot if np.isfinite(s_tot) else 0.0

        entry = {}
        for c, (f_ref, s_ref) in ref_comp.items():
            ratio = f_ref / f_tot_ref
            f_c = f_tot * ratio
            # σ² = (ratio·σ_tot)² + (f_tot/f_tot_ref·σ_ref)²
            #      + (f_tot·f_ref/f_tot_ref²·σ_tot_ref)²
            var = (ratio * s_tot) ** 2
            var += (f_tot / f_tot_ref * s_ref) ** 2
            var += (f_tot * f_ref / f_tot_ref ** 2 * s_tot_ref) ** 2
            entry[c] = {"flux_mjy": f_c, "flux_err_mjy": float(np.sqrt(var)),
                        "method": "ratio_prior"}
        entry["total"] = {"flux_mjy": f_tot, "flux_err_mjy": s_tot,
                          "method": "ratio_prior"}
        out[survey] = entry

    return out


# ---------------------------------------------------------------------------
# 2. simplified T-PHOT
# ---------------------------------------------------------------------------
def _ref_native_pixscale(ref_wcs, ref_survey):
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        ps = float(abs(proj_plane_pixel_scales(ref_wcs)).mean() * 3600.0)
        if ps > 0:
            return ps
    except Exception:
        pass
    return SURVEY_DEFAULTS.get(ref_survey, {}).get("pixscale", 0.262)


def _build_templates(ref_data, ref_wcs, ref_survey, masks_native, mask_wcs):
    """Normalized morphological templates per component in the Legacy_r grid.

    Each template sums to 1. Returns dict comp -> template, or None on failure.
    """
    masks_ref = _reproject_masks(masks_native, mask_wcs, ref_wcs, ref_data.shape)

    total_mask = None
    for m in masks_ref.values():
        if m is None:
            continue
        total_mask = m.copy() if total_mask is None else (total_mask | m)
    if total_mask is None:
        return None

    bkg, _ = estimate_background(ref_data, exclude_mask=total_mask)

    templates = {}
    for name, m in masks_ref.items():
        if m is None or not np.any(m):
            continue
        img = np.zeros_like(ref_data, dtype=float)
        img[m] = ref_data[m] - bkg
        s = float(np.sum(img))
        if not np.isfinite(s) or s <= 0:  # negative/empty support -> skip
            continue
        templates[name] = img / s
    return templates or None


def _gaussian_psf(shape, cx, cy, sigma_px):
    """Normalised 2D Gaussian PSF template centred at (cx, cy) in pixel coords."""
    yy, xx = np.indices(shape, dtype=float)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    g = np.exp(-r2 / (2.0 * sigma_px ** 2))
    s = g.sum()
    return g / s if s > 0 else g


def _neighbor_psf_templates(neighbor_sources, merger_centroids_sky,
                             band_wcs, data_shape, psf_fwhm_arcsec,
                             exclude_radius_arcsec=4.0):
    """Build unit-sum PSF templates for unWISE neighbors (excluding merger components).

    neighbor_sources : list of dicts with keys 'ra', 'dec' (sky coords)
    merger_centroids_sky : list of (ra, dec) tuples for the merger components
    Returns list of 2D arrays, one per included neighbor.
    """
    if not neighbor_sources or band_wcs is None:
        return []

    # pixel scale of the band image (arcsec/px)
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        ps = float(abs(proj_plane_pixel_scales(band_wcs)).mean() * 3600.0)
    except Exception:
        ps = 1.375  # fallback WISE default

    sigma_px = (psf_fwhm_arcsec / 2.355) / ps
    templates = []

    for src in neighbor_sources:
        sra, sdec = float(src["ra"]), float(src["dec"])

        # skip sources that match a merger component
        too_close = False
        for (mra, mdec) in merger_centroids_sky:
            if mra is None:
                continue
            from .unwise_forced import _angular_sep_deg
            sep = _angular_sep_deg(mra, mdec, sra, sdec) * 3600.0
            if sep < exclude_radius_arcsec:
                too_close = True
                break
        if too_close:
            continue

        try:
            cx, cy = band_wcs.world_to_pixel_values(sra, sdec)
            cx, cy = float(cx), float(cy)
        except Exception:
            continue

        # skip if outside image
        if not (0 <= cx < data_shape[1] and 0 <= cy < data_shape[0]):
            continue

        templates.append(_gaussian_psf(data_shape, cx, cy, sigma_px))

    return templates


def deblend_tphot(images, masks_native, mask_wcs, surveys, ref_survey="Legacy_r",
                   photometry=None, ra_center=None, dec_center=None,
                   fit_neighbors=True):
    """Per-component flux in blended bands via Legacy_r morphological templates.

    Only runs on bands where blend_flag=True (requires photometry dict to check).
    For each band: convolve every Legacy_r template up to the band PSF, reproject
    it to the band's native grid, then fit amplitudes by non-negative LSQ.

    When fit_neighbors=True and ra_center/dec_center are provided, neighbor
    sources from the unWISE catalog are included in the simultaneous fit so
    their flux does not contaminate the merger component amplitudes.
    """
    from reproject import reproject_interp
    from astropy.convolution import convolve_fft

    out = {}

    ref_entry = images.get(ref_survey, {})
    ref_data = ref_entry.get("data")
    ref_wcs = ref_entry.get("wcs")
    if ref_data is None or ref_wcs is None or mask_wcs is None:
        return out

    try:
        templates = _build_templates(ref_data, ref_wcs, ref_survey,
                                     masks_native, mask_wcs)
    except Exception as exc:
        logger.warning("tphot: template construction failed: %s", exc)
        return out
    if not templates:
        return out

    comp_names = list(templates.keys())
    ref_ps = _ref_native_pixscale(ref_wcs, ref_survey)
    ref_psf = SURVEY_DEFAULTS[ref_survey]["psf_fwhm"]

    # sky centroids of the merger components (for neighbor exclusion)
    merger_centroids = []
    if fit_neighbors and ra_center is not None:
        from .unwise_forced import _mask_centroid_sky, _query_unwise_catalog, _angular_sep_deg
        for name, mask in masks_native.items():
            ra_c, dec_c = _mask_centroid_sky(mask, mask_wcs)
            merger_centroids.append((ra_c, dec_c))

        # Only query within 30" — beyond that, neighbors rarely overlap the
        # merger PSF footprint significantly, and including too many templates
        # on a small image makes NNLS degenerate.
        raw_neighbors = _query_unwise_catalog(ra_center, dec_center, radius_arcsec=45)

        # Keep only neighbors with FW1 > 30 Vega nanomaggies (~0.009 mJy) —
        # fainter sources contribute < 1% of a typical merger flux and just
        # add noise to the fit.
        neighbor_sources = [
            s for s in raw_neighbors
            if float(s.get("FW1", 0) or 0) > 30
        ]
        logger.info("tphot neighbors: %d/%d unWISE sources (bright, <45\")",
                    len(neighbor_sources), len(raw_neighbors))
    else:
        neighbor_sources = []

    for survey in surveys:
        # skip non-blended bands — mask photometry is already reliable there
        if photometry is not None:
            if not photometry.get(survey, {}).get("blend_flag", False):
                continue

        meta = SURVEY_DEFAULTS.get(survey, {})
        source_fwhm = float(meta.get("psf_fwhm", 0.0))

        entry = images.get(survey, {})
        data = entry.get("data")
        band_wcs = entry.get("wcs")
        if data is None or band_wcs is None:
            continue

        try:
            kernel, _ = matching_kernel(ref_psf, source_fwhm, ref_ps)

            band_T = {}
            for name in comp_names:
                T = templates[name]
                if kernel is not None:
                    T = convolve_fft(T, kernel, normalize_kernel=True,
                                     allow_huge=True, nan_treatment="fill",
                                     fill_value=0.0)
                Tb, _ = reproject_interp((T, ref_wcs), band_wcs,
                                         shape_out=data.shape, order="bilinear")
                Tb = np.nan_to_num(Tb, nan=0.0)
                s = float(np.sum(Tb))
                band_T[name] = Tb / s if s > 0 else Tb

            # neighbor PSF templates — only for WISE bands where unWISE
            # positions are exact; for GALEX the PSF/depth mismatch makes
            # neighbor fitting unreliable.
            nbr_templates = []
            if neighbor_sources and survey.startswith("WISE"):
                nbr_templates = _neighbor_psf_templates(
                    neighbor_sources, merger_centroids,
                    band_wcs, data.shape, source_fwhm,
                )
                if nbr_templates:
                    logger.info("tphot %s: fitting %d merger + %d neighbor templates",
                                survey, len(comp_names), len(nbr_templates))

            # amplitudes are fitted in counts on this band's own grid;
            # correct to the ZP-native pixel grid (see photometry module)
            from .photometry import zp_grid_correction
            zp_corr = zp_grid_correction(band_wcs, survey)
            res = _fit_amplitudes(band_T, comp_names, data, survey,
                                  neighbor_templates=nbr_templates,
                                  zp_corr=zp_corr)
            if res is not None:
                out[survey] = res
        except Exception as exc:
            logger.warning("tphot failed for %s: %s", survey, exc)
            continue

    return out


def _fit_amplitudes(band_T, comp_names, data, survey, neighbor_templates=None,
                    zp_corr=1.0):
    """Non-negative LSQ of merger + neighbor templates against background-subtracted image.

    Merger templates (band_T) go first in the design matrix; their amplitudes
    are the science result. Neighbor PSF templates are appended so that their
    flux is absorbed rather than contaminating the merger components. All
    amplitudes are constrained ≥ 0 via scipy.optimize.nnls.

    Returns dict with per-merger-component fluxes plus private _resid_rms /
    _bkg_rms keys. Returns None if any merger amplitude is invalid.
    """
    from scipy.optimize import nnls

    neighbor_templates = neighbor_templates or []

    # fit region: union of merger template footprints + dilation ring
    support = None
    for T in band_T.values():
        peak = T.max() if T.max() > 0 else 1.0
        s = T > (1e-6 * peak)
        support = s if support is None else (support | s)
    # also include footprints of bright neighbors in the support
    for T in neighbor_templates:
        peak = T.max() if T.max() > 0 else 1.0
        s = T > (1e-4 * peak)   # tighter threshold for neighbors
        if support is None:
            support = s
        else:
            support = support | s

    if support is None or not np.any(support):
        return None

    try:
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(support, iterations=3)
    except Exception:
        ring = support

    bkg, rms = estimate_background(data, exclude_mask=support)
    fit_mask = ring & np.isfinite(data)
    if not np.any(fit_mask):
        return None

    # design matrix: merger templates first, then neighbors
    all_templates = [band_T[n] for n in comp_names] + neighbor_templates
    A = np.column_stack([T[fit_mask] for T in all_templates])
    b = data[fit_mask] - bkg

    try:
        coeffs, resid_norm = nnls(A, b)
    except Exception:
        return None

    merger_coeffs = coeffs[:len(comp_names)]
    if np.any(~np.isfinite(merger_coeffs)):
        return None

    resid = b - A @ coeffs
    resid_rms = float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.inf

    n_neighbors = len(neighbor_templates)
    method_tag = "tphot_nbr" if n_neighbors > 0 else "tphot"

    entry = {"_resid_rms": resid_rms, "_bkg_rms": float(rms),
             "_n_neighbors": n_neighbors}
    total_flux = 0.0
    total_var = 0.0
    for name, a in zip(comp_names, merger_coeffs):
        flux = counts_to_mjy(float(a), survey) * zp_corr
        peak = band_T[name].max()
        n_pix = int(np.sum(band_T[name] > (1e-6 * peak))) if peak > 0 else 1
        flux_err = abs(counts_to_mjy(resid_rms * np.sqrt(max(n_pix, 1)),
                                     survey)) * zp_corr
        entry[name] = {"flux_mjy": flux, "flux_err_mjy": flux_err,
                       "method": method_tag}
        total_flux += flux
        total_var += flux_err ** 2
    entry["total"] = {"flux_mjy": total_flux,
                      "flux_err_mjy": float(np.sqrt(total_var)),
                      "method": method_tag}
    return entry


# ---------------------------------------------------------------------------
# 3. combined
# ---------------------------------------------------------------------------
def deblend_combined(photometry, images, masks_native, mask_wcs, surveys,
                     ref_survey="Legacy_r"):
    """tphot where well-behaved (positive amplitudes, low residual), else ratio."""
    try:
        tphot = deblend_tphot(images, masks_native, mask_wcs, surveys, ref_survey,
                               photometry=photometry)
    except Exception as exc:
        logger.warning("deblend_combined: tphot stage failed: %s", exc)
        tphot = {}
    try:
        ratio = deblend_ratio_prior(photometry, ref_survey)
    except Exception as exc:
        logger.warning("deblend_combined: ratio stage failed: %s", exc)
        ratio = {}

    # only process genuinely blended bands
    blended = set()
    for survey, phot in photometry.items():
        if isinstance(phot, dict) and phot.get("blend_flag", False):
            blended.add(survey)

    out = {}
    for survey in blended:
        t = tphot.get(survey)
        use_tphot = False
        if isinstance(t, dict):
            resid_rms = t.get("_resid_rms", np.inf)
            bkg_rms = t.get("_bkg_rms", 0.0)
            # tphot enforces non-negative amplitudes already (else None);
            # accept when the residual is not pathological.
            if np.isfinite(resid_rms) and bkg_rms > 0 and resid_rms < 3.0 * bkg_rms:
                use_tphot = True

            # Physical sanity: if component sum diverges from the catalog total
            # by more than 40%, tphot is fitting PSF wings and overcounting;
            # fall back to ratio_prior which is conservative but consistent.
            if use_tphot:
                comp_keys = [k for k in t if not k.startswith("_") and k != "total"]
                comp_sum = sum(t[c].get("flux_mjy", np.nan) for c in comp_keys
                               if isinstance(t.get(c), dict))
                phot_total = photometry.get(survey, {}).get("total", {}).get("flux_mjy", np.nan)
                if np.isfinite(phot_total) and phot_total > 0 and np.isfinite(comp_sum):
                    ratio_check = comp_sum / phot_total
                    if ratio_check > 1.4 or ratio_check < 0.6:
                        logger.warning("tphot %s: component_sum/total=%.2f outside [0.6,1.4];"
                                       " falling back to ratio_prior", survey, ratio_check)
                        use_tphot = False

        if use_tphot:
            out[survey] = {k: v for k, v in t.items() if not k.startswith("_")}
        elif survey in ratio:
            out[survey] = ratio[survey]

    return out
