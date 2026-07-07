"""Aperture photometry on the common grid using xmask component masks.

Flux convention: counts are summed inside the mask after background
subtraction, then converted to flux density in mJy via the survey AB zero
point. Circular aperture photometry is produced as a cross-check.
"""

import logging

import numpy as np
from astropy.convolution import convolve_fft
from astropy.stats import sigma_clipped_stats

from .downloader import SURVEY_DEFAULTS

logger = logging.getLogger(__name__)

# AB: m = -2.5 log10(f / 3631 Jy). counts -> Jy with zp: f_Jy = 3631 * 10^(-0.4 zp) * counts
_AB_ZP_JY = 3631.0


def counts_to_mjy(counts, survey):
    meta = SURVEY_DEFAULTS[survey]
    f_jy = _AB_ZP_JY * 10 ** (-0.4 * meta["zp"]) * counts
    return f_jy * meta["flux_factor"]  # flux_factor: Jy -> mJy (1000)


def zp_grid_correction(band_wcs, survey):
    """Correction factor for counts summed on a grid whose pixel scale
    differs from the survey's ZP-native pixel scale.

    The AB zero points in SURVEY_DEFAULTS are calibrated for counts on the
    survey's NATIVE pixel grid (e.g. WISE 1.375"/px, GALEX 1.5"/px).
    SkyView (and any resampling service) interpolates pixel VALUES like a
    surface brightness: summing counts on a grid with pixel scale ``p``
    yields ``(p0/p)**2`` times fewer counts than the native grid ``p0``
    for the same sky aperture (verified empirically: sum x p^2 is constant
    across download scales). Multiplying summed counts by
    ``(p/p0)**2`` restores the ZP-native calibration.

    This also retroactively fixes older cached SkyView images that were
    downloaded at 2x the native pixel scale (see downloader._download_skyview).
    Returns 1.0 when the actual scale is unknown or already native.
    """
    zp_ps = SURVEY_DEFAULTS.get(survey, {}).get("pixscale")
    if not zp_ps or band_wcs is None:
        return 1.0
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        actual_ps = float(np.abs(proj_plane_pixel_scales(band_wcs)).mean()
                          * 3600.0)
    except Exception:
        return 1.0
    if not np.isfinite(actual_ps) or actual_ps <= 0:
        return 1.0
    return (actual_ps / zp_ps) ** 2


def estimate_background(data, exclude_mask=None):
    """Background level and rms from sigma-clipped stats outside exclude_mask.

    Uses sep.Background for large images (>= 128x128 with enough valid sky
    pixels); otherwise falls back to sigma_clipped_stats directly on sky pixels.
    Exact-zero pixels are excluded when they make up more than 20% of the image
    (GALEX coverage gaps, WISE padding, etc.).
    """
    if data is None:
        return 0.0, 0.0

    finite = np.isfinite(data)
    if exclude_mask is not None:
        sky = finite & (~exclude_mask)
    else:
        sky = finite

    # Exclude exact-zero pixels when they dominate — they are coverage gaps
    # (GALEX circular FOV, WISE tile edges) not real sky.
    exact_zero = finite & (data == 0)
    if exact_zero.sum() > 0.2 * max(finite.sum(), 1):
        sky = sky & ~exact_zero

    if sky.sum() < 5:
        sky = finite & ~exact_zero if exact_zero.sum() > 0 else finite

    vals = data[sky]
    if vals.size == 0:
        return 0.0, 0.0

    # Use sep.Background only for large images with ample valid sky pixels
    # (small/sparse stamps like GALEX 60×60 cause sep to return rms=1.0).
    use_sep = (data.shape[0] >= 128 and data.shape[1] >= 128
               and vals.size >= 0.1 * data.size)
    if use_sep:
        try:
            import sep
            sub = np.ascontiguousarray(data, dtype=np.float32)
            bkg = sep.Background(sub, mask=(~sky).astype(bool))
            bkg_val, rms_val = float(bkg.globalback), float(bkg.globalrms)
            # Sanity check: sep sometimes returns rms=1.0 on degenerate inputs
            if np.isfinite(rms_val) and rms_val < vals.std() * 10:
                return bkg_val, rms_val
        except Exception:
            pass

    mean, median, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
    return float(median), float(std)


def measure_mask_flux(data, mask, survey, bkg=None, rms=None, area_correction=1.0):
    """Background-subtracted flux (mJy) inside mask, with sqrt(N) error.

    area_correction: multiplicative factor applied to flux and flux_err to
    account for pixel-scale changes when images are reprojected to a grid
    with a different pixel scale than the native ZP calibration.
    area_correction = (common_pixscale / native_pixscale)^2.
    """
    if data is None or mask is None or not np.any(mask):
        return {"flux_mjy": np.nan, "flux_err_mjy": np.nan, "n_pix": 0,
                "counts": np.nan, "bkg": np.nan, "rms": np.nan}

    if bkg is None or rms is None:
        bkg, rms = estimate_background(data, exclude_mask=mask)

    pix = data[mask]
    pix = pix[np.isfinite(pix)]
    n_pix = pix.size
    counts = float(np.sum(pix - bkg))

    flux = counts_to_mjy(counts, survey) * area_correction
    counts_err = np.sqrt(max(n_pix, 0)) * rms
    flux_err = counts_to_mjy(counts_err, survey) * area_correction

    return {"flux_mjy": flux, "flux_err_mjy": abs(flux_err), "n_pix": int(n_pix),
            "counts": counts, "bkg": float(bkg), "rms": float(rms)}


def circular_mask(shape, center_xy, radius_pix):
    yy, xx = np.indices(shape)
    cx, cy = center_xy
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_pix ** 2


# spectrograph fiber diameters (arcsec) — must match the spectrum source
FIBER_DIAM_ARCSEC = {"sdss": 2.5, "desi": 1.5}


def measure_fiber_aperture_photometry(images, target_wcs, ra, dec,
                                      fiber_diam_arcsec, surveys,
                                      pixscale_arcsec=None):
    """Flux (mJy) in a fiber-equivalent circular aperture, per band.

    Measures a SMALL fixed circular aperture (diameter = the spectrograph
    fiber: 2.5\" SDSS, 1.5\" DESI) centered on the target (ra, dec) — the
    fiber pointing — so imaging photometry and fiber spectroscopy share a
    consistent aperture ("apertura imagen de la apertura espectro").

    Each band is measured on its *native*, un-convolved image
    (``entry['data']`` + ``entry['wcs']``): convolving to the common W4 PSF
    (~12\") would spread most of the light outside a 1.5-2.5\" aperture and
    make the fiber-matched flux meaningless. If a band only carries a
    reprojected image, it falls back to ``entry['reprojected']`` on the
    common grid (``target_wcs`` / ``pixscale_arcsec``). Fractional pixel
    overlap is handled exactly via photutils; if photutils is unavailable a
    10x-supersampled whole-pixel mask approximates it.

    Returns dict survey -> {flux_mjy, flux_err_mjy, n_pix, counts, bkg, rms,
    radius_pix, on_image}. ``on_image`` is False (and flux NaN) when the
    aperture falls partly outside the cutout.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales

    radius_arcsec = float(fiber_diam_arcsec) / 2.0
    results = {}

    for survey in surveys:
        entry = images.get(survey, {})
        data, wcs, ps = entry.get("data"), entry.get("wcs"), None
        if data is not None and wcs is not None:
            try:
                ps = float(np.abs(proj_plane_pixel_scales(wcs)).mean() * 3600.0)
            except Exception:
                ps = SURVEY_DEFAULTS.get(survey, {}).get("pixscale")
        if data is None or wcs is None or not ps or ps <= 0:
            data, wcs, ps = entry.get("reprojected"), target_wcs, pixscale_arcsec

        bad = {"flux_mjy": np.nan, "flux_err_mjy": np.nan, "n_pix": 0,
               "counts": np.nan, "bkg": np.nan, "rms": np.nan,
               "radius_pix": np.nan, "on_image": False}
        if data is None or wcs is None or not ps or ps <= 0:
            results[survey] = bad
            continue

        radius_pix = radius_arcsec / ps
        try:
            cx, cy = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
        except Exception:
            results[survey] = bad
            continue

        ny, nx = data.shape
        if not (cx - radius_pix >= -0.5 and cx + radius_pix <= nx - 0.5
                and cy - radius_pix >= -0.5 and cy + radius_pix <= ny - 0.5
                and np.isfinite(cx) and np.isfinite(cy)):
            logger.warning("fiber aperture off image for %s "
                           "(center %.1f,%.1f r=%.2fpx shape=%s)",
                           survey, cx, cy, radius_pix, data.shape)
            results[survey] = bad
            continue

        bkg, rms = estimate_background(data)
        clean = np.where(np.isfinite(data), data - bkg, 0.0)

        try:
            from photutils.aperture import CircularAperture, aperture_photometry
            ap = CircularAperture((cx, cy), r=radius_pix)
            counts = float(aperture_photometry(clean, ap,
                                               method="exact")["aperture_sum"][0])
            area_pix = float(ap.area)
        except Exception:
            # supersampled whole-pixel fallback
            f = 10
            big = np.repeat(np.repeat(clean, f, axis=0), f, axis=1) / f ** 2
            m = circular_mask(big.shape,
                              ((cx + 0.5) * f - 0.5, (cy + 0.5) * f - 0.5),
                              radius_pix * f)
            counts = float(np.sum(big[m]))
            area_pix = m.sum() / f ** 2

        # correct summed counts to the ZP-native pixel grid (no-op when the
        # image is already at the survey's native pixel scale)
        zp_corr = zp_grid_correction(wcs, survey)
        flux = counts_to_mjy(counts, survey) * zp_corr
        flux_err = counts_to_mjy(np.sqrt(max(area_pix, 1.0)) * rms,
                                 survey) * zp_corr

        results[survey] = {"flux_mjy": flux, "flux_err_mjy": abs(flux_err),
                           "n_pix": int(np.ceil(area_pix)), "counts": counts,
                           "bkg": float(bkg), "rms": float(rms),
                           "radius_pix": radius_pix, "on_image": True}

    return results


def measure_per_component_native(images, masks_native, mask_wcs, surveys,
                                 separation_arcsec=0.0):
    """Per-component flux at each band's *native* resolution (no convolution).

    For each band the component masks are reprojected from mask_wcs to that
    band's own WCS and flux is measured there.  This is the physically correct
    approach for merger pairs: convolution to a common PSF would dilute flux
    from one component into the other's mask.

    Adds ``blend_flag`` per band: True when PSF_FWHM > separation_arcsec,
    meaning the two components are unresolved and the per-component fluxes
    should be treated as upper limits.

    Parameters
    ----------
    images : dict
        Output of ``hostphot_downloader`` — native-resolution images.
    masks_native : dict
        Boolean masks {name: array} in mask_wcs pixel space (from xdebpair).
    mask_wcs : WCS or None
        WCS of the native r-band image used to build masks_native.
    surveys : list[str]
    separation_arcsec : float
        Nuclear separation in arcsec (for blend flagging).

    Returns
    -------
    dict  survey -> {comp_name: flux_dict, ..., 'total': flux_dict,
                     'blend_flag': bool, 'psf_fwhm_arcsec': float}
    """
    from .reprojection import reproject_masks as _reproject_masks

    sep = float(separation_arcsec) if separation_arcsec else 0.0
    results = {}

    for survey in surveys:
        meta = SURVEY_DEFAULTS.get(survey, {})
        psf_fwhm = float(meta.get("psf_fwhm", 0.0))
        blend_flag = (sep > 0.0) and (psf_fwhm >= sep)

        entry = images.get(survey, {})
        data = entry.get("data")
        band_wcs = entry.get("wcs")

        base = {"blend_flag": blend_flag, "psf_fwhm_arcsec": psf_fwhm}

        if data is None:
            results[survey] = base
            continue

        # counts are summed on this band's own grid; correct to the ZP grid
        zp_corr = zp_grid_correction(band_wcs, survey)

        # If either WCS is missing we can still measure total flux but cannot
        # reproject masks to this band's grid.
        if band_wcs is None or mask_wcs is None:
            bkg, rms = estimate_background(data)
            results[survey] = {**base,
                               "total": measure_mask_flux(data, None, survey,
                                                          bkg=bkg, rms=rms,
                                                          area_correction=zp_corr)}
            continue

        shape = data.shape
        try:
            masks_band = _reproject_masks(masks_native, mask_wcs, band_wcs, shape)
        except Exception as exc:
            logger.warning("mask reprojection failed for %s: %s", survey, exc)
            results[survey] = base
            continue

        total_mask = None
        for m in masks_band.values():
            if m is not None:
                total_mask = m.copy() if total_mask is None else (total_mask | m)

        bkg, rms = estimate_background(data, exclude_mask=total_mask)

        res = dict(base)
        for name, m in masks_band.items():
            res[name] = measure_mask_flux(data, m, survey, bkg=bkg, rms=rms,
                                          area_correction=zp_corr)
        res["total"] = measure_mask_flux(data, total_mask, survey,
                                         bkg=bkg, rms=rms,
                                         area_correction=zp_corr)
        results[survey] = res

    return results


def measure_all(images, masks, target_wcs, ra, dec, separation_arcsec,
                pixscale_arcsec, surveys, native_pixscales=None):
    """Measure component, total, and circular fluxes per band on the common grid.

    native_pixscales: dict survey -> actual native pixel scale (arcsec/px).
    Used to apply pixel-area correction so that flux is expressed in the same
    units as the native-resolution measurement. Without this correction, sums
    on the common grid over/under-estimate flux by (common_ps/native_ps)^2.

    Returns dict survey -> {"gal1": {...}, "gal2": {...}, "total": {...},
    "circular": {...}}.
    """
    shape = (int(target_wcs.array_shape[0]), int(target_wcs.array_shape[1]))

    total_mask = None
    for m in masks.values():
        if m is None:
            continue
        total_mask = m.copy() if total_mask is None else (total_mask | m)

    # circular aperture: 1.5x nuclear separation, min 5"
    sep_arcsec = separation_arcsec if (separation_arcsec and separation_arcsec > 0) else 0.0
    radius_arcsec = max(1.5 * sep_arcsec, 5.0)
    radius_pix = radius_arcsec / pixscale_arcsec
    try:
        cx, cy = target_wcs.world_to_pixel_values(ra, dec)
    except Exception:
        cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    circ_mask = circular_mask(shape, (cx, cy), radius_pix)

    results = {}
    for survey in surveys:
        entry = images.get(survey, {})
        data = entry.get("reprojected")
        bkg, rms = estimate_background(data, exclude_mask=total_mask)

        # pixel-area correction: ZP calibrated for native pixel scale;
        # reprojected image has different pixel scale.
        area_corr = 1.0
        if native_pixscales and survey in native_pixscales:
            native_ps = native_pixscales[survey]
            if native_ps and native_ps > 0:
                area_corr = (pixscale_arcsec / native_ps) ** 2

        res = {}
        for name, m in masks.items():
            res[name] = measure_mask_flux(data, m, survey, bkg=bkg, rms=rms,
                                          area_correction=area_corr)
        res["total"] = measure_mask_flux(data, total_mask, survey, bkg=bkg, rms=rms,
                                         area_correction=area_corr)
        res["circular"] = measure_mask_flux(data, circ_mask, survey, bkg=bkg, rms=rms,
                                            area_correction=area_corr)
        results[survey] = res

    return results, {"circ_radius_arcsec": radius_arcsec, "circ_mask": circ_mask,
                     "total_mask": total_mask}


def measure_isolated_photometry(images, masks_native, mask_wcs, surveys,
                                 target_wcs, common_shape, common_pixscale,
                                 separation_arcsec=0.0,
                                 target_fwhm_arcsec=None):
    """Per-component flux via isolated excess images convolved to a common PSF.

    For each band x component we build an image that is zero everywhere except
    inside the component mask (where it holds data - background), convolve it to
    the target PSF, and reproject it to the common grid. The PSF spread leaks
    into empty space rather than the neighbouring component, so close-pair flux
    is not cross-contaminated while total flux is conserved.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales
    from .convolution import matching_kernel, TARGET_SURVEY
    from .reprojection import reproject_masks as _reproject_masks

    if target_fwhm_arcsec is None:
        target_fwhm_arcsec = SURVEY_DEFAULTS[TARGET_SURVEY]["psf_fwhm"]

    sep = float(separation_arcsec) if separation_arcsec else 0.0
    results = {}

    for survey in surveys:
        meta = SURVEY_DEFAULTS.get(survey, {})
        psf_fwhm = float(meta.get("psf_fwhm", 0.0))
        blend_flag = (sep > 0.0) and (psf_fwhm >= sep)
        base = {"blend_flag": blend_flag, "psf_fwhm_arcsec": psf_fwhm}

        entry = images.get(survey, {})
        data = entry.get("data")
        band_wcs = entry.get("wcs")
        if data is None or band_wcs is None or mask_wcs is None:
            results[survey] = base
            continue

        native_pixscale = None
        try:
            native_pixscale = float(abs(proj_plane_pixel_scales(band_wcs)).mean() * 3600.0)
        except Exception:
            native_pixscale = meta.get("pixscale")
        if not native_pixscale or native_pixscale <= 0:
            native_pixscale = meta.get("pixscale", common_pixscale)

        # counts are summed on this band's own grid, so the only correction
        # needed is to the ZP-native pixel scale (see zp_grid_correction).
        zp_corr = zp_grid_correction(band_wcs, survey)

        try:
            masks_band = _reproject_masks(masks_native, mask_wcs, band_wcs, data.shape)
        except Exception as exc:
            logger.warning("mask reprojection failed for %s: %s", survey, exc)
            results[survey] = base
            continue

        total_mask = None
        for m in masks_band.values():
            if m is None:
                continue
            total_mask = m.copy() if total_mask is None else (total_mask | m)

        bkg, rms = estimate_background(data, exclude_mask=total_mask)

        source_fwhm = meta.get("psf_fwhm", 0.0)
        kernel, _ = matching_kernel(source_fwhm, target_fwhm_arcsec, native_pixscale)

        # Flux is conserved through convolution (normalized kernel), so we
        # measure on the convolved native image — no reprojection needed.
        # zp_corr rescales the summed counts to the ZP-native pixel grid
        # when the downloaded image is at a different pixel scale.
        def _measure(mask):
            if mask is None or not np.any(mask):
                return {"flux_mjy": np.nan, "flux_err_mjy": np.nan,
                        "n_pix": 0, "counts": np.nan}
            n_pix = int(mask.sum())
            img_isolated = np.zeros_like(data, dtype=float)
            img_isolated[mask] = data[mask] - bkg
            if kernel is not None:
                img_conv = convolve_fft(
                    img_isolated, kernel, normalize_kernel=True,
                    allow_huge=True, nan_treatment="fill", fill_value=0.0,
                )
            else:
                img_conv = img_isolated
            counts = float(np.nansum(img_conv))
            flux_mjy = counts_to_mjy(counts, survey) * zp_corr
            counts_err = np.sqrt(max(n_pix, 1)) * rms
            flux_err_mjy = counts_to_mjy(counts_err, survey) * zp_corr
            return {"flux_mjy": flux_mjy, "flux_err_mjy": abs(flux_err_mjy),
                    "n_pix": n_pix, "counts": counts}

        res = dict(base)
        for name, m in masks_band.items():
            res[name] = _measure(m)

        # total uses the union mask — avoids double-counting pixels shared
        # between components at the mask boundary
        tot = _measure(total_mask)
        res["total"] = tot
        res["n_pix"] = tot["n_pix"]
        results[survey] = res

    return results


# ===========================================================================
# aperture_mode = "sep_apertures": SEP-detection-driven elliptical apertures
# ===========================================================================

# how many times the SEP (a, b) half-axes the measuring ellipse spans;
# 2.5 is the classic Kron-like factor used by SExtractor AUTO photometry
SEP_ELLIPSE_K = 2.5


def detect_sep_ellipse(images, ra, dec, ref_survey="Legacy_r",
                       thresh_sigma=1.5, max_match_arcsec=10.0):
    """Detect the target with SEP on the reference band and return its ellipse.

    Runs ``sep.extract`` on the background-subtracted reference image and
    picks the detected source closest to the target (ra, dec) (within
    ``max_match_arcsec``). The SEP shape parameters (a, b, theta) define the
    elliptical aperture used by :func:`measure_sep_elliptical_photometry`.

    Returns a dict with the ellipse in SKY units so it can be placed on any
    band's pixel grid:
    {ra, dec, a_arcsec, b_arcsec, theta_rad, n_detected, match_sep_arcsec}
    or None when detection/matching fails.

    theta convention: SEP measures theta counter-clockwise from the +x axis
    of the reference image. All cutouts in this pipeline are north-up TAN
    grids (RA along -x, Dec along +y), so the angle transfers directly
    between bands without rotation.
    """
    import sep as sep_lib
    from astropy.wcs.utils import proj_plane_pixel_scales

    entry = images.get(ref_survey, {})
    data, wcs = entry.get("data"), entry.get("wcs")
    if data is None or wcs is None:
        logger.warning("sep_apertures: reference band %s unavailable",
                       ref_survey)
        return None

    img = np.ascontiguousarray(np.nan_to_num(data), dtype=np.float32)
    try:
        bkg = sep_lib.Background(img)
        sub = img - bkg.back()
        objects = sep_lib.extract(sub, thresh_sigma, err=bkg.globalrms)
    except Exception as exc:
        logger.warning("sep_apertures: SEP extraction failed: %s", exc)
        return None
    if len(objects) == 0:
        logger.warning("sep_apertures: no sources detected on %s", ref_survey)
        return None

    try:
        ps = float(np.abs(proj_plane_pixel_scales(wcs)).mean() * 3600.0)
        cx, cy = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
    except Exception as exc:
        logger.warning("sep_apertures: WCS failure on %s: %s", ref_survey, exc)
        return None

    d_pix = np.hypot(objects["x"] - cx, objects["y"] - cy)
    i = int(np.argmin(d_pix))
    match_sep = float(d_pix[i] * ps)
    if match_sep > max_match_arcsec:
        logger.warning("sep_apertures: nearest SEP source is %.1f\" from the "
                       "target (max %.1f\"); no match", match_sep,
                       max_match_arcsec)
        return None

    obj = objects[i]
    sky = wcs.pixel_to_world(float(obj["x"]), float(obj["y"]))
    return {
        "ra": float(sky.ra.deg), "dec": float(sky.dec.deg),
        "a_arcsec": float(obj["a"]) * ps, "b_arcsec": float(obj["b"]) * ps,
        "theta_rad": float(obj["theta"]),
        "n_detected": int(len(objects)),
        "match_sep_arcsec": match_sep,
    }


def sep_ellipse_mask(ellipse, wcs, shape, k=SEP_ELLIPSE_K):
    """Boolean pixel mask of the k-scaled SEP ellipse on the given grid.

    Used so the sep_apertures mode plugs into the same plotting / table
    machinery as the mask-based modes.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales

    if ellipse is None or wcs is None:
        return None
    try:
        ps = float(np.abs(proj_plane_pixel_scales(wcs)).mean() * 3600.0)
        cx, cy = (float(v) for v in
                  wcs.world_to_pixel_values(ellipse["ra"], ellipse["dec"]))
    except Exception:
        return None
    a = k * ellipse["a_arcsec"] / ps
    b = k * ellipse["b_arcsec"] / ps
    th = ellipse["theta_rad"]
    yy, xx = np.indices(shape, dtype=float)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(th) + dy * np.sin(th)
    v = -dx * np.sin(th) + dy * np.cos(th)
    return (u / max(a, 1e-6)) ** 2 + (v / max(b, 1e-6)) ** 2 <= 1.0


def measure_sep_elliptical_photometry(images, surveys, ra, dec,
                                      ref_survey="Legacy_r",
                                      k=SEP_ELLIPSE_K):
    """Elliptical-aperture photometry from a SEP detection (no masks/xdebpair).

    The ellipse (a, b, theta) is detected once on the reference band
    (:func:`detect_sep_ellipse`) and the SAME sky ellipse, scaled by ``k``,
    is measured on every band's native image with
    ``photutils.aperture.EllipticalAperture`` (exact fractional-pixel
    overlap), mirroring the circular-aperture code style used elsewhere in
    this module.

    Returns (photometry, ellipse):
      photometry: dict survey -> {"gal1": flux_dict, "total": flux_dict,
                                  "blend_flag": False, "psf_fwhm_arcsec": ...}
      (same shape as measure_isolated_photometry, so output_table /
      plotting code needs no special-casing), or ({}, None) on failure.
    """
    from astropy.wcs.utils import proj_plane_pixel_scales
    from photutils.aperture import EllipticalAperture, aperture_photometry

    ellipse = detect_sep_ellipse(images, ra, dec, ref_survey=ref_survey)
    if ellipse is None:
        return {}, None

    results = {}
    for survey in surveys:
        meta = SURVEY_DEFAULTS.get(survey, {})
        base = {"blend_flag": False,
                "psf_fwhm_arcsec": float(meta.get("psf_fwhm", 0.0))}
        bad = {"flux_mjy": np.nan, "flux_err_mjy": np.nan, "n_pix": 0,
               "counts": np.nan, "bkg": np.nan, "rms": np.nan}

        entry = images.get(survey, {})
        data, wcs = entry.get("data"), entry.get("wcs")
        if data is None or wcs is None:
            results[survey] = {**base, "gal1": dict(bad), "total": dict(bad)}
            continue
        try:
            ps = float(np.abs(proj_plane_pixel_scales(wcs)).mean() * 3600.0)
            cx, cy = (float(v) for v in
                      wcs.world_to_pixel_values(ellipse["ra"], ellipse["dec"]))
        except Exception:
            results[survey] = {**base, "gal1": dict(bad), "total": dict(bad)}
            continue

        a_pix = k * ellipse["a_arcsec"] / ps
        b_pix = k * ellipse["b_arcsec"] / ps
        excl = sep_ellipse_mask(ellipse, wcs, data.shape, k=k)
        bkg, rms = estimate_background(data, exclude_mask=excl)
        clean = np.where(np.isfinite(data), data - bkg, 0.0)

        try:
            ap = EllipticalAperture((cx, cy), a=max(a_pix, 0.5),
                                    b=max(b_pix, 0.5),
                                    theta=ellipse["theta_rad"])
            counts = float(aperture_photometry(clean, ap,
                                               method="exact")["aperture_sum"][0])
            area_pix = float(ap.area)
        except Exception as exc:
            logger.warning("sep_apertures: photometry failed for %s: %s",
                           survey, exc)
            results[survey] = {**base, "gal1": dict(bad), "total": dict(bad)}
            continue

        zp_corr = zp_grid_correction(wcs, survey)
        flux = counts_to_mjy(counts, survey) * zp_corr
        flux_err = counts_to_mjy(np.sqrt(max(area_pix, 1.0)) * rms,
                                 survey) * zp_corr
        m = {"flux_mjy": flux, "flux_err_mjy": abs(flux_err),
             "n_pix": int(np.ceil(area_pix)), "counts": counts,
             "bkg": float(bkg), "rms": float(rms)}
        # single elliptical aperture: the one component IS the total
        results[survey] = {**base, "gal1": dict(m), "total": dict(m)}

    return results, ellipse


def measure_circular_aperture_photometry(images, surveys, ra, dec,
                                         radius_arcsec):
    """Fixed circular-aperture photometry per band (aperture_mode="aperture").

    Thin wrapper around :func:`measure_fiber_aperture_photometry` (which
    already does exact circular photometry on each band's native grid) that
    reshapes its output to the {survey: {gal1, total, blend_flag}} structure
    the rest of the pipeline consumes.
    """
    fiber = measure_fiber_aperture_photometry(
        images, None, ra, dec, 2.0 * float(radius_arcsec), surveys)
    results = {}
    for survey in surveys:
        meta = SURVEY_DEFAULTS.get(survey, {})
        m = dict(fiber.get(survey, {}))
        m.pop("radius_pix", None)
        m.pop("on_image", None)
        results[survey] = {"blend_flag": False,
                           "psf_fwhm_arcsec": float(meta.get("psf_fwhm", 0.0)),
                           "gal1": dict(m), "total": dict(m)}
    return results
