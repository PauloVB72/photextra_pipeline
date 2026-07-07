"""Convolve all bands to the common (WISE W4, 12" FWHM) PSF.

We approximate each band's PSF as a Gaussian and build a matching kernel so that
sqrt(target_fwhm^2 - source_fwhm^2) is applied per band. Bands already at or
above the target resolution are left untouched (with a warning) since
deconvolution is not attempted.
"""

import logging

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft

from .downloader import SURVEY_DEFAULTS

logger = logging.getLogger(__name__)

TARGET_SURVEY = "WISE_W4"
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def matching_kernel(source_fwhm_arcsec, target_fwhm_arcsec, pixscale_arcsec):
    """Gaussian matching kernel that takes source PSF -> target PSF.

    Returns (kernel_array, sigma_pix). Returns (None, 0) when the source is
    already broader than the target.
    """
    if target_fwhm_arcsec <= source_fwhm_arcsec:
        return None, 0.0

    fwhm_diff = np.sqrt(target_fwhm_arcsec ** 2 - source_fwhm_arcsec ** 2)
    sigma_pix = (fwhm_diff * _FWHM_TO_SIGMA) / pixscale_arcsec
    if sigma_pix <= 0:
        return None, 0.0

    kernel = Gaussian2DKernel(x_stddev=sigma_pix)
    kernel.normalize()
    return kernel.array, sigma_pix


def convolve_image(data, survey, target_fwhm_arcsec=None, pixscale_override=None):
    """Convolve one band's data to the common PSF.

    Returns (convolved_data, kernel_array). kernel_array is None when no
    convolution was applied.

    pixscale_override: actual pixel scale in arcsec/px read from the WCS.
    Falls back to SURVEY_DEFAULTS when None.
    """
    if data is None:
        return None, None

    meta = SURVEY_DEFAULTS[survey]
    pixscale = pixscale_override if pixscale_override is not None else meta["pixscale"]
    source_fwhm = meta["psf_fwhm"]
    if target_fwhm_arcsec is None:
        target_fwhm_arcsec = SURVEY_DEFAULTS[TARGET_SURVEY]["psf_fwhm"]

    kernel, sigma_pix = matching_kernel(source_fwhm, target_fwhm_arcsec, pixscale)
    if kernel is None:
        logger.info("%s already at/above target PSF, no convolution", survey)
        return data.copy(), None

    try:
        conv = convolve_fft(
            data, kernel,
            normalize_kernel=True,
            preserve_nan=True,
            allow_huge=True,
            nan_treatment="interpolate",
        )
    except Exception as exc:
        logger.error("Convolution failed for %s: %s", survey, exc)
        return data.copy(), None

    return conv, kernel


def _pixscale_from_wcs(wcs):
    """Return pixel scale in arcsec from an astropy WCS, or None on failure."""
    if wcs is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(wcs)
        return float(abs(scales).mean() * 3600.0)
    except Exception:
        return None


def convolve_all(images, target_fwhm_arcsec=None):
    """images: survey -> dict with 'data'. Returns survey -> dict adding
    'convolved' and 'kernel'.

    Pixel scale is read from the WCS of each image; SURVEY_DEFAULTS is used
    only as a fallback so that incorrect nominal values do not propagate into
    the matching-kernel computation.
    """
    out = {}
    for survey, entry in images.items():
        actual_pixscale = _pixscale_from_wcs(entry.get("wcs"))
        conv, kernel = convolve_image(entry.get("data"), survey,
                                      target_fwhm_arcsec,
                                      pixscale_override=actual_pixscale)
        new = dict(entry)
        new["convolved"] = conv
        new["kernel"] = kernel
        out[survey] = new
    return out
