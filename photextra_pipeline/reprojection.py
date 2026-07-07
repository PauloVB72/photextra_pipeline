"""Reproject convolved images and xmask masks onto a common WISE-W4 grid."""

import logging

import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from reproject import reproject_interp

logger = logging.getLogger(__name__)


def build_common_wcs(ra, dec, pixscale_arcsec, size_pix):
    """TAN WCS centered on (ra, dec) for a size_pix square grid."""
    wcs = WCS(naxis=2)
    crpix = (size_pix + 1) / 2.0
    wcs.wcs.crpix = [crpix, crpix]
    wcs.wcs.crval = [ra, dec]
    deg = pixscale_arcsec / 3600.0
    wcs.wcs.cdelt = [-deg, deg]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.array_shape = (size_pix, size_pix)
    return wcs


def reproject_image(data, src_wcs, target_wcs, shape, order="bilinear"):
    if data is None or src_wcs is None:
        return None
    try:
        out, _ = reproject_interp((data, src_wcs), target_wcs, shape_out=shape, order=order)
    except Exception as exc:
        logger.error("Reprojection failed: %s", exc)
        return None
    return out


def reproject_all(images, target_wcs, shape):
    """Reproject each band's convolved data onto the common grid."""
    out = {}
    for survey, entry in images.items():
        src = entry.get("convolved")
        if src is None:
            src = entry.get("data")
        rep = reproject_image(src, entry.get("wcs"), target_wcs, shape, order="bilinear")
        new = dict(entry)
        new["reprojected"] = rep
        out[survey] = new
    return out


def reproject_masks(masks, mask_wcs, target_wcs, shape):
    """Reproject boolean component masks with nearest-neighbour (order=0)."""
    out = {}
    if masks is None or mask_wcs is None:
        return out
    for name, mask in masks.items():
        rep = reproject_image(
            np.asarray(mask, dtype=float), mask_wcs, target_wcs, shape, order="nearest-neighbor"
        )
        if rep is None:
            out[name] = None
        else:
            out[name] = np.nan_to_num(rep) > 0.5
    return out
