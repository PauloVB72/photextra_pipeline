"""XdebPair adapter for photextra_pipeline.

Wraps xdebpair to produce per-component pixel masks in Legacy r-band pixel
space, with a matching WCS, ready for ``reproject_masks``.

The result object is intentionally compatible with the xmask result interface
used throughout the pipeline (masks, wcs, n_components, separation_arcsec,
is_merger).
"""

import logging
import sys
import os

import numpy as np

logger = logging.getLogger(__name__)

# Legacy DR10 pixel scale (arcsec/px) — fixed for all cutouts from the viewer
_LEGACY_PIXSCALE = 0.262


class XdebPairResult:
    """Minimal result container compatible with the xmask result interface."""

    def __init__(self, masks, wcs, n_components, separation_arcsec,
                 classification, nuclei_yx=None, xdeb_result=None):
        self.masks = masks
        self.wcs = wcs
        self.n_components = n_components
        self.separation_arcsec = separation_arcsec
        self.is_merger = n_components > 1
        self.classification = classification
        self.nuclei_yx = nuclei_yx
        self.xdeb_result = xdeb_result

    def __repr__(self):
        return (f"XdebPairResult(class={self.classification}, "
                f"n_comp={self.n_components}, "
                f"sep={self.separation_arcsec:.1f}\", "
                f"masks={list(self.masks.keys())})")


class XdebPairAdapter:
    """Run xdebpair on already-downloaded images and return a pipeline-ready result.

    Parameters
    ----------
    ra, dec : float
        Target centre in degrees.
    images : dict
        Output from ``hostphot_downloader`` — keys are survey names
        (e.g. 'Legacy_r'), values are dicts with 'data', 'wcs', 'header'.
        Needs at least 'Legacy_r'; uses 'Legacy_g' and 'Legacy_z' for the
        colour cube if available.
    verbose : bool
        Pass-through to XdebPair.
    """

    def __init__(self, ra, dec, images, verbose=False):
        self.ra = ra
        self.dec = dec
        self.images = images
        self.verbose = verbose

    def run(self):
        # --- import xdebpair --------------------------------------------------
        try:
            _xdeb_path = "/home/polo/Escritorio/PHD/code/xdebpair"
            if _xdeb_path not in sys.path:
                sys.path.insert(0, _xdeb_path)
            from xdebpair import XdebPair
        except ImportError as exc:
            raise RuntimeError(
                f"xdebpair not importable ({exc}). "
                f"Check that {_xdeb_path} exists."
            ) from exc

        # --- pull Legacy images -----------------------------------------------
        r_entry = self.images.get("Legacy_r", {})
        r_data = r_entry.get("data")
        r_wcs = r_entry.get("wcs")

        if r_data is None:
            raise ValueError(
                "Legacy_r image not available — "
                "add 'Legacy_r' to the pipeline surveys list."
            )

        # optional colour cube (g, r, z) for xdebpair
        bands = []
        for band in ("Legacy_g", "Legacy_r", "Legacy_z"):
            entry = self.images.get(band, {})
            data = entry.get("data")
            if data is not None and data.shape == r_data.shape:
                bands.append(data)
        cube = np.array(bands) if len(bands) > 1 else None

        # --- run xdebpair -----------------------------------------------------
        deb = XdebPair(use_nmf=False, verbose=self.verbose)
        try:
            res = deb.fit(r_data, wcs=r_wcs, ra=self.ra, dec=self.dec,
                          image_cube=cube)
        except Exception as exc:
            raise RuntimeError(f"XdebPair.fit() failed: {exc}") from exc

        # --- convert result to pipeline interface -----------------------------
        classification = res.classification
        sep_arcsec = res.separation_px * _LEGACY_PIXSCALE

        if classification == "single":
            masks = {"gal1": res.masks["gal1"]}
            n_comp = 1
        else:
            masks = {"gal1": res.masks["gal1"], "gal2": res.masks["gal2"]}
            n_comp = 2

        logger.info(
            "xdebpair: class=%s  sep=%.1fpx (%.1f\")  masks=%s",
            classification, res.separation_px, sep_arcsec, list(masks.keys()),
        )

        return XdebPairResult(
            masks=masks,
            wcs=r_wcs,
            n_components=n_comp,
            separation_arcsec=sep_arcsec,
            classification=classification,
            nuclei_yx=res.nuclei_yx,
            xdeb_result=res,
        )
