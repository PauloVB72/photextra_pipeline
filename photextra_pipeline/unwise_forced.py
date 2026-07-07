"""Fetch unWISE forced photometry from the Legacy Survey Tractor catalog.

Legacy Survey DR10 fits PSF/galaxy profiles at optically-known positions on
unWISE coadd images.  For merger pairs resolved in grz (0.262"/px), this
delivers individual W1/W2 fluxes per component without any assumption about
colour ratios (unlike ratio_prior) or profile-normalization artefacts
(unlike our simplified T-PHOT).

Fluxes are returned in mJy.  Conversion: 1 nanomaggy = 3.631e-3 mJy (AB).

Limitations (see module docstring for details):
  - Only W1 and W2 (not W3/W4).
  - Requires Legacy Survey footprint coverage.
  - Profile fitting assumes exponential/deVauc/PSF — poor for disturbed morphologies.
  - Position prior from optical centroid; fails if IR nucleus is dust-offset.
  - DR10 may have different deblending than DR8 IDs in mm.csv.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

# retry/backoff policy for the Vizier query (same pattern as the Legacy
# Survey cutouts in downloader.py, which did hit 429s at scale)
RETRY_ATTEMPTS = 5


class UnwiseQueryError(RuntimeError):
    """The unWISE catalog query failed after all retries (rate limit /
    network / server error). Raised instead of silently returning no
    matches so the caller can record the failure for a later re-run."""

# unWISE source catalog (Schlafly+2019) — Vizier II/363
# Fluxes are in Vega nanomaggies; Vega zero points per band:
#   W1: 309.054 Jy  →  1 Vega nanomaggy = 309.054e-9 Jy = 3.09054e-4 mJy
#   W2: 171.787 Jy  →  1 Vega nanomaggy = 171.787e-9 Jy = 1.71787e-4 mJy
_UNWISE_VIZIER_CATALOG = "II/363/unwise"
_VEGA_NM_TO_MJY = {"WISE_W1": 3.09054e-4, "WISE_W2": 1.71787e-4}

# W1/W2 metadata consistent with SURVEY_DEFAULTS
_WISE_META = {
    "WISE_W1": {"lambda_eff": 3.4,  "psf_fwhm": 6.1},
    "WISE_W2": {"lambda_eff": 4.6,  "psf_fwhm": 6.4},
}


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _mask_centroid_sky(mask, wcs):
    """Return (ra, dec) centroid of a boolean mask in sky coordinates."""
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return None, None
    cx, cy = float(xx.mean()), float(yy.mean())
    try:
        ra, dec = wcs.pixel_to_world_values(cx, cy)
        return float(ra), float(dec)
    except Exception:
        return None, None


def _angular_sep_deg(ra1, dec1, ra2, dec2):
    """Great-circle separation in degrees (Vincenty small-angle approx)."""
    d2r = np.pi / 180.0
    dra = (ra2 - ra1) * d2r
    ddec = (dec2 - dec1) * d2r
    a = np.sin(ddec / 2) ** 2 + np.cos(dec1 * d2r) * np.cos(dec2 * d2r) * np.sin(dra / 2) ** 2
    return 2.0 * np.degrees(np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def _query_unwise_catalog(ra, dec, radius_arcsec=60):
    """Query unWISE source catalog (Schlafly+2019, Vizier II/363) around (ra, dec).

    Returns list of source dicts with keys:
      ra, dec, FW1, FW2, e_FW1, e_FW2, q_W1, q_W2
    Fluxes are in Vega nanomaggies (see _VEGA_NM_TO_MJY for conversion).

    An EMPTY catalog result (no sources in the cone) legitimately returns
    []. A query FAILURE (network / rate limit / server error) is retried
    with exponential backoff + jitter (same policy as the Legacy Survey
    cutouts in downloader.py); after exhausting the retries an
    :class:`UnwiseQueryError` is raised — no silent degradation.
    """
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # reuse the shared backoff schedule (honors Retry-After when given,
    # else exponential with jitter); astroquery does not expose response
    # headers, so retry_after stays None here.
    from .downloader import _retry_sleep_seconds

    last_exc = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            v = Vizier(
                columns=["RAJ2000", "DEJ2000",
                         "FW1", "e_FW1", "q_W1",
                         "FW2", "e_FW2", "q_W2"],
                row_limit=500,
            )
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            result = v.query_region(coord, radius=radius_arcsec * u.arcsec,
                                    catalog=_UNWISE_VIZIER_CATALOG)
        except Exception as exc:
            last_exc = exc
            sleep = _retry_sleep_seconds(attempt)
            logger.warning("unWISE: Vizier query failed (%s); retry %d/%d "
                           "in %.1fs", exc, attempt + 1, RETRY_ATTEMPTS,
                           sleep)
            time.sleep(sleep)
            continue

        if not result or len(result) == 0:
            return []

        tbl = result[0]
        sources = []
        for row in tbl:
            try:
                src = {
                    "ra":    float(row["RAJ2000"]),
                    "dec":   float(row["DEJ2000"]),
                    "FW1":   float(row["FW1"])   if row["FW1"]   is not None else np.nan,
                    "e_FW1": float(row["e_FW1"]) if row["e_FW1"] is not None else np.nan,
                    "q_W1":  float(row["q_W1"])  if row["q_W1"]  is not None else 0.0,
                    "FW2":   float(row["FW2"])   if row["FW2"]   is not None else np.nan,
                    "e_FW2": float(row["e_FW2"]) if row["e_FW2"] is not None else np.nan,
                    "q_W2":  float(row["q_W2"])  if row["q_W2"]  is not None else 0.0,
                }
                sources.append(src)
            except Exception:
                continue
        return sources

    raise UnwiseQueryError(
        f"unWISE Vizier query failed after {RETRY_ATTEMPTS} attempts "
        f"(rate limit / network / server error): {last_exc}")


def _source_w1w2(src):
    """Extract W1/W2 flux (mJy) from an unWISE catalog source dict.

    Fluxes (FW1/FW2) are in Vega nanomaggies. Quality flag q_W1/q_W2 = 1.0
    means the source was detected; 0 means only an upper limit.
    """
    result = {}
    for band, fkey, ekey, qkey in (
        ("WISE_W1", "FW1", "e_FW1", "q_W1"),
        ("WISE_W2", "FW2", "e_FW2", "q_W2"),
    ):
        conv = _VEGA_NM_TO_MJY[band]
        fnm = src.get(fkey, np.nan)
        enm = src.get(ekey, np.nan)
        qflag = src.get(qkey, 0.0)
        if not np.isfinite(fnm):
            continue
        flux_mjy = fnm * conv
        err_mjy  = enm * conv if np.isfinite(enm) and enm > 0 else np.nan
        result[band] = {
            "flux_mjy":     flux_mjy,
            "flux_err_mjy": err_mjy,
            "method":       "unwise_catalog",
            "q_flag":       qflag,        # 1=detected, 0=upper limit
            "flux_vega_nm": fnm,
        }
    return result


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def query_unwise_forced(masks_native, mask_wcs, ra_center, dec_center,
                         match_radius_arcsec=3.0, query_radius_arcsec=60):
    """Return W1/W2 forced-photometry fluxes per component from Legacy Survey DR10.

    Parameters
    ----------
    masks_native : dict  {comp_name: bool_array}
        Component masks in mask_wcs pixel space (from xdebpair).
    mask_wcs : astropy.wcs.WCS
        WCS of the native Legacy_r image used to build masks_native.
    ra_center, dec_center : float
        Field centre for the catalog query.
    match_radius_arcsec : float
        Max angular distance (arcsec) to accept a catalog match.
    query_radius_arcsec : float
        Cone radius for the catalog query.

    Returns
    -------
    dict  survey -> {comp_name: {flux_mjy, flux_err_mjy, method, ra_cat, dec_cat}, ...}
        Only W1/W2 entries that were successfully matched are included.
        Returns {} when no sources/matches exist; raises
        :class:`UnwiseQueryError` when the catalog query itself fails
        after all retries (the pipeline records it and continues).
    """
    if mask_wcs is None or not masks_native:
        return {}

    # compute sky centroids for each component
    centroids = {}
    for name, mask in masks_native.items():
        if mask is None or not np.any(mask):
            continue
        ra, dec = _mask_centroid_sky(mask, mask_wcs)
        if ra is not None:
            centroids[name] = (ra, dec)

    if not centroids:
        return {}

    sources = _query_unwise_catalog(ra_center, dec_center, query_radius_arcsec)
    if not sources:
        return {}

    logger.info("unWISE forced: %d LS sources in %.0f\" cone", len(sources), query_radius_arcsec)

    # for each component, find nearest catalog source within match_radius
    per_comp = {}
    for name, (cra, cdec) in centroids.items():
        best_sep = np.inf
        best_src = None
        for src in sources:
            sra = src.get("ra") or src.get("RA")
            sdec = src.get("dec") or src.get("DEC")
            if sra is None or sdec is None:
                continue
            sep_arcsec = _angular_sep_deg(cra, cdec, float(sra), float(sdec)) * 3600.0
            if sep_arcsec < best_sep:
                best_sep = sep_arcsec
                best_src = src

        if best_src is None or best_sep > match_radius_arcsec:
            logger.warning("unWISE forced: no match for %s within %.1f\" (best=%.2f\")",
                           name, match_radius_arcsec, best_sep)
            continue

        w1w2 = _source_w1w2(best_src)
        if not w1w2:
            logger.warning("unWISE forced: matched %s but no W1/W2 keys in source", name)
            continue

        sra = best_src.get("ra") or best_src.get("RA")
        sdec = best_src.get("dec") or best_src.get("DEC")
        for band, entry in w1w2.items():
            entry["ra_cat"] = float(sra)
            entry["dec_cat"] = float(sdec)
            entry["match_sep_arcsec"] = float(best_sep)
        per_comp[name] = w1w2
        logger.info("unWISE forced: %s matched at %.2f\" -> W1=%.4g mJy  W2=%.4g mJy",
                    name, best_sep,
                    w1w2.get("WISE_W1", {}).get("flux_mjy", np.nan),
                    w1w2.get("WISE_W2", {}).get("flux_mjy", np.nan))

    if not per_comp:
        return {}

    # reorganise: survey -> {comp: entry}
    out = {}
    for name, w1w2 in per_comp.items():
        for band, entry in w1w2.items():
            out.setdefault(band, {})[name] = entry

    return out
