"""Clean per-component output table for the photextra_pipeline.

Produces one row per galaxy component (gal1, gal2, ...) with clean,
publication-ready column names. For non-blended bands the direct mask flux is
used; for blended bands the direct flux column is NaN and the comb / ratio
deblended estimates carry the reliable values.
"""

import logging

import numpy as np
from astropy.table import Table

from .downloader import SURVEY_DEFAULTS

logger = logging.getLogger(__name__)


# internal survey name -> clean column prefix
SURVEY_PREFIX = {
    "GALEX_FUV": "GALEX_FUV",
    "GALEX_NUV": "GALEX_NUV",
    "Legacy_g": "Legacy_Survey_g",
    "Legacy_r": "Legacy_Survey_r",
    "Legacy_z": "Legacy_Survey_z",
    "WISE_W1": "WISE_W1",
    "WISE_W2": "WISE_W2",
}

# canonical survey ordering FUV, NUV, g, r, z, W1, W2
_SURVEY_ORDER = ["GALEX_FUV", "GALEX_NUV", "Legacy_g", "Legacy_r", "Legacy_z",
                 "WISE_W1", "WISE_W2"]


def _prefix(survey):
    return SURVEY_PREFIX.get(survey, survey)


def _ordered_surveys(surveys):
    """Surveys in the canonical FUV..W2 order, dropping unknowns at the end."""
    known = [s for s in _SURVEY_ORDER if s in surveys]
    extra = [s for s in surveys if s not in _SURVEY_ORDER]
    return known + extra


def _component_centroid(mask, wcs):
    """Return (ra_deg, dec_deg) for the centroid of a boolean mask, or NaN."""
    if mask is None or wcs is None:
        return np.nan, np.nan
    try:
        if not np.any(mask):
            return np.nan, np.nan
        yy, xx = np.where(mask)
        xc, yc = float(xx.mean()), float(yy.mean())
        sky = wcs.pixel_to_world(xc, yc)
        return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        return np.nan, np.nan


def _get(d, *keys, default=np.nan):
    """Nested .get() that tolerates missing dicts / non-dict values."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    if cur is None:
        return default
    return cur


def make_galaxy_table(target_id, surveys, photometry, seg_result,
                      deblend_ratio=None, deblend_tphot=None,
                      deblend_comb=None, unwise_forced=None):
    """Build a one-row-per-component flux table.

    Parameters mirror the pipeline data structures. Returns an
    ``astropy.table.Table`` with clean column names (see module docstring).
    """
    deblend_ratio = deblend_ratio or {}
    deblend_tphot = deblend_tphot or {}
    deblend_comb = deblend_comb or {}
    unwise_forced = unwise_forced or {}

    masks = getattr(seg_result, "masks", {}) or {}
    wcs = getattr(seg_result, "wcs", None)
    n_components = getattr(seg_result, "n_components", len(masks)) or len(masks)
    separation = getattr(seg_result, "separation_arcsec", 0.0)
    if separation is None:
        separation = 0.0

    # component names: prefer mask order, fall back to gal1
    comp_names = [c for c in masks.keys()] or ["gal1"]

    surveys_ord = _ordered_surveys(surveys)

    rows = []
    for comp in comp_names:
        ra_c, dec_c = _component_centroid(masks.get(comp), wcs)
        row = {
            "target_id": str(target_id),
            "component": str(comp),
            "ra_deg": ra_c,
            "dec_deg": dec_c,
            "n_components": int(n_components),
            "separation_arcsec": float(separation),
            # separation-policy bookkeeping (photometry.separation config +
            # optional per-target "type" column; see
            # Pipeline._apply_separation_policy). has_companion_not_measured
            # is True ONLY in separation="central" when xdebpair detected a
            # companion whose flux was deliberately NOT measured.
            "separation_policy": str(getattr(seg_result,
                                             "separation_policy", "")),
            "target_type": str(getattr(seg_result, "target_type", "")),
            "n_components_detected": int(getattr(
                seg_result, "n_components_detected", n_components)),
            "has_companion_not_measured": bool(getattr(
                seg_result, "has_companion_not_measured", False)),
        }

        for survey in surveys_ord:
            pfx = _prefix(survey)
            phot = photometry.get(survey, {}) if isinstance(photometry, dict) else {}
            blended = bool(phot.get("blend_flag", False))

            mask_flux = _get(phot, comp, "flux_mjy")
            mask_err = _get(phot, comp, "flux_err_mjy")

            # direct flux only trustworthy when not blended
            if blended:
                row[f"{pfx}_flux_mjy"] = np.nan
                row[f"{pfx}_flux_err_mjy"] = np.nan
            else:
                row[f"{pfx}_flux_mjy"] = float(mask_flux) if np.isfinite(_to_float(mask_flux)) else np.nan
                row[f"{pfx}_flux_err_mjy"] = float(mask_err) if np.isfinite(_to_float(mask_err)) else np.nan

            row[f"{pfx}_blend_flag"] = int(blended)

            # combined deblended (best estimate for blended bands)
            if blended:
                row[f"{pfx}_comb_flux_mjy"] = _to_float(_get(deblend_comb, survey, comp, "flux_mjy"))
                row[f"{pfx}_comb_flux_err_mjy"] = _to_float(_get(deblend_comb, survey, comp, "flux_err_mjy"))
                row[f"{pfx}_ratio_flux_mjy"] = _to_float(_get(deblend_ratio, survey, comp, "flux_mjy"))
                row[f"{pfx}_ratio_flux_err_mjy"] = _to_float(_get(deblend_ratio, survey, comp, "flux_err_mjy"))
            else:
                row[f"{pfx}_comb_flux_mjy"] = np.nan
                row[f"{pfx}_comb_flux_err_mjy"] = np.nan
                row[f"{pfx}_ratio_flux_mjy"] = np.nan
                row[f"{pfx}_ratio_flux_err_mjy"] = np.nan

            # unWISE forced (W1/W2 only, NaN otherwise)
            row[f"{pfx}_unwise_flux_mjy"] = _to_float(_get(unwise_forced, survey, comp, "flux_mjy"))
            row[f"{pfx}_unwise_flux_err_mjy"] = _to_float(_get(unwise_forced, survey, comp, "flux_err_mjy"))

        rows.append(row)

    # build column order explicitly
    colnames = ["target_id", "component", "ra_deg", "dec_deg",
                "n_components", "separation_arcsec",
                "separation_policy", "target_type",
                "n_components_detected", "has_companion_not_measured"]
    for survey in surveys_ord:
        pfx = _prefix(survey)
        colnames += [
            f"{pfx}_flux_mjy", f"{pfx}_flux_err_mjy", f"{pfx}_blend_flag",
            f"{pfx}_comb_flux_mjy", f"{pfx}_comb_flux_err_mjy",
            f"{pfx}_ratio_flux_mjy", f"{pfx}_ratio_flux_err_mjy",
            f"{pfx}_unwise_flux_mjy", f"{pfx}_unwise_flux_err_mjy",
        ]

    tbl = Table(rows, names=colnames)
    return tbl


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def save_galaxy_table(table, out_path):
    """Save ``table`` to both ``out_path.csv`` and ``out_path.ecsv``.

    ``out_path`` should NOT carry an extension; the suffixes are appended here.
    """
    csv_path = f"{out_path}.csv"
    ecsv_path = f"{out_path}.ecsv"
    try:
        table.write(csv_path, format="csv", overwrite=True)
        table.write(ecsv_path, format="ascii.ecsv", overwrite=True)
    except Exception as exc:
        logger.error("failed to save galaxy table: %s", exc)
    return csv_path, ecsv_path
