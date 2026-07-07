"""Adapter to hand photextra photometry off to XpectraFit.

Optional integration: if XpectraFit is installed we can feed the per-component
SED as photometric constraints. Degrades to a no-op when unavailable.
"""

import logging

logger = logging.getLogger(__name__)


def sed_to_xpectrafit(comp_seds, component="total"):
    """Convert a component SED (list of point dicts) to XpectraFit photometry.

    Returns a list of (lambda_micron, flux_mjy, flux_err_mjy) tuples; XpectraFit
    consumes these as broadband constraints alongside spectra.
    """
    points = comp_seds.get(component, [])
    out = []
    for p in points:
        lam = p.get("lambda_eff")
        flux = p.get("flux_mjy")
        err = p.get("flux_err_mjy")
        if lam is None or flux is None:
            continue
        out.append((lam, flux, err))
    return out


def run_xpectrafit(comp_seds, component="total", **kwargs):
    """Run XpectraFit on a component SED if the package is importable."""
    try:
        import xpectrafit  # noqa: F401
    except Exception:
        logger.info("XpectraFit not installed; skipping spectral fit")
        return None

    phot = sed_to_xpectrafit(comp_seds, component=component)
    try:
        from xpectrafit import fit_photometry
        return fit_photometry(phot, **kwargs)
    except Exception as exc:
        logger.warning("XpectraFit integration unavailable: %s", exc)
        return None
