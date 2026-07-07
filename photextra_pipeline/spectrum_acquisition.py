"""Resolve and download a spectrum (DESI or SDSS) from a target dict alone.

The photometric side of the pipeline only needs coordinates; the spectral
side historically required the caller to pre-supply ``spectrum_path``. This
module closes the gap.

CHANCES targets carry a DESI targetid as their ``id``, so :func:`ensure_spectrum`
tries DESI first (:func:`resolve_desi_spectrum`: SPARCL, falling back to
NOIRLab Astro Data Lab SQL + direct HTTPS if SPARCL is unreachable), then
falls back to a coordinate-based SDSS cross-match (:func:`resolve_sdss_spectrum`)
for targets without a DESI match. Both cache their result under the target's
``spectroscopy/`` directory so repeated runs never re-download.

All failures are soft: every public function returns ``None`` (never raises)
so a missing/unreachable spectrum can never crash a batch run.
"""

import glob
import logging
import os
import random
import time

import numpy as np

logger = logging.getLogger(__name__)

# match radius for the spectroscopic cross-match (arcsec); SDSS fibers are
# 2-3" wide, so 3" comfortably covers astrometric offsets without picking
# up unrelated neighbours.
DEFAULT_RADIUS_ARCSEC = 3.0

# DESI healpix coadd redux tree used by the Data Lab HTTPS fallback.
_DESI_DR1_HEALPIX_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"

# retry/backoff policy for transient spectrum-service failures. SPARCL's
# client hard-caps its connect timeout at 3.1 s (MAX_CONNECT_TIMEOUT inside
# sparclclient; the connect_timeout=30 we pass is silently min()'d down), so
# under parallel batch load a TLS handshake occasionally exceeds it and one
# cheap retry (seconds) avoids the Data Lab coadd fallback (~15 min/target).
RETRY_ATTEMPTS = 4
RETRY_BASE_S = 2.0
RETRY_MAX_SLEEP_S = 30.0


def _is_transient(exc):
    """True when ``exc`` looks like a retryable network/service hiccup
    (timeouts, connection resets, 5xx/429) rather than a genuine
    "no data for this object" or a programming/setup error."""
    if isinstance(exc, (ImportError, KeyError, ValueError, TypeError,
                        AttributeError, IndexError)):
        return False
    try:
        import requests
        if isinstance(exc, (requests.exceptions.Timeout,
                            requests.exceptions.ConnectionError)):
            return True
    except ImportError:
        pass
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    msg = f"{type(exc).__name__}: {exc}".lower()
    keys = ("timed out", "timeout", "connection", "temporarily unavailable",
            "bad gateway", "service unavailable", "gateway time",
            "too many requests", "http 429", "http 500", "http 502",
            "http 503", "http 504", "status 429", "status 500",
            "status 502", "status 503", "status 504", "remote end closed",
            "reset by peer", "broken pipe")
    return any(k in msg for k in keys)


def _call_with_retries(fn, what, attempts=RETRY_ATTEMPTS,
                       base_s=RETRY_BASE_S, max_sleep_s=RETRY_MAX_SLEEP_S):
    """Run ``fn()`` retrying transient failures with exponential backoff +
    jitter (base 2s -> ~2, 4, 8 s). Non-transient exceptions (and the last
    transient one) propagate; a ``None``/empty return is NOT retried, so
    genuine "no spectrum for this object" results still fail fast."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not _is_transient(exc) or attempt == attempts - 1:
                raise
            sleep = min(base_s * (2 ** attempt) + random.uniform(0, 1),
                        max_sleep_s)
            logger.warning("%s: transient failure (%s); retry %d/%d in %.1fs",
                           what, exc, attempt + 1, attempts - 1, sleep)
            time.sleep(sleep)


def _cached_spectrum(spec_dir):
    """Return a previously downloaded spec-*.fits in ``spec_dir``, if any."""
    hits = sorted(glob.glob(os.path.join(spec_dir, "spec-*.fits")))
    for h in hits:
        if os.path.getsize(h) > 0:
            return h
    return None


def _best_match(results, z=None):
    """Pick the best row from an astroquery SDSS spectro result table.

    Primary criterion: smallest positional separation is implicit in the
    small search radius, so when several rows survive we prefer the one whose
    spectroscopic redshift is closest to the target's known ``z`` (fibers of
    superposed/plate-duplicate objects), falling back to the first row.
    """
    if len(results) == 1 or z is None or "z" not in results.colnames:
        return results[0]
    zcol = np.asarray(results["z"], dtype=float)
    dz = np.abs(zcol - float(z))
    dz[~np.isfinite(dz)] = np.inf
    return results[int(np.argmin(dz))]


def resolve_sdss_spectrum(ra, dec, z=None, output_dir=".",
                          radius_arcsec=DEFAULT_RADIUS_ARCSEC):
    """Resolve (ra, dec[, z]) -> local SDSS spectrum FITS path, or ``None``.

    1. If a ``spec-*.fits`` already sits in ``output_dir``, reuse it.
    2. Otherwise query SDSS (spectro=True) within ``radius_arcsec``,
       pick the best match (closest redshift when ``z`` given), and
       download plate/mjd/fiberid via ``xpectrafit.io.downloader.download_sdss``
       into ``output_dir``.
    """
    os.makedirs(output_dir, exist_ok=True)

    cached = _cached_spectrum(output_dir)
    if cached is not None:
        logger.info("using cached SDSS spectrum: %s", cached)
        return cached

    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.sdss import SDSS
    except Exception as exc:
        logger.error("astroquery/astropy unavailable for SDSS resolution: %s", exc)
        return None

    try:
        coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg)
        results = SDSS.query_region(coord, radius=radius_arcsec * u.arcsec,
                                    spectro=True)
    except Exception as exc:
        logger.error("SDSS.query_region failed at (%.5f, %.5f): %s",
                     float(ra), float(dec), exc)
        return None

    if results is None or len(results) == 0:
        logger.warning("no SDSS spectrum within %.1f\" of (%.5f, %.5f)",
                       radius_arcsec, float(ra), float(dec))
        return None

    row = _best_match(results, z=z)
    try:
        plate = int(row["plate"])
        mjd = int(row["mjd"])
        fiberid = int(row["fiberID"])
    except Exception as exc:
        logger.error("could not read plate/mjd/fiberID from SDSS match: %s", exc)
        return None

    logger.info("SDSS match: plate=%d mjd=%d fiber=%d (z_spec=%s)",
                plate, mjd, fiberid,
                row["z"] if "z" in results.colnames else "n/a")

    try:
        from xpectrafit.io.downloader import download_sdss
        path = download_sdss(plate, mjd, fiberid, output_dir=output_dir)
    except Exception as exc:
        logger.error("SDSS spectrum download failed (plate=%d mjd=%d fiber=%d): %s",
                     plate, mjd, fiberid, exc)
        return None

    if path is None or not os.path.exists(path):
        logger.error("SDSS download returned no file (plate=%d mjd=%d fiber=%d)",
                     plate, mjd, fiberid)
        return None
    return path


def _cached_desi_npz(spec_dir, targetid):
    path = os.path.join(spec_dir, f"desi_{targetid}.npz")
    return path if os.path.exists(path) and os.path.getsize(path) > 0 else None


def _save_desi_npz(spec_dir, targetid, wave, flux, ivar, z):
    os.makedirs(spec_dir, exist_ok=True)
    path = os.path.join(spec_dir, f"desi_{targetid}.npz")
    np.savez(path, wave=np.asarray(wave, float), flux=np.asarray(flux, float),
              ivar=np.asarray(ivar, float), z=float(z) if z is not None else np.nan)
    return path


def _desi_via_sparcl(targetid, dataset="DESI-DR1"):
    """Primary DESI path: SPARCL. Returns (wave, flux, ivar, z) or None."""
    from xpectrafit.io.downloader import download_desi_sparcl
    recs = _call_with_retries(
        lambda: download_desi_sparcl(targetids=[int(targetid)], dataset=dataset),
        f"SPARCL fetch (targetid={targetid})")
    if not recs:
        return None
    r = recs[0]
    return r["wave"], r["flux"], r["ivar"], r.get("z")


def _desi_via_datalab(targetid, table="desi_dr1.zpix"):
    """Fallback DESI path when SPARCL is unreachable: resolve the target's
    (survey, program, healpix) via a NOIRLab Astro Data Lab SQL query, then
    read the matching row straight out of the public healpix coadd FITS file
    over HTTPS (range reads via fsspec, no local staging of the multi-GB
    file). Returns (wave, flux, ivar, z) or None. No credentials required.
    """
    from dl import queryClient as qc
    from astropy.io import fits

    q = (f"SELECT targetid, survey, program, healpix, z FROM {table} "
        f"WHERE targetid = {int(targetid)} AND main_primary")
    csv = qc.query(sql=q, fmt="csv")
    lines = csv.strip().splitlines()
    if len(lines) < 2:
        return None
    _, survey, program, healpix, z = lines[1].split(",")
    healpix = int(healpix)

    url = (f"{_DESI_DR1_HEALPIX_BASE}/{survey}/{program}/{healpix // 100}/"
          f"{healpix}/coadd-{survey}-{program}-{healpix}.fits")
    with fits.open(url, use_fsspec=True,
                  fsspec_kwargs={"block_size": 4 * 1024 * 1024}) as h:
        fmap_tid = np.asarray(h["FIBERMAP"].data["TARGETID"])
        matches = np.where(fmap_tid == int(targetid))[0]
        if len(matches) == 0:
            return None
        row = int(matches[0])
        names = [hd.name for hd in h]
        ws, fs, iv = [], [], []
        for arm in ("B", "R", "Z"):
            if f"{arm}_WAVELENGTH" not in names:
                continue
            ws.append(np.asarray(h[f"{arm}_WAVELENGTH"].data, float))
            fl = np.asarray(h[f"{arm}_FLUX"].section[row], float)
            iv.append(np.asarray(h[f"{arm}_IVAR"].section[row], float)
                      if f"{arm}_IVAR" in names else np.ones_like(fl))
            fs.append(fl)
        wave = np.concatenate(ws)
        flux = np.concatenate(fs)
        ivar = np.concatenate(iv)
        order = np.argsort(wave)
    return wave[order], flux[order], ivar[order], float(z)


def resolve_desi_spectrum(targetid, z=None, output_dir="."):
    """Resolve a DESI targetid -> local cached spectrum (``.npz``), or ``None``.

    Tries SPARCL first (fast, ~5s/spectrum once batched); on any failure
    (including the connection flakiness this pipeline previously hit) falls
    back to the NOIRLab Astro Data Lab SQL + direct-HTTPS path, which is
    slower per spectrum but does not depend on the SPARCL service at all.
    """
    if targetid is None:
        return None
    os.makedirs(output_dir, exist_ok=True)

    cached = _cached_desi_npz(output_dir, targetid)
    if cached is not None:
        logger.info("using cached DESI spectrum: %s", cached)
        return cached

    got, source = None, None
    try:
        got = _desi_via_sparcl(targetid)
        source = "sparcl"
    except Exception as exc:
        logger.warning("SPARCL failed for targetid=%s (%s); falling back "
                       "to Data Lab", targetid, exc)

    if got is None:
        try:
            got = _desi_via_datalab(targetid)
            source = "datalab"
        except Exception as exc:
            logger.error("Data Lab fallback also failed for targetid=%s: %s",
                        targetid, exc)

    if got is None:
        logger.warning("no DESI spectrum found for targetid=%s", targetid)
        return None

    wave, flux, ivar, z_ret = got
    logger.info("DESI spectrum for targetid=%s resolved via %s",
               targetid, source)
    return _save_desi_npz(output_dir, targetid, wave, flux, ivar,
                          z_ret if z_ret is not None else z)


def _infer_spectrum_source(path):
    """Best-effort spectrum source from a file path: 'desi' | 'sdss' | None.

    DESI spectra are cached by this module as ``desi_{targetid}.npz``; SDSS
    ones as ``spec-*.fits``. Used for pre-supplied ``spectrum_path`` values.
    """
    if not path:
        return None
    name = os.path.basename(path).lower()
    if name.endswith(".npz") or name.startswith("desi"):
        return "desi"
    if name.startswith("spec-") or name.endswith((".fits", ".fit")):
        return "sdss"
    return None


def ensure_spectrum(target, target_dir):
    """Make sure ``target['spectrum_path']`` points at a real file.

    If it is already present and exists, do nothing. Otherwise:

    1. Try DESI by targetid (``target['id']``, since CHANCES targets carry
       DESI targetids) via :func:`resolve_desi_spectrum` (SPARCL -> Data Lab).
    2. Fall back to a coordinate-based SDSS cross-match via
       :func:`resolve_sdss_spectrum`, for targets without a DESI match.

    Caches under ``{target_dir}/spectroscopy/`` and writes the resolved path
    back into the target dict, together with ``target['spectrum_source']``
    (``"desi"`` or ``"sdss"``) so downstream stages can pick the matching
    fiber aperture. Returns the path or ``None`` (never raises).
    """
    path = target.get("spectrum_path")
    if path is not None and os.path.exists(path):
        if not target.get("spectrum_source"):
            target["spectrum_source"] = _infer_spectrum_source(path)
        return path
    if path is not None:
        logger.warning("spectrum_path %r for %s does not exist; trying to "
                       "resolve it", path, target.get("id"))

    spec_dir = os.path.join(target_dir, "spectroscopy")

    targetid = target.get("targetid", target.get("id"))
    if targetid is not None:
        try:
            targetid_int = int(targetid)
        except (TypeError, ValueError):
            targetid_int = None
        if targetid_int is not None:
            resolved = resolve_desi_spectrum(targetid_int, z=target.get("z"),
                                             output_dir=spec_dir)
            if resolved is not None:
                target["spectrum_path"] = resolved
                target["spectrum_source"] = "desi"
                return resolved

    if target.get("ra") is None or target.get("dec") is None:
        logger.error("target %s lacks ra/dec; cannot resolve spectrum",
                     target.get("id"))
        return None

    resolved = resolve_sdss_spectrum(target["ra"], target["dec"],
                                     z=target.get("z"), output_dir=spec_dir)
    if resolved is not None:
        target["spectrum_path"] = resolved
        target["spectrum_source"] = "sdss"
    return resolved
