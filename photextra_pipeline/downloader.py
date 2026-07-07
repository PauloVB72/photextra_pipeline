"""Image download for the 9 supported bands.

Tries hostphot first; falls back to astroquery SkyView. All physical survey
properties live in SURVEY_DEFAULTS so the rest of the pipeline never hardcodes
them.
"""

import os
import time
import random
import logging

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

# API etiquette: identify the pipeline to the cutout services and keep a
# small pause between consecutive requests from the same process.
USER_AGENT = "photextra_pipeline/0.1 (CHANCES galaxy cluster survey; academic use)"
REQUEST_DELAY_S = 0.5

# retry/backoff policy for the Legacy Survey cutout API (HTTP 429 / 5xx)
RETRY_ATTEMPTS = 5
RETRY_BASE_S = 2.0
RETRY_MAX_SLEEP_S = 60.0


# lambda_eff in micron, psf_fwhm and pixscale in arcsec, zp in AB mag.
# flux_factor converts the natural flux unit (Jy) to the pipeline unit (mJy).
SURVEY_DEFAULTS = {
    "GALEX_FUV": {"lambda_eff": 0.153, "psf_fwhm": 4.5,  "pixscale": 1.5,   "zp": 18.82, "flux_factor": 1000.0,
                  "skyview": "GALEX Far UV"},
    "GALEX_NUV": {"lambda_eff": 0.231, "psf_fwhm": 6.0,  "pixscale": 1.5,   "zp": 20.08, "flux_factor": 1000.0,
                  "skyview": "GALEX Near UV"},
    "Legacy_g":  {"lambda_eff": 0.472, "psf_fwhm": 1.3,  "pixscale": 0.262, "zp": 22.5,  "flux_factor": 1000.0,
                  "skyview": "DECaLS DR7 g", "legacy_band": "g"},
    "Legacy_r":  {"lambda_eff": 0.641, "psf_fwhm": 1.3,  "pixscale": 0.262, "zp": 22.5,  "flux_factor": 1000.0,
                  "skyview": "DECaLS DR7 r", "legacy_band": "r"},
    "Legacy_z":  {"lambda_eff": 0.906, "psf_fwhm": 1.3,  "pixscale": 0.262, "zp": 22.5,  "flux_factor": 1000.0,
                  "skyview": "DECaLS DR7 z", "legacy_band": "z"},
    "WISE_W1":   {"lambda_eff": 3.4,   "psf_fwhm": 6.1,  "pixscale": 1.375, "zp": 20.5,  "flux_factor": 1000.0,
                  "skyview": "WISE 3.4"},
    "WISE_W2":   {"lambda_eff": 4.6,   "psf_fwhm": 6.4,  "pixscale": 1.375, "zp": 19.8,  "flux_factor": 1000.0,
                  "skyview": "WISE 4.6"},
    "WISE_W3":   {"lambda_eff": 12.0,  "psf_fwhm": 6.5,  "pixscale": 1.375, "zp": 17.8,  "flux_factor": 1000.0,
                  "skyview": "WISE 12"},
    "WISE_W4":   {"lambda_eff": 22.0,  "psf_fwhm": 12.0, "pixscale": 1.375, "zp": 12.6,  "flux_factor": 1000.0,
                  "skyview": "WISE 22"},
}


class DownloadError(RuntimeError):
    pass


def _cache_path(cache_dir, survey):
    return os.path.join(cache_dir, f"{survey}.fits")


def _download_skyview(ra, dec, survey, size_arcmin, pixscale):
    """Query SkyView via astroquery. Returns an astropy HDU."""
    from astroquery.skyview import SkyView
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    meta = SURVEY_DEFAULTS[survey]
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    npix = int(np.ceil(size_arcmin * 60.0 / pixscale))

    # NOTE: astroquery sends the SkyView "size" as radius*2, so passing
    # radius=size_arcmin produced images with DOUBLE the intended field of
    # view and half the resolution (e.g. WISE at 2.727"/px instead of the
    # native 1.375"/px the ZP is calibrated for). Using width/height makes
    # the image exactly size_arcmin wide with npix pixels, i.e. the pixel
    # scale in SURVEY_DEFAULTS — matching Legacy cutouts and the ZP grid.
    hdulists = SkyView.get_images(
        position=coord,
        survey=[meta["skyview"]],
        width=size_arcmin * u.arcmin,
        height=size_arcmin * u.arcmin,
        pixels=str(npix),
    )
    if not hdulists:
        raise DownloadError(f"SkyView returned no image for {survey}")
    return hdulists[0][0]


def _retry_sleep_seconds(attempt, retry_after=None):
    """Backoff for the given (0-based) attempt: honor Retry-After, else
    exponential with jitter (base 2s -> 2, 4, 8, 16... plus 0-1s jitter)."""
    if retry_after is not None:
        try:
            return min(float(retry_after), RETRY_MAX_SLEEP_S)
        except (TypeError, ValueError):
            pass
    return min(RETRY_BASE_S * (2 ** attempt) + random.uniform(0, 1),
               RETRY_MAX_SLEEP_S)


def _download_legacy(ra, dec, survey, size_arcmin, pixscale):
    """Legacy Survey cutout service, with retry/backoff on 429/5xx.

    Returns an astropy HDU. Raises DownloadError after exhausting
    RETRY_ATTEMPTS on rate-limit/server errors.
    """
    import requests
    import io

    meta = SURVEY_DEFAULTS[survey]
    npix = int(np.ceil(size_arcmin * 60.0 / pixscale))
    npix = min(npix, 3000)
    url = (
        "https://www.legacysurvey.org/viewer/fits-cutout"
        f"?ra={ra}&dec={dec}&layer=ls-dr10&pixscale={pixscale}"
        f"&bands={meta['legacy_band']}&size={npix}"
    )
    headers = {"User-Agent": USER_AGENT}

    last_exc = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, timeout=120, headers=headers)
        except requests.exceptions.RequestException as exc:
            # transient network error: retry with the same backoff
            last_exc = exc
            sleep = _retry_sleep_seconds(attempt)
            logger.warning("Legacy cutout %s network error (%s); retry %d/%d "
                           "in %.1fs", survey, exc, attempt + 1,
                           RETRY_ATTEMPTS, sleep)
            time.sleep(sleep)
            continue

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            last_exc = requests.exceptions.HTTPError(
                f"{resp.status_code} for {url}", response=resp)
            sleep = _retry_sleep_seconds(
                attempt, resp.headers.get("Retry-After"))
            logger.warning("Legacy cutout %s got HTTP %d; retry %d/%d in "
                           "%.1fs", survey, resp.status_code, attempt + 1,
                           RETRY_ATTEMPTS, sleep)
            time.sleep(sleep)
            continue

        resp.raise_for_status()
        hdul = fits.open(io.BytesIO(resp.content))
        return hdul[0]

    raise DownloadError(
        f"Legacy cutout for {survey} failed after {RETRY_ATTEMPTS} attempts "
        f"(rate limit / server error): {last_exc}")


def hostphot_downloader(ra, dec, surveys, size_arcmin, cache_dir, overwrite=False):
    """Download all requested surveys into cache_dir.

    Returns dict survey -> {"data": ndarray, "header": Header, "wcs": WCS,
    "path": str}. Bands that fail to download are returned with data=None and
    an "error" string describing the failure so the pipeline can distinguish
    an avoidable download failure (rate limit) from genuine absence.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = {}
    did_network_call = False

    for survey in surveys:
        if survey not in SURVEY_DEFAULTS:
            logger.warning("Unknown survey %s, skipping", survey)
            continue

        meta = SURVEY_DEFAULTS[survey]
        path = _cache_path(cache_dir, survey)

        if os.path.exists(path) and not overwrite:
            try:
                with fits.open(path) as hdul:
                    hdu = hdul[0]
                    out[survey] = _pack(hdu, path)
                logger.info("Loaded cached %s", survey)
                continue
            except Exception:
                logger.warning("Cached %s unreadable, re-downloading", survey)

        # small pause between consecutive network requests (API etiquette)
        if did_network_call and REQUEST_DELAY_S > 0:
            time.sleep(REQUEST_DELAY_S)
        did_network_call = True

        hdu = None
        try:
            if survey.startswith("Legacy"):
                try:
                    hdu = _download_legacy(ra, dec, survey, size_arcmin, meta["pixscale"])
                except Exception as exc:
                    logger.warning("Legacy cutout failed for %s (%s), trying SkyView", survey, exc)
                    hdu = _download_skyview(ra, dec, survey, size_arcmin, meta["pixscale"])
            else:
                hdu = _download_skyview(ra, dec, survey, size_arcmin, meta["pixscale"])
        except Exception as exc:
            logger.error("Download failed for %s: %s", survey, exc)
            out[survey] = {"data": None, "header": None, "wcs": None,
                           "path": None, "error": str(exc)}
            continue

        try:
            hdu.writeto(path, overwrite=True)
        except Exception as exc:
            logger.warning("Could not cache %s: %s", survey, exc)

        out[survey] = _pack(hdu, path)

    return out


def prefetch_images(targets, surveys, size_arcmin, output_dir,
                    max_workers=2, overwrite=False):
    """Batch download coordinator: pre-populate every target's image cache.

    Decouples the rate-limited download stage from the CPU-bound parallel
    fit stage. Downloads run at LOW concurrency (default 2 threads) so the
    Legacy Survey cutout API is never hit by N fit workers at once; the fit
    stage then finds all FITS files already cached under
    ``{output_dir}/{target_id}/cache/`` (the exact layout ``Pipeline.run``
    uses) and does no network I/O.

    Naturally resumable: ``hostphot_downloader`` skips any band whose cached
    FITS file already exists, so re-running after an interruption only
    fetches what is missing.

    Parameters
    ----------
    targets : list of dict
        Each needs "id", "ra", "dec".
    surveys : list of str
        Survey names (keys of SURVEY_DEFAULTS).
    size_arcmin : float
        Cutout size, matching the pipeline's ``download_size``.
    output_dir : str
        The pipeline ``output_dir``; caches go to ``{output_dir}/{id}/cache``.
    max_workers : int
        Concurrent download threads (keep at 1-3; this is a shared external
        rate limit, not a CPU resource).

    Returns
    -------
    failures : list of dict
        One entry per target that has at least one failed band:
        {"target_id", "failed_bands": {survey: reason}}. Also appended to
        ``{output_dir}/failed_targets.jsonl`` (stage="prefetch").
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(tgt):
        cache_dir = os.path.join(output_dir, str(tgt["id"]), "cache")
        images = hostphot_downloader(tgt["ra"], tgt["dec"], surveys,
                                     size_arcmin, cache_dir,
                                     overwrite=overwrite)
        failed = {s: e.get("error", "unknown") for s, e in images.items()
                  if e.get("data") is None}
        return tgt["id"], failed

    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futs = {ex.submit(_fetch_one, t): t for t in targets}
        for fut in as_completed(futs):
            tgt = futs[fut]
            try:
                tid, failed = fut.result()
            except Exception as exc:
                tid, failed = str(tgt["id"]), {"__all__": str(exc)}
            if failed:
                logger.error("prefetch: target %s failed bands: %s",
                             tid, sorted(failed))
                failures.append({"target_id": tid, "failed_bands": failed})
                append_failed_target(output_dir, tid,
                                     reason=f"prefetch download failure: "
                                            f"{sorted(failed)}",
                                     stage="prefetch", details=failed)
            else:
                logger.info("prefetch: target %s complete (%d bands)",
                            tid, len(surveys))
    return failures


def append_failed_target(output_dir, target_id, reason, stage="pipeline",
                         details=None):
    """Append one failure record to {output_dir}/failed_targets.jsonl.

    JSON-lines with O_APPEND writes: safe under parallel worker processes
    (a read-modify-write .json would race). Each line:
    {"target_id", "stage", "reason", "details", "timestamp"}.
    """
    import json
    from datetime import datetime, timezone

    os.makedirs(output_dir, exist_ok=True)
    rec = {"target_id": str(target_id), "stage": stage, "reason": reason,
           "details": details,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    path = os.path.join(output_dir, "failed_targets.jsonl")
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as exc:  # never let bookkeeping kill the pipeline
        logger.error("could not append to %s: %s", path, exc)
    return path


def _pack(hdu, path):
    data = np.asarray(hdu.data, dtype=float)
    if data.ndim == 3:
        data = data[0]
    header = hdu.header
    try:
        wcs = WCS(header).celestial
    except Exception:
        wcs = None
    return {"data": data, "header": header, "wcs": wcs, "path": path}
