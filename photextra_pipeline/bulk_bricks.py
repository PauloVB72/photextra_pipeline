"""Bulk Legacy Survey imaging via DR10 brick mosaics.

OPT-IN alternative to the per-target ``fits-cutout`` API in ``downloader.py``.
Instead of one rate-limited HTTP call per target per band, this module:

  1. resolves integer ``brickid`` values to brick name strings using the
     Legacy Surveys brick index table (``survey-bricks.fits.gz``, downloaded
     once and cached locally);
  2. bulk-downloads the per-brick, per-band coadd image mosaics
     (``legacysurvey-<brickname>-image-<band>.fits.fz``) from the NERSC file
     server -- one GET per brick x band, shared by every target in that
     brick, with the same retry/backoff policy as the cutout API path;
  3. extracts each target's cutout locally (Cutout2D against the brick WCS,
     no network) and writes it to the exact per-target cache path the
     existing pipeline uses (``{output_dir}/{id}/cache/Legacy_g.fits``...),
     so ``hostphot_downloader`` / ``Pipeline.run`` find the files as if the
     cutout API had produced them and need no changes at all.

Brick pixels are nanomaggies (AB zp 22.5), identical to what the cutout API
serves for layer ``ls-dr10`` at the native 0.262"/px -- same underlying
data, different transport (verified 2026-07-17: bulk vs API cutout for the
same target agree pixel-by-pixel except the API zero-fills the brick's NaN
no-coverage pixels).

Data URLs
---------
* Brick index (id -> name -> corners), whole-sky tessellation, ~13 MB:
  https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/survey-bricks.fits.gz
* South (DECam, dec <~ +32): DR10
  https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/coadd/<AAA>/<brick>/legacysurvey-<brick>-image-<band>.fits.fz
* North (BASS+MzLS, dec >~ +32): DR10 has no separate north release; the
  viewer's ls-dr10 layer serves DR9 north there, so we fall back to
  https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/north/coadd/...
  when the DR10-south URL 404s. (Fallback-on-404 rather than a dec cut:
  the south/north boundary is ragged and many bricks near dec ~32-34 exist
  in both; trying south first always prefers the deeper DECam data, which
  is also what the cutout API's ls-dr10 layer does.)

BRICKID is defined by the sky tessellation itself (same table since DR8),
so ids from any DR8+ tractor/zcat catalog resolve against this index.

Edge case: a target close to a brick boundary may need pixels outside its
brick. Bricks are ~0.25 deg with a small overlap margin; for the pipeline's
production cutout size (1 arcmin) this is rare. ``extract_cutout`` uses
Cutout2D(mode="partial", fill_value=nan) and logs a warning with the NaN
fraction instead of silently truncating. Full multi-brick mosaicking for
those targets is a documented TODO -- send them through the per-target API
path if the warning fires.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

# reuse the existing retry/backoff policy and cache layout -- single source
# of truth lives in downloader.py
from .downloader import (SURVEY_DEFAULTS, USER_AGENT, RETRY_ATTEMPTS,
                         _retry_sleep_seconds, _cache_path, DownloadError,
                         append_failed_target)

logger = logging.getLogger(__name__)

NERSC_BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey"
BRICKS_INDEX_URL = f"{NERSC_BASE}/dr10/survey-bricks.fits.gz"
# (release, hemisphere) tried in order; first hit wins (see module docstring)
BRICK_DATA_RELEASES = (("dr10", "south"), ("dr9", "north"))

DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser(os.environ.get("XDG_CACHE_HOME", "~/.cache")),
    "photextra_pipeline")


def _download_file_with_retry(url, dest, what, timeout=300):
    """Stream a URL to ``dest`` with the downloader.py retry/backoff policy.

    Writes to ``dest + '.part'`` and renames on success so an interrupted
    download never leaves a truncated file that would be mistaken for a
    cached brick. Returns dest. Raises DownloadError on exhausted retries;
    a plain 404 raises requests.HTTPError immediately (callers use it to
    fall back from south to north).
    """
    import requests

    headers = {"User-Agent": USER_AGENT}
    tmp = dest + ".part"
    last_exc = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with requests.get(url, timeout=timeout, headers=headers,
                              stream=True) as resp:
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    sleep = _retry_sleep_seconds(
                        attempt, resp.headers.get("Retry-After"))
                    last_exc = requests.exceptions.HTTPError(
                        f"{resp.status_code} for {url}", response=resp)
                    logger.warning("%s got HTTP %d; retry %d/%d in %.1fs",
                                   what, resp.status_code, attempt + 1,
                                   RETRY_ATTEMPTS, sleep)
                    import time as _t
                    _t.sleep(sleep)
                    continue
                resp.raise_for_status()  # 404 etc: propagate immediately
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            os.replace(tmp, dest)
            return dest
        except requests.exceptions.HTTPError:
            raise
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            sleep = _retry_sleep_seconds(attempt)
            logger.warning("%s network error (%s); retry %d/%d in %.1fs",
                           what, exc, attempt + 1, RETRY_ATTEMPTS, sleep)
            import time as _t
            _t.sleep(sleep)
    raise DownloadError(f"{what} failed after {RETRY_ATTEMPTS} attempts: "
                        f"{last_exc}")


def get_bricks_index(bricks_index_path=None, cache_dir=None):
    """Return the path to a local copy of ``survey-bricks.fits.gz``.

    Downloads it once (13 MB) into ``cache_dir`` (default
    ``~/.cache/photextra_pipeline/``) if not already present.
    """
    if bricks_index_path:
        if not os.path.exists(bricks_index_path):
            raise FileNotFoundError(bricks_index_path)
        return bricks_index_path
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    path = os.path.join(cache_dir, "survey-bricks.fits.gz")
    if not os.path.exists(path):
        logger.info("Downloading brick index %s -> %s",
                    BRICKS_INDEX_URL, path)
        _download_file_with_retry(BRICKS_INDEX_URL, path,
                                  "survey-bricks index")
    return path


_INDEX_CACHE = {}  # path -> dict of numpy columns (module-level memoization)


def _load_index(path):
    if path not in _INDEX_CACHE:
        with fits.open(path) as hdul:
            t = hdul[1].data
            _INDEX_CACHE[path] = {
                "brickid": np.asarray(t["BRICKID"], dtype=np.int64),
                "brickname": np.char.strip(
                    np.asarray(t["BRICKNAME"], dtype=str)),
                "ra1": np.asarray(t["RA1"], dtype=float),
                "ra2": np.asarray(t["RA2"], dtype=float),
                "dec1": np.asarray(t["DEC1"], dtype=float),
                "dec2": np.asarray(t["DEC2"], dtype=float),
            }
    return _INDEX_CACHE[path]


def resolve_brick_names(brickids, bricks_index_path=None, cache_dir=None):
    """Map integer BRICKIDs to brickname strings ('AAAA[pm]DDD').

    Returns {brickid: brickname}. Unknown / non-positive ids are silently
    omitted from the result (brickid 0 appears in some catalogs for
    secondary targets with no tractor match).
    """
    idx = _load_index(get_bricks_index(bricks_index_path, cache_dir))
    ids = np.unique(np.asarray(list(brickids), dtype=np.int64))
    ids = ids[ids > 0]
    pos = np.searchsorted(idx["brickid"], ids)  # index table is id-sorted
    out = {}
    for i, p in zip(ids, pos):
        if p < len(idx["brickid"]) and idx["brickid"][p] == i:
            out[int(i)] = str(idx["brickname"][p])
        else:  # index not sorted at this id? fall back to a full scan
            hit = np.nonzero(idx["brickid"] == i)[0]
            if hit.size:
                out[int(i)] = str(idx["brickname"][hit[0]])
            else:
                logger.warning("brickid %d not in brick index", i)
    return out


def find_brick_for_position(ra, dec, bricks_index_path=None, cache_dir=None):
    """Spatial fallback: brickname whose RA1/RA2/DEC1/DEC2 box contains
    (ra, dec). Returns brickname or None."""
    idx = _load_index(get_bricks_index(bricks_index_path, cache_dir))
    ra = ra % 360.0
    m = ((idx["dec1"] <= dec) & (dec < idx["dec2"]) &
         (idx["ra1"] <= ra) & (ra < idx["ra2"]))
    hit = np.nonzero(m)[0]
    if not hit.size:
        return None
    return str(idx["brickname"][hit[0]])


def brick_urls(brickname, band):
    """Candidate URLs for one brick image, in preference order."""
    aaa = brickname[:3]
    fname = f"legacysurvey-{brickname}-image-{band}.fits.fz"
    return [f"{NERSC_BASE}/{rel}/{hemi}/coadd/{aaa}/{brickname}/{fname}"
            for rel, hemi in BRICK_DATA_RELEASES]


def _brick_local_path(cache_dir, brickname, band):
    return os.path.join(cache_dir, "legacy_bricks", brickname,
                        f"legacysurvey-{brickname}-image-{band}.fits.fz")


def download_bricks(bricknames, bands, cache_dir=None, max_workers=3,
                    overwrite=False):
    """Bulk-download brick image mosaics (one file per brick x band).

    Tries DR10-south first, falls back to DR9-north on 404 (see module
    docstring). Skips files already in the cache. Returns
    {(brickname, band): local_path or None}; None means every URL failed
    (target outside both footprints, or persistent server error).
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    jobs = [(b, band) for b in dict.fromkeys(bricknames) for band in bands]

    def _one(brickname, band):
        dest = _brick_local_path(cache_dir, brickname, band)
        if os.path.exists(dest) and not overwrite:
            return dest
        import requests
        last = None
        for url in brick_urls(brickname, band):
            try:
                _download_file_with_retry(
                    url, dest, f"brick {brickname} {band}")
                logger.info("downloaded brick %s band %s", brickname, band)
                return dest
            except requests.exceptions.HTTPError as exc:
                last = exc
                code = getattr(exc.response, "status_code", None)
                if code == 404:
                    continue  # try the next release/hemisphere
                raise
        logger.error("brick %s band %s not found in any release (%s)",
                     brickname, band, last)
        return None

    out = {}
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futs = {ex.submit(_one, b, band): (b, band) for b, band in jobs}
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                out[key] = fut.result()
            except Exception as exc:
                logger.error("brick download %s failed: %s", key, exc)
                out[key] = None
    return out


def extract_cutout(brick_fits_path, ra, dec, size_arcmin, pixscale=0.262):
    """Cut a target's postage stamp out of a downloaded brick mosaic.

    Pure local operation (array slicing + brick WCS). ``pixscale`` only
    sets the requested pixel COUNT (size_arcmin * 60 / pixscale); the data
    keep the brick's native 0.262"/px grid -- identical to what the cutout
    API returns for pixscale=0.262. NaN no-coverage pixels are zeroed to
    match the API's behaviour. Returns {"data", "header", "wcs", "path"}
    (the shape hostphot_downloader produces), data float32.
    """
    from astropy.nddata import Cutout2D

    npix = int(np.ceil(size_arcmin * 60.0 / pixscale))
    with fits.open(brick_fits_path) as hdul:
        # .fits.fz: image lives in a CompImageHDU (first HDU with data)
        hdu = next(h for h in hdul if h.data is not None)
        wcs = WCS(hdu.header)
        xy = wcs.all_world2pix([[float(ra), float(dec)]], 0)[0]
        cut = Cutout2D(np.asarray(hdu.data, dtype=np.float32),
                       position=(float(xy[0]), float(xy[1])),
                       size=(npix, npix), wcs=wcs, mode="partial",
                       fill_value=np.nan)
        data = np.asarray(cut.data, dtype=np.float32)
        nan_frac = float(np.isnan(data).mean())
        if nan_frac > 0:
            logger.warning(
                "cutout at ra=%.5f dec=%.5f spills off brick %s "
                "(%.1f%% blank; TODO multi-brick mosaic -- consider the "
                "per-target API path for this target)",
                ra, dec, os.path.basename(brick_fits_path), 100 * nan_frac)
        data = np.nan_to_num(data, nan=0.0)  # cutout API zero-fills too

        header = hdu.header.copy()
        # strip tile-compression cards; overlay the cutout WCS
        for key in ("ZIMAGE", "ZTILE1", "ZTILE2", "ZCMPTYPE", "ZNAME1",
                    "ZVAL1", "ZNAME2", "ZVAL2", "ZSIMPLE", "ZBITPIX",
                    "ZNAXIS", "ZNAXIS1", "ZNAXIS2", "ZEXTEND", "XTENSION",
                    "PCOUNT", "GCOUNT", "TFIELDS", "CHECKSUM", "DATASUM"):
            header.pop(key, None)
        header.update(cut.wcs.to_header())
        header["NAXIS"] = 2
        header["NAXIS1"], header["NAXIS2"] = data.shape[1], data.shape[0]
        header["BRICKSRC"] = (os.path.basename(brick_fits_path),
                              "bulk brick this cutout was extracted from")
    return {"data": data, "header": header, "wcs": cut.wcs,
            "path": brick_fits_path}


def bulk_prefetch_legacy(targets, surveys, size_arcmin, output_dir,
                         brick_cache_dir=None, bricks_index_path=None,
                         max_workers=3, overwrite=False):
    """Populate the per-target Legacy image caches from bulk brick files.

    Drop-in alternative to the Legacy part of ``prefetch_images``: writes
    ``{output_dir}/{id}/cache/Legacy_<band>.fits`` (exactly
    ``downloader._cache_path``), so the fit stage / ``Pipeline.run`` load
    them as ordinary cached downloads.

    Parameters
    ----------
    targets : list of dict
        Each needs "id", "ra", "dec"; an integer "brickid" key is used if
        present (e.g. from a DESI/tractor catalog), otherwise the brick is
        found spatially from the index.
    surveys : list of str
        Legacy survey keys only ("Legacy_g", "Legacy_r", "Legacy_z").
    size_arcmin : float
        Cutout size, matching the pipeline's ``download_size``.
    output_dir : str
        Pipeline output dir; caches go to ``{output_dir}/{id}/cache``.
    brick_cache_dir : str
        Where brick mosaics + the brick index are stored (default
        ``~/.cache/photextra_pipeline``). Bricks are reused across runs.
    max_workers : int
        Concurrent brick downloads (NERSC file server; 3 is polite).

    Returns
    -------
    failures : list of dict -- same shape as ``prefetch_images``:
        [{"target_id", "failed_bands": {survey: reason}}, ...], also
        appended to ``{output_dir}/failed_targets.jsonl``
        (stage="prefetch_bulk").
    """
    surveys = [s for s in surveys if s.startswith("Legacy")]
    if not surveys:
        return []
    bad = [s for s in surveys if s not in SURVEY_DEFAULTS]
    if bad:
        raise ValueError(f"unknown Legacy surveys: {bad}")
    bands = [SURVEY_DEFAULTS[s]["legacy_band"] for s in surveys]
    brick_cache_dir = brick_cache_dir or DEFAULT_CACHE_DIR

    # 1. resolve each target to a brickname ------------------------------
    ids_present = [int(t["brickid"]) for t in targets
                   if int(t.get("brickid") or 0) > 0]
    id2name = resolve_brick_names(ids_present, bricks_index_path,
                                  brick_cache_dir) if ids_present else {}
    target_brick = {}
    failures_map = {}
    for t in targets:
        bid = int(t.get("brickid") or 0)
        name = id2name.get(bid)
        if name is None:
            name = find_brick_for_position(t["ra"], t["dec"],
                                           bricks_index_path,
                                           brick_cache_dir)
        if name is None:
            failures_map[str(t["id"])] = {
                s: "no brick contains this position" for s in surveys}
        else:
            target_brick[str(t["id"])] = name

    # 2. skip targets whose cache is already complete ---------------------
    def _cache_done(tid):
        cdir = os.path.join(output_dir, tid, "cache")
        return all(os.path.exists(_cache_path(cdir, s)) for s in surveys)

    todo = [t for t in targets if str(t["id"]) in target_brick
            and (overwrite or not _cache_done(str(t["id"])))]

    # 3. bulk-download the unique bricks actually needed ------------------
    needed = sorted({target_brick[str(t["id"])] for t in todo})
    logger.info("bulk prefetch: %d targets -> %d unique bricks x %d bands",
                len(todo), len(needed), len(bands))
    brick_files = download_bricks(needed, bands, brick_cache_dir,
                                  max_workers=max_workers,
                                  overwrite=False)

    # 4. extract + cache each target's cutouts (local, no network) --------
    for t in todo:
        tid = str(t["id"])
        brick = target_brick[tid]
        cache_dir = os.path.join(output_dir, tid, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        failed = {}
        for survey, band in zip(surveys, bands):
            path = _cache_path(cache_dir, survey)
            if os.path.exists(path) and not overwrite:
                continue
            src = brick_files.get((brick, band))
            if src is None:
                failed[survey] = f"brick {brick} band {band} unavailable"
                continue
            try:
                cut = extract_cutout(src, t["ra"], t["dec"], size_arcmin,
                                     SURVEY_DEFAULTS[survey]["pixscale"])
                fits.PrimaryHDU(data=cut["data"],
                                header=cut["header"]).writeto(
                    path, overwrite=True)
            except Exception as exc:
                logger.error("cutout %s %s failed: %s", tid, survey, exc)
                failed[survey] = str(exc)
        if failed:
            failures_map[tid] = {**failures_map.get(tid, {}), **failed}

    failures = []
    for tid, failed in failures_map.items():
        logger.error("bulk prefetch: target %s failed bands: %s",
                     tid, sorted(failed))
        failures.append({"target_id": tid, "failed_bands": failed})
        append_failed_target(output_dir, tid,
                             reason=f"bulk prefetch failure: "
                                    f"{sorted(failed)}",
                             stage="prefetch_bulk", details=failed)
    return failures
