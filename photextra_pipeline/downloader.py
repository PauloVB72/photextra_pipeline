"""Image download for the supported bands.

Tries hostphot first; falls back to astroquery SkyView. All physical survey
properties live in SURVEY_DEFAULTS so the rest of the pipeline never hardcodes
them.

Naming convention for survey keys: ``<Survey>_<band>`` where <Survey> is a
CamelCase survey tag with no underscores (GALEX, Legacy, WISE, SDSS, PS1,
TwoMASS, Spitzer) and <band> is the survey's own band name (FUV, g, W1, u,
J, Ks, IRAC1...). The dispatcher in ``hostphot_downloader`` routes on the
prefix before the first underscore.

Download backends per survey family:
  - GALEX, WISE, SDSS : astroquery SkyView (``_download_skyview``)
  - Legacy            : legacysurvey.org fits-cutout (SkyView fallback)
  - PS1               : ps1images.stsci.edu ps1filenames.py + fitscut.cgi
  - TwoMASS           : IRSA IBE (per-image MAGZP normalised, see
                        ``_download_2mass``); SkyView is NOT used because its
                        2MASS tiles keep the per-atlas-image zero point
                        (measured 0.5 mag field-to-field scatter).
  - Spitzer (SEIP)    : IRSA SIA v2 + lazy range-request cutout of the
                        public SEIP mosaics (MJy/sr); sparse sky coverage,
                        expect "no coverage" for most positions.

Surveys with no public blind-cutout API (HST, HSC, S-PLUS, J-PLUS) are NOT in
SURVEY_DEFAULTS; see ``survey_coverage.py`` for footprint/availability checks.
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
    # WISE Atlas MAGZP (Vega: 20.5/19.5/18.0/13.0) converted to AB via the
    # standard Vega->AB offsets (Jarrett et al. 2011: +2.699/+3.339/+5.174/+6.620)
    # so counts_to_mjy's fixed AB zero-flux (3631 Jy) applies uniformly.
    "WISE_W1":   {"lambda_eff": 3.4,   "psf_fwhm": 6.1,  "pixscale": 1.375, "zp": 23.199, "flux_factor": 1000.0,
                  "skyview": "WISE 3.4"},
    "WISE_W2":   {"lambda_eff": 4.6,   "psf_fwhm": 6.4,  "pixscale": 1.375, "zp": 22.839, "flux_factor": 1000.0,
                  "skyview": "WISE 4.6"},
    "WISE_W3":   {"lambda_eff": 12.0,  "psf_fwhm": 6.5,  "pixscale": 1.375, "zp": 23.174, "flux_factor": 1000.0,
                  "skyview": "WISE 12"},
    "WISE_W4":   {"lambda_eff": 22.0,  "psf_fwhm": 12.0, "pixscale": 1.375, "zp": 19.620, "flux_factor": 1000.0,
                  "skyview": "WISE 22"},
    # --- SDSS ugriz via SkyView ---------------------------------------------
    # SkyView serves SDSS calibrated frame files, whose pixels are nanomaggies
    # (1 nmgy = 3.631e-6 Jy), i.e. AB zero point exactly 22.5 (SDSS DR8+ frame
    # calibration; the SDSS photometric system is AB by design, Fukugita et
    # al. 1996 / Padmanabhan et al. 2008). Verified empirically 2026-07-16:
    # a star with catalog rmag(AB)=14.72 gives an implied zp of 22.50 in a
    # SkyView SDSSr cutout. (Known small AB offsets, u ~ -0.04, z ~ +0.02 mag,
    # are ignored — same order as the calibration scatter.)
    # lambda_eff from Doi et al. 2010; pixscale 0.396"/px native SDSS.
    "SDSS_u":    {"lambda_eff": 0.3551, "psf_fwhm": 1.3, "pixscale": 0.396, "zp": 22.5, "flux_factor": 1000.0,
                  "skyview": "SDSSu"},
    "SDSS_g":    {"lambda_eff": 0.4686, "psf_fwhm": 1.3, "pixscale": 0.396, "zp": 22.5, "flux_factor": 1000.0,
                  "skyview": "SDSSg"},
    "SDSS_r":    {"lambda_eff": 0.6166, "psf_fwhm": 1.3, "pixscale": 0.396, "zp": 22.5, "flux_factor": 1000.0,
                  "skyview": "SDSSr"},
    "SDSS_i":    {"lambda_eff": 0.7480, "psf_fwhm": 1.3, "pixscale": 0.396, "zp": 22.5, "flux_factor": 1000.0,
                  "skyview": "SDSSi"},
    "SDSS_z":    {"lambda_eff": 0.8932, "psf_fwhm": 1.3, "pixscale": 0.396, "zp": 22.5, "flux_factor": 1000.0,
                  "skyview": "SDSSz"},
    # --- Pan-STARRS1 grizy via ps1images.stsci.edu ---------------------------
    # PS1 stack pixels are counts with zp = 25.0 + 2.5 log10(EXPTIME) in AB
    # (FPA.ZP=25.0 header keyword; PS1 photometry is AB-calibrated, Tonry et
    # al. 2012; stack zero point convention Waters et al. 2020).
    # _download_ps1 divides the image by EXPTIME so a FIXED zp of 25.0 AB
    # applies to every cutout. lambda_eff from Tonry et al. 2012.
    "PS1_g":     {"lambda_eff": 0.4866, "psf_fwhm": 1.3, "pixscale": 0.25, "zp": 25.0, "flux_factor": 1000.0,
                  "ps1_band": "g"},
    "PS1_r":     {"lambda_eff": 0.6215, "psf_fwhm": 1.2, "pixscale": 0.25, "zp": 25.0, "flux_factor": 1000.0,
                  "ps1_band": "r"},
    "PS1_i":     {"lambda_eff": 0.7545, "psf_fwhm": 1.1, "pixscale": 0.25, "zp": 25.0, "flux_factor": 1000.0,
                  "ps1_band": "i"},
    "PS1_z":     {"lambda_eff": 0.8679, "psf_fwhm": 1.1, "pixscale": 0.25, "zp": 25.0, "flux_factor": 1000.0,
                  "ps1_band": "z"},
    "PS1_y":     {"lambda_eff": 0.9633, "psf_fwhm": 1.1, "pixscale": 0.25, "zp": 25.0, "flux_factor": 1000.0,
                  "ps1_band": "y"},
    # --- 2MASS JHKs via IRSA IBE ---------------------------------------------
    # 2MASS Atlas images are Vega-calibrated with a PER-IMAGE zero point
    # (MAGZP header keyword, ~20.9 +/- a few 0.1 mag). _download_2mass
    # rescales every cutout to a common Vega zp of 20.0, so the stored zp is
    # 20.0 + the Vega->AB offset (Blanton et al. 2005: J +0.91, H +1.39,
    # Ks +1.85; consistent with Cohen et al. 2003 zero-flux calibration).
    # SkyView is deliberately NOT used for 2MASS: its tiles keep the native
    # per-image MAGZP (verified 2026-07-16: implied Vega zp 20.7-21.2 across
    # three fields), which a fixed zp cannot represent.
    # lambda_eff from Cohen et al. 2003 (1.235/1.662/2.159 um).
    "TwoMASS_J":  {"lambda_eff": 1.235, "psf_fwhm": 2.9, "pixscale": 1.0, "zp": 20.91, "flux_factor": 1000.0,
                   "twomass_band": "j"},
    "TwoMASS_H":  {"lambda_eff": 1.662, "psf_fwhm": 2.9, "pixscale": 1.0, "zp": 21.39, "flux_factor": 1000.0,
                   "twomass_band": "h"},
    "TwoMASS_Ks": {"lambda_eff": 2.159, "psf_fwhm": 2.9, "pixscale": 1.0, "zp": 21.85, "flux_factor": 1000.0,
                   "twomass_band": "k"},
    # --- Spitzer IRAC (SEIP mosaics) via IRSA SIA ----------------------------
    # SEIP super-mosaics are in PHYSICAL surface-brightness units (BUNIT =
    # MJy/sr) on a 0.6"/px grid, so no Vega->AB offset is involved: the AB zp
    # follows directly from the pixel solid angle.
    #   1 px = 0.6" x 0.6" = 8.4616e-12 sr  ->  1 MJy/sr = 8.4616e-6 Jy/px
    #   zp = -2.5 log10(8.4616e-6 / 3631) = 21.582 AB (same for all channels)
    # (Equivalent surface-brightness route to the IRAC handbook calibration;
    # cross-checked against the ZPAB keyword in SEIP mosaic headers.)
    # lambda_eff and PSF FWHM from the IRAC Instrument Handbook.
    # NOTE: SEIP only covers targeted Spitzer fields (~a few % of the sky);
    # most random positions will legitimately return "no coverage".
    "Spitzer_IRAC1": {"lambda_eff": 3.550, "psf_fwhm": 1.66, "pixscale": 0.6, "zp": 21.582, "flux_factor": 1000.0,
                      "seip_band": "IRAC1"},
    "Spitzer_IRAC2": {"lambda_eff": 4.493, "psf_fwhm": 1.72, "pixscale": 0.6, "zp": 21.582, "flux_factor": 1000.0,
                      "seip_band": "IRAC2"},
    "Spitzer_IRAC3": {"lambda_eff": 5.731, "psf_fwhm": 1.88, "pixscale": 0.6, "zp": 21.582, "flux_factor": 1000.0,
                      "seip_band": "IRAC3"},
    "Spitzer_IRAC4": {"lambda_eff": 7.872, "psf_fwhm": 1.98, "pixscale": 0.6, "zp": 21.582, "flux_factor": 1000.0,
                      "seip_band": "IRAC4"},
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


def _get_with_retry(url, what, timeout=120):
    """GET with the same retry/backoff policy as the Legacy downloader.

    Returns the requests Response. Raises DownloadError after exhausting
    RETRY_ATTEMPTS on 429/5xx or repeated network errors.
    """
    import requests

    headers = {"User-Agent": USER_AGENT}
    last_exc = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            sleep = _retry_sleep_seconds(attempt)
            logger.warning("%s network error (%s); retry %d/%d in %.1fs",
                           what, exc, attempt + 1, RETRY_ATTEMPTS, sleep)
            time.sleep(sleep)
            continue
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            last_exc = requests.exceptions.HTTPError(
                f"{resp.status_code} for {url}", response=resp)
            sleep = _retry_sleep_seconds(attempt,
                                         resp.headers.get("Retry-After"))
            logger.warning("%s got HTTP %d; retry %d/%d in %.1fs", what,
                           resp.status_code, attempt + 1, RETRY_ATTEMPTS,
                           sleep)
            time.sleep(sleep)
            continue
        resp.raise_for_status()
        return resp
    raise DownloadError(f"{what} failed after {RETRY_ATTEMPTS} attempts: "
                        f"{last_exc}")


def ps1_stack_filenames(ra, dec, filters="grizy"):
    """Query ps1images.stsci.edu for the stack skycell filenames at a
    position. Returns {filter: filename} (empty dict = no PS1 coverage,
    e.g. dec < -30)."""
    url = ("https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
           f"?ra={ra}&dec={dec}&filters={filters}&type=stack")
    resp = _get_with_retry(url, "PS1 filenames query", timeout=60)
    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        return {}
    cols = lines[0].split()
    i_filt, i_fname = cols.index("filter"), cols.index("filename")
    out = {}
    for line in lines[1:]:
        parts = line.split()
        out[parts[i_filt]] = parts[i_fname]
    return out


def _download_ps1(ra, dec, survey, size_arcmin, pixscale):
    """Pan-STARRS1 stack cutout via the public two-step API:
    ps1filenames.py (locate the skycell stack) + fitscut.cgi (FITS cutout).

    The returned image is divided by EXPTIME so the fixed AB zero point of
    25.0 in SURVEY_DEFAULTS applies (native stack zp = 25 + 2.5 log10(t_exp),
    Waters et al. 2020). Returns an astropy HDU.
    """
    import io

    band = SURVEY_DEFAULTS[survey]["ps1_band"]
    fnames = ps1_stack_filenames(ra, dec, filters=band)
    if band not in fnames:
        raise DownloadError(
            f"no PS1 stack image at ra={ra} dec={dec} (outside footprint, "
            f"dec < -30?)")
    npix = int(np.ceil(size_arcmin * 60.0 / pixscale))
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"
           f"?ra={ra}&dec={dec}&size={npix}&format=fits&red={fnames[band]}")
    resp = _get_with_retry(url, f"PS1 fitscut {survey}")
    hdu = fits.open(io.BytesIO(resp.content))[0]
    exptime = float(hdu.header.get("EXPTIME", 0) or 0)
    if exptime <= 0:
        raise DownloadError(f"PS1 cutout for {survey} has no EXPTIME; "
                            "cannot normalise to zp=25")
    if "BSOFTEN" in hdu.header:
        # fitscut normally serves linearised pixels; refuse silently wrong
        # photometry if an asinh-compressed image ever comes through.
        raise DownloadError(f"PS1 cutout for {survey} is asinh-compressed "
                            "(BSOFTEN present); not supported")
    hdu.data = np.asarray(hdu.data, dtype=np.float32) / exptime
    hdu.header["EXPNORM"] = (exptime, "pixels divided by EXPTIME; zp=25 AB")
    return hdu


def _download_2mass(ra, dec, survey, size_arcmin, pixscale):
    """2MASS Atlas cutout via IRSA IBE, normalised to a common zero point.

    IBE search gives the Atlas image (and its per-image Vega MAGZP) covering
    the position; the IBE data endpoint serves a center/size cutout of it.
    Pixels are rescaled by 10^(-0.4 (MAGZP - 20)) so the fixed zp in
    SURVEY_DEFAULTS (20.0 Vega + Vega->AB offset) applies. Returns an HDU.
    """
    import io

    band = SURVEY_DEFAULTS[survey]["twomass_band"]
    search = ("https://irsa.ipac.caltech.edu/ibe/search/twomass/allsky/allsky"
              f"?POS={ra},{dec}&where=filter%3D%27{band}%27"
              "&columns=fname,ordate,hemisphere,scanno,magzp&ct=csv")
    resp = _get_with_retry(search, f"2MASS IBE search {survey}", timeout=60)
    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        raise DownloadError(f"no 2MASS Atlas image at ra={ra} dec={dec} "
                            f"band {band}")
    cols = lines[0].split(",")
    row = lines[1].split(",")
    rec = dict(zip(cols, row))
    path = (f"{rec['ordate']}{rec['hemisphere']}/"
            f"s{int(rec['scanno']):03d}/image/{rec['fname']}")
    size_arcsec = size_arcmin * 60.0
    url = ("https://irsa.ipac.caltech.edu/ibe/data/twomass/allsky/allsky/"
           f"{path}?center={ra},{dec}&size={size_arcsec}arcsec&gzip=false")
    resp = _get_with_retry(url, f"2MASS IBE cutout {survey}")
    hdu = fits.open(io.BytesIO(resp.content))[0]
    magzp = float(hdu.header.get("MAGZP", rec.get("magzp")))
    scale = 10.0 ** (-0.4 * (magzp - 20.0))
    hdu.data = np.asarray(hdu.data, dtype=np.float32) * scale
    hdu.header["ZPNORM"] = (magzp, "native Vega MAGZP; rescaled to zp=20.0")
    return hdu


def seip_mosaic_urls(ra, dec, band=None, radius_deg=0.002):
    """Query IRSA SIA v2 for public Spitzer SEIP science mosaics covering a
    position. Returns list of access URLs (empty = no SEIP coverage).
    band: 'IRAC1'..'IRAC4' (or 'MIPS24') to filter; None = all."""
    url = ("https://irsa.ipac.caltech.edu/SIA?COLLECTION=spitzer_seip"
           f"&POS=circle+{ra}+{dec}+{radius_deg}&RESPONSEFORMAT=CSV")
    resp = _get_with_retry(url, "SEIP SIA query", timeout=90)
    import csv as _csv
    import io as _io
    rows = list(_csv.DictReader(_io.StringIO(resp.text)))
    urls = []
    for r in rows:
        if not r.get("access_url", "").endswith(".mosaic.fits"):
            continue
        if band and r.get("energy_bandpassname") != band:
            continue
        urls.append(r["access_url"])
    return urls


def _download_seip(ra, dec, survey, size_arcmin, pixscale):
    """Spitzer SEIP mosaic cutout via IRSA SIA v2 + lazy range requests.

    SEIP has no cutout endpoint and the full mosaics are ~100 MB, so the
    FITS is opened remotely with fsspec and only the cutout pixels are
    fetched (``hdu.section``). Pixels stay in the native MJy/sr; the zp in
    SURVEY_DEFAULTS converts them (see the SURVEY_DEFAULTS comment).
    Returns an astropy HDU.
    """
    band = SURVEY_DEFAULTS[survey]["seip_band"]
    urls = seip_mosaic_urls(ra, dec, band=band)
    if not urls:
        raise DownloadError(f"no SEIP {band} mosaic at ra={ra} dec={dec} "
                            "(outside Spitzer coverage)")
    half = int(np.ceil(size_arcmin * 60.0 / pixscale / 2.0))
    last_exc = None
    for url in urls[:2]:  # try at most 2 overlapping mosaics
        try:
            with fits.open(url, use_fsspec=True) as hdul:
                hdu0 = hdul[0]
                wcs = WCS(hdu0.header)
                x, y = wcs.all_world2pix(ra, dec, 0)
                x, y = int(round(float(x))), int(round(float(y)))
                ny = hdu0.header["NAXIS2"]
                nx = hdu0.header["NAXIS1"]
                y0, y1 = max(0, y - half), min(ny, y + half)
                x0, x1 = max(0, x - half), min(nx, x + half)
                if y1 - y0 < 4 or x1 - x0 < 4:
                    raise DownloadError("target on mosaic edge")
                data = np.asarray(hdu0.section[y0:y1, x0:x1],
                                  dtype=np.float32)
                header = hdu0.header.copy()
                header["NAXIS1"], header["NAXIS2"] = data.shape[1], data.shape[0]
                header["CRPIX1"] = header["CRPIX1"] - x0
                header["CRPIX2"] = header["CRPIX2"] - y0
                header["SEIPURL"] = (url[-68:], "source SEIP mosaic")
                return fits.PrimaryHDU(data=data, header=header)
        except Exception as exc:
            last_exc = exc
            logger.warning("SEIP cutout from %s failed (%s), trying next",
                           url, exc)
    raise DownloadError(f"SEIP cutout for {survey} failed: {last_exc}")


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
            elif survey.startswith("PS1"):
                hdu = _download_ps1(ra, dec, survey, size_arcmin, meta["pixscale"])
            elif survey.startswith("TwoMASS"):
                # no SkyView fallback: SkyView 2MASS tiles keep per-image
                # zero points that the fixed zp cannot represent
                hdu = _download_2mass(ra, dec, survey, size_arcmin, meta["pixscale"])
            elif survey.startswith("Spitzer"):
                hdu = _download_seip(ra, dec, survey, size_arcmin, meta["pixscale"])
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
                    max_workers=2, overwrite=False, legacy_bulk=False,
                    brick_cache_dir=None):
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
    legacy_bulk : bool
        OPT-IN (default False = behaviour unchanged): fetch the Legacy_*
        bands via bulk DR10 brick mosaics (one file-server GET per brick,
        shared by all targets in it, cutouts extracted locally) instead of
        one rate-limited cutout-API call per target per band. Targets may
        carry an optional integer "brickid" key to skip the spatial brick
        lookup. All non-Legacy surveys still use the per-target flow. The
        cache files produced are identical in layout, so this only changes
        HOW the cache is populated. See ``bulk_bricks.py``.
    brick_cache_dir : str, optional
        Only with legacy_bulk=True: where brick mosaics are cached
        (default ``~/.cache/photextra_pipeline``).

    Returns
    -------
    failures : list of dict
        One entry per target that has at least one failed band:
        {"target_id", "failed_bands": {survey: reason}}. Also appended to
        ``{output_dir}/failed_targets.jsonl`` (stage="prefetch").
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    failures = []
    per_target_surveys = list(surveys)
    if legacy_bulk:
        legacy_surveys = [s for s in surveys if s.startswith("Legacy")]
        if legacy_surveys:
            from .bulk_bricks import bulk_prefetch_legacy
            failures.extend(bulk_prefetch_legacy(
                targets, legacy_surveys, size_arcmin, output_dir,
                brick_cache_dir=brick_cache_dir, max_workers=max_workers,
                overwrite=overwrite))
            # Legacy bands are handled entirely by the bulk path above
            # (successes cached, failures recorded); the per-target loop
            # below only fetches the remaining surveys.
            per_target_surveys = [s for s in surveys
                                  if not s.startswith("Legacy")]
            if not per_target_surveys:
                return failures

    def _fetch_one(tgt):
        cache_dir = os.path.join(output_dir, str(tgt["id"]), "cache")
        images = hostphot_downloader(tgt["ra"], tgt["dec"],
                                     per_target_surveys,
                                     size_arcmin, cache_dir,
                                     overwrite=overwrite)
        failed = {s: e.get("error", "unknown") for s, e in images.items()
                  if e.get("data") is None}
        return tgt["id"], failed
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
                            tid, len(per_target_surveys))
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
