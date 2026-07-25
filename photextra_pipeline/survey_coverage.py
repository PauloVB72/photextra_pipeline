"""Survey coverage / footprint checks for the pipeline's imaging surveys.

Answers "does survey X have data at position Y" with lightweight
metadata/footprint queries only (no full image downloads), for every survey
in ``downloader.SURVEY_DEFAULTS`` plus the pointed / auth-gated surveys that
have no blind-cutout API (HST, HSC, S-PLUS, J-PLUS).

Per-survey method:
  - WISE, TwoMASS            : all-sky -> always "covered".
  - PS1                      : ps1filenames.py metadata query (a stack image
                               listed at the position == coverage; none south
                               of dec -30).
  - SDSS                     : astroquery.sdss photoobj cone search; SDSS
                               source density makes an empty 60" cone a
                               reliable "outside footprint" signal.
  - Legacy                   : 32-px single-band test cutout (few kB); an
                               all-blank/NaN image == outside footprint.
  - GALEX, HST               : MAST Observations metadata query. For HST the
                               per-target instrument/filter list is returned
                               in an extra "HST_detail" column.
  - Spitzer (SEIP)           : IRSA SIA v2 metadata query.
  - JPLUS                    : public SIAP query at archive.cefca.es
                               (J-PLUS DR3). Full images are public but only
                               as ~50 MB .fz files, hence footprint-only.
  - HSC, SPLUS               : no public unauthenticated footprint/cutout
                               endpoint (STARS / splus.cloud tokens needed)
                               -> "unknown (auth required)".

Example (CLI)::

    python -m photextra_pipeline.survey_coverage targets.csv \
        --surveys SDSS_r,PS1_r,TwoMASS_J,HST,JPLUS \
        --out-csv coverage.csv --out-png coverage_map.png

The targets CSV needs id/ra/dec columns (also accepts the run_mkw8_full.py
schema ind/RAdeg/DEdeg). Programmatic use::

    from photextra_pipeline.survey_coverage import check_coverage, plot_coverage_map
    df = check_coverage([{"id": "g1", "ra": 150.1, "dec": 2.2}],
                        ["SDSS_r", "PS1_r", "HST"])
    plot_coverage_map(df, "coverage_map.png")
"""

import logging

import numpy as np

from .downloader import (SURVEY_DEFAULTS, ps1_stack_filenames,
                         seip_mosaic_urls, _get_with_retry)

logger = logging.getLogger(__name__)

COVERED = "covered"
NOT_COVERED = "not_covered"
UNKNOWN_AUTH = "unknown (auth required)"

# surveys checkable here that are NOT in SURVEY_DEFAULTS (no blind cutout API)
EXTRA_SURVEYS = {
    "HST":   {"requires_auth": False,
              "note": "pointed observations only; see HST_detail column"},
    "HSC":   {"requires_auth": True,
              "note": "cutouts and footprint behind STARS account"},
    "SPLUS": {"requires_auth": True,
              "note": "splus.cloud cutouts require a token"},
    "JPLUS": {"requires_auth": False,
              "note": "footprint via public SIAP; no public cutout API"},
}


# ---------------------------------------------------------------- checkers

def _check_allsky(ra, dec, radius_arcsec):
    return COVERED


def _check_ps1(ra, dec, radius_arcsec):
    if dec < -30.5:
        return NOT_COVERED
    return COVERED if ps1_stack_filenames(ra, dec, filters="grizy") \
        else NOT_COVERED


def _check_sdss(ra, dec, radius_arcsec):
    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coord = SkyCoord(ra * u.deg, dec * u.deg)
    tab = SDSS.query_region(coord, radius=max(radius_arcsec, 30) * u.arcsec,
                            photoobj_fields=["objID"])
    return COVERED if tab is not None and len(tab) > 0 else NOT_COVERED


def _check_legacy(ra, dec, radius_arcsec):
    import io
    import requests
    from astropy.io import fits

    url = ("https://www.legacysurvey.org/viewer/fits-cutout"
           f"?ra={ra}&dec={dec}&layer=ls-dr10&pixscale=1.0&bands=r&size=32")
    # no retry loop here: the cutout service answers HTTP 500 when NO brick
    # exists at the position (verified at dec=-69), so a 500 means "outside
    # footprint", not a transient failure worth 60 s of backoff per target.
    resp = requests.get(url, timeout=60,
                        headers={"User-Agent":
                                 "photextra_pipeline survey_coverage"})
    if resp.status_code == 500:
        return NOT_COVERED
    resp.raise_for_status()
    data = fits.open(io.BytesIO(resp.content))[0].data
    if data is None:
        return NOT_COVERED
    data = np.asarray(data, float)
    finite = data[np.isfinite(data)]
    if finite.size == 0 or np.all(finite == 0):
        return NOT_COVERED
    return COVERED


def _check_galex(ra, dec, radius_arcsec):
    from astroquery.mast import Observations
    import astropy.units as u

    n = Observations.query_criteria_count(
        coordinates=f"{ra} {dec}", radius=radius_arcsec * u.arcsec,
        obs_collection="GALEX")
    return COVERED if n > 0 else NOT_COVERED


def _check_seip(ra, dec, radius_arcsec, band=None):
    urls = seip_mosaic_urls(ra, dec, band=band,
                            radius_deg=radius_arcsec / 3600.0)
    return COVERED if urls else NOT_COVERED


def _check_hst(ra, dec, radius_arcsec):
    """Return (status, detail). detail = 'instrument:filters; ...' summary of
    the HST imaging available around the position."""
    from astroquery.mast import Observations
    import astropy.units as u

    tab = Observations.query_criteria(
        coordinates=f"{ra} {dec}", radius=radius_arcsec * u.arcsec,
        obs_collection="HST", dataproduct_type="image")
    if tab is None or len(tab) == 0:
        return NOT_COVERED, ""
    pairs = sorted({f"{row['instrument_name']}:{row['filters']}"
                    for row in tab})
    return COVERED, "; ".join(pairs)


def _check_jplus(ra, dec, radius_arcsec):
    """J-PLUS DR3 public SIAP. The service matches on image *centers* inside
    the POS/SIZE box and silently returns nothing for SIZE > 1 (verified
    2026-07-16), so SIZE=1.0 (the maximum) is used: an image center within
    ~0.5 deg means the position is on a J-PLUS pointing (T80Cam FoV is
    1.4x1.4 deg). Approximate at pointing edges - a position 0.5-0.9 deg
    from the nearest image center is reported not_covered even though it may
    sit on the image corner."""
    url = ("https://archive.cefca.es/catalogues/vo/siap/jplus-dr3"
           f"?POS={ra},{dec}&SIZE=1.0")
    resp = _get_with_retry(url, "J-PLUS SIAP probe", timeout=60)
    return COVERED if "<TR>" in resp.text else NOT_COVERED


def _checker_for(survey):
    """Return callable (ra, dec, radius_arcsec) -> status for a survey name,
    or None if the survey cannot be checked without credentials."""
    fam = survey.split("_")[0]
    if fam in ("WISE", "TwoMASS"):
        return _check_allsky
    if fam == "PS1":
        return _check_ps1
    if fam == "SDSS":
        return _check_sdss
    if fam == "Legacy":
        return _check_legacy
    if fam == "GALEX":
        return _check_galex
    if fam == "Spitzer":
        band = SURVEY_DEFAULTS.get(survey, {}).get("seip_band")
        return lambda ra, dec, r: _check_seip(ra, dec, r, band=band)
    if survey == "JPLUS":
        return _check_jplus
    if survey == "HST":
        return lambda ra, dec, r: _check_hst(ra, dec, r)[0]
    return None  # HSC, SPLUS: auth required


# ---------------------------------------------------------------- public API

def check_coverage(targets, surveys, radius_arcsec=60):
    """Coverage status of each survey at each target position.

    Parameters
    ----------
    targets : list of dict
        Each needs "id", "ra", "dec" (deg). Any extra keys are ignored.
    surveys : list of str
        Keys of SURVEY_DEFAULTS (e.g. "SDSS_r") and/or the footprint-only
        names in EXTRA_SURVEYS ("HST", "HSC", "SPLUS", "JPLUS").
    radius_arcsec : float
        Search radius for the metadata queries.

    Returns
    -------
    pandas.DataFrame
        Columns: target_id, ra, dec, then one column per requested survey
        with a status in {"covered", "not_covered",
        "unknown (auth required)", "error: <msg>"}. If "HST" is requested an
        extra "HST_detail" column lists the instrument:filter combinations
        found near each covered target.
    """
    import pandas as pd

    unknown = [s for s in surveys
               if s not in SURVEY_DEFAULTS and s not in EXTRA_SURVEYS]
    if unknown:
        raise ValueError(f"unknown survey names: {unknown}")

    # bands of the same family share one footprint query per target
    fam_key = {}
    for s in surveys:
        fam = s.split("_")[0]
        # SEIP coverage can differ per channel -> key per survey, not family
        fam_key[s] = s if fam == "Spitzer" else fam

    rows = []
    for tgt in targets:
        tid, ra, dec = str(tgt["id"]), float(tgt["ra"]), float(tgt["dec"])
        row = {"target_id": tid, "ra": ra, "dec": dec}
        fam_cache = {}
        for s in surveys:
            key = fam_key[s]
            if key not in fam_cache:
                checker = _checker_for(s)
                if checker is None:
                    fam_cache[key] = UNKNOWN_AUTH
                else:
                    try:
                        if s == "HST":
                            status, detail = _check_hst(ra, dec,
                                                        radius_arcsec)
                            row["HST_detail"] = detail
                            fam_cache[key] = status
                        else:
                            fam_cache[key] = checker(ra, dec, radius_arcsec)
                    except Exception as exc:  # per-cell isolation
                        logger.warning("coverage check %s @ %s failed: %s",
                                       s, tid, exc)
                        fam_cache[key] = f"error: {exc}"
            row[s] = fam_cache[key]
        rows.append(row)
        logger.info("coverage: %s done", tid)

    cols = ["target_id", "ra", "dec"] + list(surveys)
    if "HST" in surveys:
        cols.append("HST_detail")
    return pd.DataFrame(rows, columns=cols)


# statuses -> (color, marker). Okabe-Ito colorblind-safe hues plus marker
# shape as secondary encoding, so status is never color-alone.
_STATUS_STYLE = {
    COVERED:      ("#009E73", "o", "covered"),
    NOT_COVERED:  ("#D55E00", "x", "not covered"),
    UNKNOWN_AUTH: ("#767676", "s", "unknown (auth)"),
    "error":      ("#0072B2", "^", "error"),
}


def plot_coverage_map(coverage_df, out_png):
    """Sky map of coverage: one panel per survey, one point per target.

    RA/Dec scatter (RA axis inverted, astronomy convention); point color AND
    marker shape encode the coverage status; single shared legend. Saved as
    a PNG with the Agg backend (safe on headless nodes).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta_cols = {"target_id", "ra", "dec", "HST_detail"}
    survey_cols = [c for c in coverage_df.columns if c not in meta_cols]
    n = len(survey_cols)
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.6 * ncols, 3.9 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    def style_of(status):
        if status in _STATUS_STYLE:
            return _STATUS_STYLE[status]
        return _STATUS_STYLE["error"]  # "error: <msg>" cells

    for ax, col in zip(axes.flat, survey_cols):
        for status_key, (color, marker, _label) in _STATUS_STYLE.items():
            if status_key == "error":
                sel = coverage_df[col].astype(str).str.startswith("error")
            else:
                sel = coverage_df[col] == status_key
            if not sel.any():
                continue
            ax.scatter(coverage_df.loc[sel, "ra"],
                       coverage_df.loc[sel, "dec"],
                       s=22, c=color, marker=marker, linewidths=1.2,
                       alpha=0.85)
        ax.set_title(col, fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("RA [deg]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dec [deg]")
    axes[0, 0].invert_xaxis()

    handles = [Line2D([], [], linestyle="", marker=m, color=c, label=lab,
                      markersize=7)
               for c, m, lab in _STATUS_STYLE.values()]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), frameon=False)
    fig.suptitle("Survey coverage", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("coverage map written to %s", out_png)
    return out_png


def _load_targets_csv(path, limit=None):
    """Generic id/ra/dec CSV loader (also accepts ind/RAdeg/DEdeg headers,
    the run_mkw8_full.py schema)."""
    import csv

    id_keys, ra_keys, dec_keys = ("id", "ind"), ("ra", "RAdeg"), ("dec", "DEdeg")
    targets = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            def pick(keys):
                for k in keys:
                    for variant in (k, k.lower(), k.upper()):
                        if variant in row:
                            return row[variant]
                raise KeyError(f"none of {keys} in CSV header")
            targets.append({"id": str(pick(id_keys)).strip(),
                            "ra": float(pick(ra_keys)),
                            "dec": float(pick(dec_keys))})
    return targets[:limit] if limit else targets


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Check survey coverage for a list of targets.")
    p.add_argument("targets_csv", help="CSV with id/ra/dec columns")
    p.add_argument("--surveys", required=True,
                   help="comma-separated, e.g. SDSS_r,PS1_r,TwoMASS_J,HST")
    p.add_argument("--radius", type=float, default=60,
                   help="search radius in arcsec (default 60)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-csv", default="coverage.csv")
    p.add_argument("--out-png", default=None,
                   help="also write a sky coverage map PNG")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")
    targets = _load_targets_csv(args.targets_csv, args.limit)
    df = check_coverage(targets, args.surveys.split(","),
                        radius_arcsec=args.radius)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten: {args.out_csv}")
    if args.out_png:
        plot_coverage_map(df, args.out_png)
        print(f"written: {args.out_png}")


if __name__ == "__main__":
    main()
