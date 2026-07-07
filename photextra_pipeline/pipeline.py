"""Main Pipeline class orchestrating the multiband photometry flow.

Flow: download -> xdebpair (source separation) -> convolve to W4 PSF ->
reproject to common grid -> reproject masks -> measure fluxes -> write outputs.

xdebpair is tried first; falls back to xmask if unavailable, then to a
built-in stub that produces a single circular component.

The ``mode`` selector chooses which analysis runs:

- ``"photometry"`` (default): full photometric pipeline only; the spectral
  fit is skipped even if the target carries ``spectrum_path`` + ``z``.
- ``"spectroscopy"``: only the XpectraFit spectral fit (no image downloads,
  deblending, SED or galaxy table); requires ``z``. If ``spectrum_path`` is
  absent, the matching SDSS spectrum is resolved from (ra, dec, z) via
  :mod:`.spectrum_acquisition` and cached under ``{id}/spectroscopy/``.
- ``"both"``: photometric pipeline plus the spectral fit, COMBINED: the
  spectrum's synthetic broadband photometry (Legacy/SDSS bands) is
  normalized to the real imaging photometry via an S/N^2-weighted grey
  factor over all usable Legacy bands to correct fibre aperture losses
  (extension of Zou et al. 2024, ApJ 961, 173; see
  :meth:`Pipeline._combine_phot_spec`), and one merged per-target product
  ``{id}_combined.csv/.ecsv`` is written.

Checkpointing/resume: :meth:`Pipeline.run` skips a target whose final
product for the selected mode already exists on disk, and ``mode="both"``
reuses cached ``_photometry``/``_spectral`` products from earlier
photometry/spectroscopy passes over the same ``output_dir``, only
computing the missing pieces and the combination step.
"""

import os
import logging

import numpy as np
import yaml
from astropy.table import Table

from .downloader import (hostphot_downloader, SURVEY_DEFAULTS,
                         DownloadError, append_failed_target)
from .convolution import convolve_all, TARGET_SURVEY
from .reprojection import build_common_wcs, reproject_all, reproject_masks
from .photometry import measure_isolated_photometry
from .output_table import make_galaxy_table, save_galaxy_table
from .spec_normalization import (collect_anchors, fit_spec_normalization,
                                 line_observed_wavelength_aa)
from . import validation_plots as vp

logger = logging.getLogger(__name__)


def _load_xmask():
    """Import real xmask or fall back to a built-in stub."""
    try:
        from xmask import XmaskPy
        return XmaskPy, False
    except Exception:
        logger.warning("xmask not available; using built-in stub")
        return _StubXmask, True


def _xdebpair_available():
    """Check whether xdebpair can be imported."""
    try:
        import sys
        _p = "/home/polo/Escritorio/PHD/code/xdebpair"
        if _p not in sys.path:
            sys.path.insert(0, _p)
        import xdebpair  # noqa: F401
        return True
    except Exception:
        return False


class _StubXmaskResult:
    def __init__(self, masks, wcs, n_components, separation_arcsec):
        self.masks = masks
        self.wcs = wcs
        self.n_components = n_components
        self.is_merger = n_components > 1
        self.separation_arcsec = separation_arcsec


class _StubXmask:
    """Minimal stand-in producing one component on a synthetic Legacy grid."""

    def __init__(self, ra, dec, size_arcmin=1.0):
        self.ra = ra
        self.dec = dec
        self.size_arcmin = size_arcmin

    def run(self):
        from .reprojection import build_common_wcs
        pixscale = 0.262
        npix = int(np.ceil(self.size_arcmin * 60.0 / pixscale))
        wcs = build_common_wcs(self.ra, self.dec, pixscale, npix)
        yy, xx = np.indices((npix, npix))
        c = (npix - 1) / 2.0
        r = (10.0 / pixscale)  # ~10" radius blob
        mask = (xx - c) ** 2 + (yy - c) ** 2 <= r ** 2
        return _StubXmaskResult({"gal1": mask}, wcs, 1, 0.0)


VALID_MODES = ("photometry", "spectroscopy", "both")
VALID_APERTURE_MODES = ("mask", "aperture", "sep_apertures")
VALID_SEPARATIONS = ("central", "total", "pair")


class Pipeline:
    def __init__(self, config="config_xmask.yaml", use_xdebpair=None,
                 mode=None):
        """
        Parameters
        ----------
        config : str or dict
            Path to the YAML config, or an already-loaded config dict.
            All pipeline knobs live in the config file (see
            ``config_xmask.yaml`` for the commented schema); the keyword
            arguments below OVERRIDE the config value when passed
            explicitly (backward compatibility with existing driver
            scripts).
        use_xdebpair : bool, optional
            Try xdebpair for source separation before xmask/stub.
            Default: config key ``use_xdebpair`` (true).
        mode : {"photometry", "spectroscopy", "both"}, optional
            Which analysis :meth:`run` performs. ``"photometry"`` runs only
            the photometric pipeline; ``"spectroscopy"`` runs only the
            XpectraFit spectral fit; ``"both"`` runs the photometric
            pipeline plus the spectral fit when the target has a spectrum.
            Default: config key ``mode`` (photometry).

        Notes
        -----
        The CIGALE knobs are config-only (no kwargs):

        - ``cigale_run`` (bool, default false): enable the CIGALE SED fit
          for ANY mode, not only ``both``. It is the ONLY switch for CIGALE;
          the built-in ``_combine_phot_spec`` combination ALWAYS runs in
          mode="both" regardless, and CIGALE is simply an ADDITIONAL fit on
          top. In mode="spectroscopy" CIGALE fits the spectrum alone at the known
          z (less constrained than a combined fit); in mode="photometry"
          it fits the broadband fluxes with the redshift left blank so
          CIGALE chi2-scans the ``redshifting`` z grid (photo-z,
          ``cigale_z_phot`` column) — that grid must be hand-edited into
          the packaged ``photextra_pipeline/data/cigale/pcigale.ini``
          (the pipeline copies it as-is and NEVER writes its SED-module
          parameters).
        - ``cigale_batch`` (bool, default false): when true, :meth:`run`
          SKIPS the per-target CIGALE fit entirely; the driver script
          MUST call :meth:`run_cigale_batch` itself once after the whole
          run — the pipeline cannot detect a forgotten call.
        """
        if isinstance(config, dict):
            self.config = config
        else:
            with open(config) as fh:
                self.config = yaml.safe_load(fh)

        # --- mode / method selectors: config value, kwarg overrides ------
        self.mode = mode if mode is not None else \
            self.config.get("mode", "photometry")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, "
                             f"got {self.mode!r}")
        # CIGALE flags (config-only; see __init__ Notes and
        # config_xmask.yaml). cigale_run is the single switch that enables
        # the CIGALE SED fit for ANY mode. cigale_batch=True means the
        # DRIVER must call run_cigale_batch() manually after the full run.
        # All CIGALE SED-module tuning (z grid, use_spectro, ...) is done by
        # hand-editing photextra_pipeline/data/cigale/pcigale.ini — the
        # pipeline never auto-writes that file.
        self.cigale_run = bool(self.config.get("cigale_run", False))
        self.cigale_batch = bool(self.config.get("cigale_batch", False))
        if self.cigale_run:
            logger.warning(
                "cigale_run=True: the CIGALE fit uses the packaged "
                "pcigale.ini / pcigale_photoz.ini under "
                "photextra_pipeline/data/cigale/ AS-IS. Their redshift grid "
                "(photo-z) and SED-module grids are NEVER auto-configured — "
                "hand-edit those .ini templates for your science case.")

        # --- photometry sub-config ----------------------------------------
        phot_cfg = self.config.get("photometry") or {}
        self.aperture_mode = phot_cfg.get("aperture_mode", "mask")
        if self.aperture_mode not in VALID_APERTURE_MODES:
            raise ValueError(f"photometry.aperture_mode must be one of "
                             f"{VALID_APERTURE_MODES}, "
                             f"got {self.aperture_mode!r}")
        self.separation = phot_cfg.get("separation", "pair")
        if self.separation not in VALID_SEPARATIONS:
            raise ValueError(f"photometry.separation must be one of "
                             f"{VALID_SEPARATIONS}, got {self.separation!r}")
        # circular aperture radius (arcsec) for aperture_mode="aperture"
        self.aperture_radius_arcsec = float(
            phot_cfg.get("aperture_radius_arcsec", 5.0))

        # --- SEP source-detection sub-config (aperture_mode="sep_apertures") -
        # These knobs feed photextra's OWN detect_sep_ellipse /
        # measure_sep_elliptical_photometry (the sep_apertures path); the
        # defaults reproduce the previously-hardcoded literals so behaviour is
        # unchanged unless the user edits the config. NOTE: xmask's internal
        # SEP parameters (thresh/minarea/deblend_nthresh/deblend_cont in
        # xmask.deblend) are NOT independently tunable yet — the public xmask
        # API (XmaskPy) does not accept them as kwargs, so this sep: section
        # does not reach the xmask mask builder used by aperture_mode="mask".
        sep_cfg = self.config.get("sep") or {}
        self.sep_thresh_sigma = float(sep_cfg.get("thresh_sigma", 1.5))
        self.sep_ellipse_k = float(sep_cfg.get("ellipse_k", 2.5))
        self.sep_max_match_arcsec = float(sep_cfg.get("max_match_arcsec", 10.0))
        self.sep_ref_survey = sep_cfg.get("ref_survey", "Legacy_r")

        # --- spectroscopy sub-config ---------------------------------------
        spec_cfg = self.config.get("spectroscopy") or {}
        self.fit_agn = bool(spec_cfg.get("fit_agn", True))
        self.spectrum_fwhm = float(spec_cfg.get("spectrum_fwhm", 2.5))

        self.surveys = self.config["surveys"]
        self.output_dir = self.config["output_dir"]
        grid = self.config["common_grid"]
        self.grid_pixscale = grid["pixscale"]
        self.grid_size = grid["size"]
        self.download_size = self.config.get("download_size", 1)
        # survey families whose synthetic photometry is computed from the
        # spectrum (spectroscopy/both modes); config key "survey_filters"
        # (accepted both at top level and under spectroscopy:)
        self.survey_filters = spec_cfg.get(
            "survey_filters", self.config.get("survey_filters",
                                              ["Legacy", "SDSS"]))
        self._XmaskPy, self._xmask_is_stub = _load_xmask()
        want_xdeb = use_xdebpair if use_xdebpair is not None else \
            bool(self.config.get("use_xdebpair", True))
        self.use_xdebpair = want_xdeb and _xdebpair_available()
        if want_xdeb and not self.use_xdebpair:
            logger.warning("xdebpair requested but not importable; "
                           "falling back to xmask/stub")

    def run(self, target):
        """Run the analysis selected by ``self.mode`` on one target dict."""
        target_id = target["id"]
        ra, dec = target["ra"], target["dec"]
        logger.info("=== %s (%.5f, %.5f) [mode=%s] ===",
                    target_id, ra, dec, self.mode)

        tdir = os.path.join(self.output_dir, target_id)
        plot_dir = os.path.join(tdir, "plots")
        prod_dir = os.path.join(tdir, "products")
        cache_dir = os.path.join(tdir, "cache")
        for d in (plot_dir, prod_dir, cache_dir):
            os.makedirs(d, exist_ok=True)

        # --- Checkpointing: skip work whose final product already exists ---
        # "already done" criterion: the mode's final product file exists and
        # is non-empty in {tdir}/products/ (photometry -> _photometry,
        # spectroscopy -> _spectral, both -> _combined).
        if self.mode == "photometry":
            phot_cached = self._cached_product(prod_dir, target_id,
                                               "photometry")
            if phot_cached:
                # fill in cigale_* columns missing from an older cached
                # product (never rerun CIGALE if they are already there)
                if self._cigale_per_target() and \
                        not self._product_has_cigale(phot_cached):
                    phot = self._load_cached_photometry(phot_cached,
                                                        self.surveys)
                    row = self._cigale_row_photometry(target, phot or {})
                    cols = self._run_cigale_for_target(
                        target, tdir, row, cigale_mode="photometry")
                    if cols:
                        self._append_cigale_columns(prod_dir, target_id,
                                                    "photometry", cols)
                logger.info("%s: cached photometry product found; skipping",
                            target_id)
                return {"output_dir": tdir, "skipped": "cached_photometry"}
        if self.mode == "spectroscopy":
            spec_cached = self._cached_product(prod_dir, target_id,
                                               "spectral")
            if spec_cached:
                if self._cigale_per_target() and \
                        not self._product_has_cigale(spec_cached):
                    row = self._load_cached_spectral(prod_dir, target_id) \
                        or {}
                    row["z"] = target.get("z")
                    cols = self._run_cigale_for_target(
                        target, tdir, row, cigale_mode="spectroscopy")
                    if cols:
                        self._append_cigale_columns(prod_dir, target_id,
                                                    "spectral", cols)
                logger.info("%s: cached spectral product found; skipping",
                            target_id)
                return {"output_dir": tdir, "skipped": "cached_spectral"}

        cached_spec = None
        if self.mode == "both":
            if self._cached_product(prod_dir, target_id, "combined"):
                logger.info("%s: cached combined product found; skipping",
                            target_id)
                return {"output_dir": tdir, "skipped": "cached_combined"}
            # reuse products from earlier photometry/spectroscopy passes
            cached_spec = self._load_cached_spectral(prod_dir, target_id)
            phot_path = self._cached_product(prod_dir, target_id,
                                             "photometry")
            if phot_path:
                return self._run_both_from_cache(
                    target, tdir, prod_dir, plot_dir, cache_dir,
                    phot_path, cached_spec)

        # --- Spectroscopy-only: skip the whole photometric pipeline ---
        if self.mode == "spectroscopy":
            spectral_result = self._run_spectroscopy(target, tdir)
            if spectral_result is not None and self._cigale_per_target():
                # spectrum-only CIGALE fit at the target's known z (no
                # broadband columns; less constrained than a combined fit)
                row = dict(spectral_result)
                row["z"] = target.get("z")
                cols = self._run_cigale_for_target(
                    target, tdir, row, cigale_mode="spectroscopy")
                if cols:
                    spectral_result.update(cols)
            self._write_spectral_products(prod_dir, target_id, spectral_result)
            logger.info("done %s -> %s", target_id, tdir)
            out = {"output_dir": tdir}
            if spectral_result is not None:
                out["spectral_result"] = spectral_result
            return out

        # Download all bands first so xdebpair can reuse Legacy_r, _g, _z
        images = hostphot_downloader(ra, dec, self.surveys, self.download_size, cache_dir)
        self._check_download_failures(target_id, images)

        target_wcs = build_common_wcs(ra, dec, self.grid_pixscale, self.grid_size)
        shape = (self.grid_size, self.grid_size)

        if self.aperture_mode == "mask":
            # Source separation: xdebpair > xmask > stub, then the
            # classification-aware separation policy (central/total/pair)
            seg_result = self._run_deblending(ra, dec, images)
            seg_result = self._apply_separation_policy(seg_result, target)
        else:
            # aperture / sep_apertures: no xdebpair masks; a single circular
            # or SEP-elliptical aperture defines the (one) component
            seg_result = self._run_aperture_detection(target, images)
        masks_native = seg_result.masks
        mask_wcs = seg_result.wcs

        # Full-image convolution + reprojection are kept ONLY for the processing
        # grid visualization, not for the science fluxes.
        images_conv = convolve_all(images)
        images_conv = reproject_all(images_conv, target_wcs, shape)
        masks_reproj = reproject_masks(masks_native, mask_wcs, target_wcs, shape)

        if self.aperture_mode == "mask":
            # --- Primary: per-component SED via isolated excess images ---
            # Each component is isolated, convolved to the common PSF, and
            # reprojected; PSF spread leaks into empty space rather than the
            # neighbour, so close-pair flux is not cross-contaminated.
            photometry = measure_isolated_photometry(
                images, masks_native, mask_wcs, self.surveys,
                target_wcs, shape, self.grid_pixscale,
                separation_arcsec=seg_result.separation_arcsec,
            )
        elif self.aperture_mode == "aperture":
            from .photometry import measure_circular_aperture_photometry
            photometry = measure_circular_aperture_photometry(
                images, self.surveys, ra, dec, self.aperture_radius_arcsec)
        else:  # sep_apertures
            from .photometry import measure_sep_elliptical_photometry
            photometry, _ellipse = measure_sep_elliptical_photometry(
                images, self.surveys, ra, dec,
                ref_survey=self.sep_ref_survey, k=self.sep_ellipse_k,
                thresh_sigma=self.sep_thresh_sigma,
                max_match_arcsec=self.sep_max_match_arcsec)

        comp_seds = self._build_seds(photometry, masks_native)

        if self.aperture_mode == "mask":
            # --- Deblending of blended bands (PSF >= separation) ---
            from .deblend_photometry import (deblend_ratio_prior,
                                             deblend_tphot, deblend_combined)
            deblend_ratio = deblend_ratio_prior(photometry)
            deblend_tphot_ = deblend_tphot(images, masks_native, mask_wcs,
                                           self.surveys, photometry=photometry,
                                           ra_center=ra, dec_center=dec,
                                           fit_neighbors=True)
            deblend_comb = deblend_combined(photometry, images, masks_native,
                                            mask_wcs, self.surveys)

            # --- unWISE forced photometry from Legacy Survey DR10 catalog ---
            from .unwise_forced import query_unwise_forced
            unwise_forced = {}
            try:
                unwise_forced = query_unwise_forced(
                    masks_native, mask_wcs, ra, dec,
                )
            except Exception as exc:
                # non-fatal for the target, but recorded so exhausted-retry
                # rate-limit failures are visible for later re-runs
                logger.warning("unWISE forced photometry failed: %s", exc)
                append_failed_target(self.output_dir, target_id,
                                     reason=f"unWISE forced photometry "
                                            f"failed: {exc}",
                                     stage="unwise_forced")
        else:
            # single-aperture modes: no deblending or forced photometry
            deblend_ratio, deblend_tphot_, deblend_comb = {}, {}, {}
            unwise_forced = {}

        # Clean per-component galaxy flux table
        galaxy_table = None
        try:
            galaxy_table = make_galaxy_table(
                target_id, self.surveys, photometry, seg_result,
                deblend_ratio=deblend_ratio, deblend_tphot=deblend_tphot_,
                deblend_comb=deblend_comb, unwise_forced=unwise_forced,
            )
        except Exception as exc:
            logger.error("galaxy table build failed: %s", exc)

        # --- Combined photometry+spectroscopy, only in mode="both" ---
        # The spectral fit describes the TOTAL fibre light of the system,
        # not one component; its synthetic broadband fluxes are normalized
        # to the real imaging photometry (see _combine_phot_spec).
        spectral_result = None
        combined_result = None
        if self.mode == "both":
            if cached_spec is not None:
                # spectral product cached by an earlier spectroscopy pass;
                # resolve the (locally cached) spectrum path for fiber-diam
                # selection and the combined plot, but skip the refit.
                from .spectrum_acquisition import ensure_spectrum
                ensure_spectrum(target, tdir)
                spectral_result = cached_spec
                logger.info("%s: reusing cached spectral product", target_id)
            else:
                spectral_result = self._run_spectroscopy(target, tdir)
            fiber_photometry, fiber_diam = self._measure_fiber_photometry(
                target, images, target_wcs)
            combined_result = self._combine_phot_spec(
                target, photometry, spectral_result,
                fiber_photometry=fiber_photometry,
                fiber_diam_arcsec=fiber_diam)
            if self._cigale_per_target() and combined_result is not None:
                self._run_cigale_for_target(target, tdir, combined_result)

        # --- photometry-only CIGALE fit (photo-z; see _cigale_enabled) ---
        cigale_phot_cols = None
        if self.mode == "photometry" and self._cigale_per_target():
            row = self._cigale_row_photometry(target, photometry)
            cigale_phot_cols = self._run_cigale_for_target(
                target, tdir, row, cigale_mode="photometry")

        self._write_products(prod_dir, target_id, photometry, masks_native,
                             deblend_ratio, deblend_tphot_, deblend_comb,
                             unwise_forced, galaxy_table,
                             spectral_result=spectral_result,
                             combined_result=combined_result)
        if cigale_phot_cols:
            self._append_cigale_columns(prod_dir, target_id, "photometry",
                                        cigale_phot_cols)
        self._make_plots(plot_dir, target_id, target, images_conv, masks_native,
                         masks_reproj, target_wcs, photometry, comp_seds,
                         seg_result, deblend_comb, unwise_forced,
                         deblend_ratio=deblend_ratio,
                         deblend_tphot=deblend_tphot_,
                         galaxy_table=galaxy_table)

        # combined photometry+spectroscopy diagnostic plot (mode="both" only)
        if combined_result is not None:
            try:
                from .combined_plots import plot_combined
                plot_combined(
                    target_id, combined_result,
                    os.path.join(plot_dir, f"{target_id}_combined.png"),
                    spectrum_path=target.get("spectrum_path"),
                )
            except Exception as exc:
                logger.error("combined plot failed: %s", exc)

        logger.info("done %s -> %s", target_id, tdir)
        out = {"photometry": photometry,
               "comp_seds": comp_seds,
               "deblend_ratio": deblend_ratio,
               "deblend_tphot": deblend_tphot_,
               "deblend_combined": deblend_comb,
               "unwise_forced": unwise_forced,
               "galaxy_table": galaxy_table,
               "seg_result": seg_result, "output_dir": tdir}
        if spectral_result is not None:
            out["spectral_result"] = spectral_result
        if combined_result is not None:
            out["combined_result"] = combined_result
        if cigale_phot_cols:
            out["cigale"] = cigale_phot_cols
        return out

    # ------------------------------------------------------------------
    # checkpointing / resume helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cached_product(prod_dir, target_id, kind):
        """Path of an existing final product ({id}_{kind}.ecsv/.csv), or None.

        A product "exists" when the file is present and non-empty; .ecsv is
        preferred (typed round-trip), .csv accepted for older runs.
        """
        for ext in (".ecsv", ".csv"):
            p = os.path.join(prod_dir, f"{target_id}_{kind}{ext}")
            try:
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    return p
            except OSError:
                pass
        return None

    def _load_cached_spectral(self, prod_dir, target_id):
        """Load a cached {id}_spectral table back into a flat dict, or None."""
        path = self._cached_product(prod_dir, target_id, "spectral")
        if path is None:
            return None
        try:
            tbl = Table.read(path)
            if len(tbl) == 0:
                return None
            out = {}
            for col in tbl.colnames:
                v = tbl[0][col]
                if np.ma.is_masked(v):
                    v = np.nan
                elif hasattr(v, "item"):
                    v = v.item()
                out[col] = v
            return out
        except Exception as exc:
            logger.warning("could not read cached spectral product %s: %s",
                           path, exc)
            return None

    @staticmethod
    def _load_cached_photometry(path, surveys):
        """Rebuild the minimal photometry dict from a cached {id}_photometry
        table: {survey: {"total": {flux_mjy, flux_err_mjy}}} — all that
        :meth:`_combine_phot_spec` needs. Returns None on failure."""
        try:
            tbl = Table.read(path)
        except Exception as exc:
            logger.warning("could not read cached photometry product %s: %s",
                           path, exc)
            return None
        phot = {}
        for row in tbl:
            survey = str(row["survey"])
            if survey not in surveys:
                continue
            entry = {}
            for key, col in (("flux_mjy", "total_flux_mjy"),
                             ("flux_err_mjy", "total_flux_err_mjy")):
                v = row[col] if col in tbl.colnames else np.nan
                if np.ma.is_masked(v):
                    v = np.nan
                entry[key] = float(v)
            phot[survey] = {"total": entry,
                            "blend_flag": bool(row["blend_flag"])
                            if "blend_flag" in tbl.colnames else False}
        return phot or None

    def _run_both_from_cache(self, target, tdir, prod_dir, plot_dir,
                             cache_dir, phot_path, cached_spec):
        """mode="both" fast path: reuse cached photometry (+spectral) products.

        Only the fiber-aperture measurement (from locally cached FITS
        cutouts), the spectral fit IF not already cached, and the
        combination step are executed — no deblending/SED recompute.
        """
        target_id = target["id"]
        logger.info("%s: combining from cached products (photometry%s)",
                    target_id,
                    "+spectral" if cached_spec is not None else " only")

        photometry = self._load_cached_photometry(phot_path, self.surveys)
        if photometry is None:
            logger.warning("%s: cached photometry unreadable; falling back "
                           "to full recompute is NOT done here — combining "
                           "with empty photometry", target_id)
            photometry = {}

        # spectral part: reuse or compute (and persist) now
        if cached_spec is not None:
            from .spectrum_acquisition import ensure_spectrum
            ensure_spectrum(target, tdir)  # local cache hit; sets source/path
            spectral_result = cached_spec
        else:
            spectral_result = self._run_spectroscopy(target, tdir)
            self._write_spectral_products(prod_dir, target_id,
                                          spectral_result)

        # fiber-matched photometry from the cached image cutouts (fast:
        # hostphot_downloader reads {tdir}/cache/*.fits, no network unless
        # a band's cache is missing)
        images = hostphot_downloader(target["ra"], target["dec"],
                                     self.surveys, self.download_size,
                                     cache_dir)
        target_wcs = build_common_wcs(target["ra"], target["dec"],
                                      self.grid_pixscale, self.grid_size)
        fiber_photometry, fiber_diam = self._measure_fiber_photometry(
            target, images, target_wcs)

        combined_result = self._combine_phot_spec(
            target, photometry, spectral_result,
            fiber_photometry=fiber_photometry,
            fiber_diam_arcsec=fiber_diam)
        if self._cigale_per_target() and combined_result is not None:
            self._run_cigale_for_target(target, tdir, combined_result)
        self._write_combined_products(prod_dir, target_id, combined_result)

        try:
            from .combined_plots import plot_combined
            plot_combined(
                target_id, combined_result,
                os.path.join(plot_dir, f"{target_id}_combined.png"),
                spectrum_path=target.get("spectrum_path"),
            )
        except Exception as exc:
            logger.error("combined plot failed: %s", exc)

        logger.info("done %s (from cache) -> %s", target_id, tdir)
        out = {"output_dir": tdir, "combined_result": combined_result,
               "resumed_from_cache": True}
        if spectral_result is not None:
            out["spectral_result"] = spectral_result
        return out

    def _check_download_failures(self, target_id, images):
        """Refuse to silently degrade science when downloads failed.

        If the Legacy optical imaging that drives xdebpair/xmask deblending
        failed to DOWNLOAD (rate limit, network — i.e. an "error" entry from
        the downloader, avoidable by re-running), the deblending would fall
        through to the synthetic circular stub mask and produce a fake
        science result nobody would notice. Instead: append the target to
        {output_dir}/failed_targets.jsonl and raise DownloadError so the
        driver marks the target FAILED and it can be re-run later.

        Partial failures of non-critical bands (e.g. one WISE band) are
        logged to the manifest but do NOT abort the target. Genuine
        stub/single-component behavior with good imaging is untouched.
        """
        failed = {s: e.get("error", "download failed")
                  for s, e in (images or {}).items()
                  if isinstance(e, dict) and e.get("data") is None
                  and "error" in e}
        if not failed:
            return

        legacy_requested = [s for s in self.surveys if s.startswith("Legacy")]
        legacy_failed = [s for s in legacy_requested if s in failed]

        # fatal: the deblending mask cannot be built from real imaging
        # (xdebpair needs Legacy_r; all-optical loss = guaranteed stub)
        fatal = ("Legacy_r" in legacy_failed or
                 (legacy_requested and
                  len(legacy_failed) == len(legacy_requested)))
        if fatal:
            reason = ("Legacy optical imaging download failed "
                      f"({sorted(legacy_failed)}); refusing stub-mask "
                      "fallback — re-run this target")
            append_failed_target(self.output_dir, target_id, reason,
                                 stage="download", details=failed)
            logger.error("%s: %s", target_id, reason)
            raise DownloadError(f"{target_id}: {reason}")

        # non-fatal: record partial band loss for later re-runs, continue
        append_failed_target(self.output_dir, target_id,
                             reason=f"partial band download failure: "
                                    f"{sorted(failed)}",
                             stage="download_partial", details=failed)
        logger.warning("%s: partial download failure in bands %s (recorded "
                       "in failed_targets.jsonl, continuing)",
                       target_id, sorted(failed))

    def _run_spectroscopy(self, target, tdir):
        """Resolve the spectrum (SDSS by coordinates if needed) and fit it.

        Never raises: returns the flat spectral-result dict, or ``None`` when
        no spectrum could be obtained or the fit failed — the caller/other
        pipeline stages must survive missing spectroscopy.
        """
        target_id = target.get("id")
        if target.get("z") is None:
            logger.error("target %s has no redshift; skipping spectral fit",
                         target_id)
            return None

        # spectrum_path missing (or stale)? try coordinate-based SDSS lookup,
        # cached under {output}/{id}/spectroscopy/.
        from .spectrum_acquisition import ensure_spectrum
        spec_path = ensure_spectrum(target, tdir)
        if spec_path is None:
            logger.error("no spectrum available for %s; skipping spectral fit",
                         target_id)
            return None

        from .spectral_fit import run_spectral_fit
        result = run_spectral_fit(target, survey_filters=self.survey_filters,
                                  fit_agn=self.fit_agn,
                                  spectrum_fwhm=self.spectrum_fwhm)
        if result is None:
            logger.error("spectral fit produced no result for %s", target_id)
        return result

    def _measure_fiber_photometry(self, target, images, target_wcs):
        """Fiber-equivalent aperture photometry on the target position.

        Picks the fiber diameter from the spectrum source resolved by
        :func:`ensure_spectrum` (SDSS 2.5\", DESI 1.5\"; inferred from the
        spectrum path when the source tag is missing) and measures every
        configured band. Never raises: returns ``({}, diam)`` on failure so
        :meth:`_combine_phot_spec` can fall back to the total aperture.
        """
        from .photometry import (measure_fiber_aperture_photometry,
                                 FIBER_DIAM_ARCSEC)
        from .spectrum_acquisition import _infer_spectrum_source

        source = target.get("spectrum_source")
        if source not in FIBER_DIAM_ARCSEC:
            source = _infer_spectrum_source(target.get("spectrum_path"))
        diam = FIBER_DIAM_ARCSEC.get(source, FIBER_DIAM_ARCSEC["desi"])
        if source not in FIBER_DIAM_ARCSEC:
            logger.warning("unknown spectrum source for %s; assuming DESI "
                           "fiber (%.1f\")", target.get("id"), diam)

        try:
            fiber = measure_fiber_aperture_photometry(
                images, target_wcs, target["ra"], target["dec"], diam,
                self.surveys, pixscale_arcsec=self.grid_pixscale)
        except Exception as exc:
            logger.warning("fiber-aperture photometry failed for %s: %s",
                           target.get("id"), exc)
            fiber = {}
        return fiber, diam

    # reference bands for the spectro->photo normalization, in order of
    # preference. Legacy_r is the only band that both the imaging pipeline
    # and the spectral synthetic photometry reliably cover (SDSS spectra
    # span ~3800-9200 A, so g and r are fully covered; z only partially).
    # Preferred single-band baseline for the historical Zou+2024 diagnostic
    # (used only if present among the collected anchors; otherwise the first
    # anchor is used). NOT a ceiling on which bands may anchor the fit.
    _SINGLE_BAND_PREF = "Legacy_r"

    @staticmethod
    def _empty_norm_columns():
        """Default (NaN/empty) values for every normalization column, so the
        schema is identical whether or not a spectrum / anchor is available."""
        return {
            "norm_band": "", "norm_factor": np.nan, "norm_aperture": "",
            "norm_factor_fiber": np.nan, "norm_factor_grey": np.nan,
            "norm_factor_single": np.nan, "norm_band_single": "",
            "norm_dispersion_pct": np.nan, "norm_n_bands": 0,
            "norm_poly_degree": -1, "norm_poly_bic": np.nan,
            "norm_poly_chi2": np.nan, "norm_poly_cond": np.nan,
            "norm_poly_n_anchors": 0, "norm_poly_n_clipped": 0,
            "norm_poly_lam0": np.nan, "norm_poly_coeffs": "",
        }

    def _combine_phot_spec(self, target, photometry, spectral_result,
                           fiber_photometry=None, fiber_diam_arcsec=None):
        """Merge imaging photometry with the spectral fit into one flat dict.

        Method (extension of Zou et al. 2024, ApJ 961, 173, Sects. 3.2-3.3):
        the fibre spectrum suffers aperture losses, so the imaging
        photometry is taken as the absolute flux reference and the spectrum
        is corrected to TOTAL light. The paper uses a single grey (r-band)
        factor; the multi-band grey average (S/N^2-weighted over all usable
        reference bands) is kept here only as a DIAGNOSTIC. The APPLIED
        correction is now WAVELENGTH-DEPENDENT: a smooth
        ``ln(scale(lambda)) = c0 + c1*x + c2*x**2`` (``x = ln(lambda/lambda0)``)
        fitted to the per-band ratios ``F_phot_total(b)/F_synth(b)``, degree
        auto-selected (BIC + leave-one-out CV, sigma-clipping) so it never
        over-reaches with few anchors (method in
        :mod:`.spec_normalization`, adapted from the CIGALE cigale2s branch's
        ``normalize_spec_to_phot``). Each synthetic filter flux and each
        emission-line flux is scaled by ``scale(lambda)`` evaluated at ITS
        OWN wavelength (band effective wavelength / line observed wavelength),
        capturing a differential colour/aperture mismatch a grey factor
        cannot.

        The anchor bands are GENERIC over the configured imaging surveys
        (``self.surveys`` x ``SURVEY_DEFAULTS[...]['lambda_eff']``): any band
        whose effective wavelength is inside the observed-spectrum coverage
        and which has both a total-aperture flux and a synthetic flux becomes
        an anchor automatically -- Legacy g/r/z today, SDSS/HSC/S-PLUS/J-PLUS
        when added to the config, no code change. With only ~3 optical anchors
        available today the degree is realistically only ever 0 or 1 (the
        thresholds correctly gate against degree 2 without enough data).

        Flagged ``norm_aperture="total"`` with ``norm_band`` listing the
        anchor bands. If no total-aperture band is usable it falls back to
        the fiber-matched fluxes (spectrophotometric recalibration only, no
        aperture correction), flagged ``norm_aperture="fiber_fallback"``.

        Normalization columns:

        - ``norm_factor`` (PRIMARY, applied): the wavelength-dependent scale
          evaluated at the anchor reference wavelength ``lambda0`` (a single
          representative scalar, ``= exp(c0)``), kept scalar for backward
          compatibility with existing consumers and with ``stellar_mass_total``.
          The per-band ``scaled_*`` columns each carry their OWN
          wavelength-appropriate scale, so ``norm_factor`` is exactly the
          applied factor only for degree 0.
        - ``norm_factor_grey``: the OLD multi-band S/N^2-weighted grey average
          (previous ``norm_factor``), kept for comparison.
        - ``norm_factor_single``/``norm_band_single``: the OLD single-band grey
          factor (Legacy_r if present, else the first anchor -- the literal
          Zou et al. 2024 baseline), kept for comparison with historical runs.
        - ``norm_poly_degree``: chosen polynomial degree (0=grey, 1=slope,
          2=curvature).
        - ``norm_poly_bic``/``norm_poly_chi2``/``norm_poly_cond``: fit
          diagnostics of the chosen model.
        - ``norm_poly_n_anchors``/``norm_poly_n_clipped``: anchors kept /
          sigma-clipped.
        - ``norm_poly_lam0``/``norm_poly_coeffs``: reference wavelength (AA)
          and log-space coefficients.
        - ``norm_dispersion_pct``: relative spread 100*(max-min)/mean of
          the per-band grey factors; values above ~15-20% flag a strong colour
          gradient or AGN contamination.
        - ``norm_n_bands``: number of anchor bands combined.

        The fiber-matched imaging photometry (2.5\" SDSS / 1.5\" DESI,
        ``fiber_photometry``) is still always measured and kept as
        DIAGNOSTIC columns (``phot_fiber_*``), and the fiber-based
        recalibration factor is reported as ``norm_factor_fiber`` for
        comparison — it no longer drives the applied ``norm_factor``.

        Stellar mass: XpectraFit's ``stellar_mass``/``stellar_mass_aperture``
        are LINEAR fibre-aperture masses (its own total-flux correction,
        ``stellar_mass_corrected``, is never applied for lack of imaging).
        In Zou et al. the spectrum is rescaled BEFORE the fit, so their
        masses are total-aperture; since mass scales linearly with flux at
        fixed M/L, rescaling after the fact is equivalent, and we export
        ``stellar_mass_total = stellar_mass * norm_factor`` (the scalar
        ``norm_factor = scale(lambda0)``, i.e. the correction at the anchor
        reference wavelength — a single representative aperture factor, as a
        mass is not itself a spectrum). The original fibre-aperture
        ``stellar_mass``/``stellar_mass_aperture`` columns are kept
        unchanged for diagnostics.

        Deliberately NOT rescaled: M_BH estimates (nonlinear in luminosity,
        and the broad-line region is unresolved — the fibre already captures
        the nucleus, so a galaxy-wide aperture ratio would over-correct),
        sigma, Dn4000, fAGN and the classification labels (flux-ratio
        quantities, aperture-independent).
        """
        combined = {
            "target_id": str(target.get("id")),
            "ra": target.get("ra"), "dec": target.get("dec"),
            "z": target.get("z"),
        }

        # real imaging photometry (total = whole system, matching the fibre
        # target better than any single deblended component)
        for survey in self.surveys:
            m = (photometry or {}).get(survey, {}).get("total", {})
            if not isinstance(m, dict):
                m = {}
            combined[f"phot_{survey}_flux_mjy"] = m.get("flux_mjy", np.nan)
            combined[f"phot_{survey}_flux_err_mjy"] = m.get("flux_err_mjy", np.nan)

        # fiber-matched imaging photometry (aperture = spectrograph fiber)
        fiber_photometry = fiber_photometry or {}
        combined["fiber_diam_arcsec"] = (
            float(fiber_diam_arcsec) if fiber_diam_arcsec else np.nan)
        combined["spectrum_source"] = target.get("spectrum_source", "") or ""
        for survey in self.surveys:
            fm = fiber_photometry.get(survey, {})
            if not isinstance(fm, dict):
                fm = {}
            combined[f"phot_fiber_{survey}_flux_mjy"] = fm.get("flux_mjy", np.nan)
            combined[f"phot_fiber_{survey}_flux_err_mjy"] = fm.get("flux_err_mjy",
                                                                   np.nan)

        if spectral_result is None:
            combined["spec_status"] = "missing"
            combined.update(self._empty_norm_columns())
            return combined
        combined["spec_status"] = spectral_result.get("status", "ok")

        # key spectral-fit outputs (fibre-aperture; see docstring)
        for key in ("fAGN", "spectral_class", "spec_class", "bpt_class",
                    "whan_class", "Dn4000", "stellar_mass",
                    "stellar_mass_aperture", "log_MBH_Halpha",
                    "log_MBH_Hbeta", "sigma", "sigma_err", "chi2_reduced",
                    "agn_dominated", "synth_photometry_source"):
            if key in spectral_result:
                combined[key] = spectral_result[key]

        synth_bands = sorted(
            k[len("synth_"):-len("_flux_mjy")]
            for k in spectral_result
            if k.startswith("synth_") and k.endswith("_flux_mjy")
            and not k.endswith("_flux_err_mjy"))

        # --- Normalization -------------------------------------------------
        # PRIMARY: wavelength-dependent scale(lambda) fitted to the
        # TOTAL-aperture imaging vs synthetic fluxes (true aperture correction
        # to total light). Anchors are generic over self.surveys (see
        # spec_normalization.collect_anchors + docstring). Fallback: the
        # fiber-matched fluxes (spectrophotometric recalibration only).
        lambda_eff_um = {s: SURVEY_DEFAULTS.get(s, {}).get("lambda_eff")
                         for s in self.surveys}
        wmin = spectral_result.get("spec_wave_min_aa")
        wmax = spectral_result.get("spec_wave_max_aa")

        def _grey(anchors):
            """S/N^2-weighted grey diagnostic over anchors:
            (grey, band_label, single, single_band, disp_pct, n)."""
            if not anchors:
                return np.nan, "", np.nan, "", np.nan, 0
            names = [a.name for a in anchors]
            vals = np.array([a.phot / a.synth for a in anchors], float)
            wts = np.array([(a.phot / a.phot_err) ** 2
                            if np.isfinite(a.phot_err) and a.phot_err > 0
                            else np.nan for a in anchors], float)
            if np.isfinite(wts).any() and np.nansum(wts) > 0:
                good = np.isfinite(wts)
                grey = float(np.average(vals[good], weights=wts[good]))
            else:
                grey = float(vals.mean())
            if self._SINGLE_BAND_PREF in names:
                si = names.index(self._SINGLE_BAND_PREF)
            else:
                si = 0
            single, single_band = float(vals[si]), names[si]
            disp = (100.0 * (vals.max() - vals.min()) / vals.mean()
                    if len(vals) >= 2 else np.nan)
            return grey, "+".join(names), single, single_band, disp, len(names)

        anchors = collect_anchors(self.surveys, lambda_eff_um, combined,
                                  spectral_result, wmin, wmax, "phot_")
        norm_aperture = "total" if anchors else ""
        if not anchors:  # fall back to the fiber-matched aperture
            anchors = collect_anchors(self.surveys, lambda_eff_um, combined,
                                      spectral_result, wmin, wmax,
                                      "phot_fiber_")
            if anchors:
                norm_aperture = "fiber_fallback"
                logger.warning("total-aperture normalization unavailable for "
                               "%s; falling back to fiber-matched flux (no "
                               "aperture correction applied)", target.get("id"))

        # diagnostic fiber-matched grey factor, always computed when possible
        fiber_anchors = collect_anchors(self.surveys, lambda_eff_um, combined,
                                        spectral_result, wmin, wmax,
                                        "phot_fiber_")
        norm_fiber = _grey(fiber_anchors)[0]

        spec_norm, meta = fit_spec_normalization(anchors)
        grey, norm_band, norm_single, norm_single_band, norm_disp, norm_nb = \
            _grey(anchors)

        # PRIMARY applied scalar: scale(lambda0) = exp(c0) (representative,
        # backward-compatible). Per-band scaled_* columns get their own
        # wavelength-appropriate scale below.
        norm = spec_norm.scale_at(meta["lam0"]) if spec_norm is not None \
            else np.nan
        combined.update(self._empty_norm_columns())
        combined["norm_band"] = norm_band
        combined["norm_factor"] = norm
        combined["norm_aperture"] = norm_aperture
        combined["norm_factor_fiber"] = norm_fiber
        combined["norm_factor_grey"] = grey
        combined["norm_factor_single"] = norm_single
        combined["norm_band_single"] = norm_single_band
        combined["norm_dispersion_pct"] = norm_disp
        combined["norm_n_bands"] = norm_nb
        combined["norm_poly_degree"] = meta["degree"]
        combined["norm_poly_bic"] = meta["bic"]
        combined["norm_poly_chi2"] = meta["chi2"]
        combined["norm_poly_cond"] = meta["cond"]
        combined["norm_poly_n_anchors"] = meta["n_anchors"]
        combined["norm_poly_n_clipped"] = meta["n_clipped"]
        combined["norm_poly_lam0"] = meta["lam0"]
        combined["norm_poly_coeffs"] = ",".join(f"{c:.6g}"
                                                for c in meta["coeffs"])
        if not np.isfinite(norm):
            logger.warning("no valid normalization band for %s; synthetic "
                           "fluxes left unscaled", target.get("id"))

        def _scale_at(lam_aa):
            """Wavelength-dependent factor at lam_aa (AA), or the scalar
            norm_factor when the wavelength is unknown / no fit exists."""
            if spec_norm is None:
                return np.nan
            if not (np.isfinite(lam_aa) if lam_aa is not None else False):
                return norm
            return float(spec_norm.scale_at(float(lam_aa)))

        # total-aperture stellar mass (paper-equivalent; see docstring). A
        # mass is not a spectrum, so the single scalar norm_factor is used.
        sm = combined.get("stellar_mass", np.nan)
        try:
            sm = float(sm)
        except (TypeError, ValueError):
            sm = np.nan
        combined["stellar_mass_total"] = (
            sm * norm if np.isfinite(sm) and np.isfinite(norm)
            and norm_aperture == "total" else np.nan)

        # pre- and post-scale synthetic fluxes: each band scaled at its own
        # effective wavelength (wavelength-dependent correction).
        for band in synth_bands:
            fs = spectral_result.get(f"synth_{band}_flux_mjy", np.nan)
            fe = spectral_result.get(f"synth_{band}_flux_err_mjy", np.nan)
            lam_um = SURVEY_DEFAULTS.get(band, {}).get("lambda_eff")
            sc = _scale_at(lam_um * 1e4 if lam_um is not None else np.nan)
            combined[f"synth_{band}_flux_mjy"] = fs
            combined[f"synth_{band}_flux_err_mjy"] = fe
            combined[f"scaled_{band}_flux_mjy"] = (
                fs * sc if np.isfinite(sc) else np.nan)
            combined[f"scaled_{band}_flux_err_mjy"] = (
                fe * sc if np.isfinite(sc) else np.nan)

        # per-line EW / velocity dispersion / velocity / S/N columns from the
        # expanded spectral extraction (already named line_*/broadline_*):
        # copied verbatim — these are flux-RATIO or kinematic quantities,
        # NOT rescaled by the aperture correction.
        for k, v in spectral_result.items():
            if k.startswith("line_") or k.startswith("broadline_"):
                combined[k] = v

        # emission-line fluxes: each line scaled at its own observed
        # wavelength; pre-scale copies kept.
        z_line = combined.get("z", spectral_result.get("z"))
        for k in list(spectral_result):
            if k.startswith("flux_") and not k.startswith("flux_err_"):
                line = k[len("flux_"):]
                fv = spectral_result.get(f"flux_{line}", np.nan)
                fe = spectral_result.get(f"flux_err_{line}", np.nan)
                combined[f"line_{line}_flux"] = fv
                combined[f"line_{line}_flux_err"] = fe
                sc = _scale_at(line_observed_wavelength_aa(line, z_line))
                try:
                    combined[f"line_{line}_flux_scaled"] = (
                        float(fv) * sc if np.isfinite(sc) else np.nan)
                    combined[f"line_{line}_flux_err_scaled"] = (
                        float(fe) * sc if np.isfinite(sc) else np.nan)
                except (TypeError, ValueError):
                    combined[f"line_{line}_flux_scaled"] = np.nan
                    combined[f"line_{line}_flux_err_scaled"] = np.nan

        return combined

    # ------------------------------------------------------------------
    # CIGALE (cigale_run enables the SED fit for any mode)
    # ------------------------------------------------------------------

    def _cigale_enabled(self):
        """CIGALE fitting requested for the current mode.

        ``cigale_run`` is the single switch, uniform across all three modes.
        In mode="both" the built-in ``_combine_phot_spec`` combination always
        runs regardless; CIGALE is an ADDITIONAL optional fit on top.
        """
        return self.cigale_run

    def _cigale_per_target(self):
        """True when CIGALE should run per target, inside :meth:`run`.

        ``cigale_batch=True`` disables the per-target fit even when
        CIGALE is enabled: the driver must call :meth:`run_cigale_batch`
        manually after the whole run.
        """
        return self._cigale_enabled() and not self.cigale_batch

    @staticmethod
    def _product_has_cigale(path):
        """True when the product at ``path`` already carries cigale_* columns
        (skip-check so CIGALE is never rerun on an already-fitted product)."""
        try:
            tbl = Table.read(path)
            return any(c.startswith("cigale_") for c in tbl.colnames)
        except Exception:
            return False

    def _append_cigale_columns(self, prod_dir, tid, kind, cols):
        """Append (broadcast) cigale_* scalar columns to a {tid}_{kind}
        product and rewrite its .csv + .ecsv."""
        path = self._cached_product(prod_dir, tid, kind)
        if path is None or not cols:
            return
        try:
            tbl = Table.read(path)
            for k, v in cols.items():
                tbl[k] = v  # scalar broadcasts over all rows
            base = os.path.join(prod_dir, f"{tid}_{kind}")
            tbl.write(base + ".csv", format="csv", overwrite=True)
            tbl.write(base + ".ecsv", format="ascii.ecsv", overwrite=True)
        except Exception as exc:
            logger.error("cigale: could not update %s: %s", path, exc)

    def _cigale_row_photometry(self, target, photometry):
        """Flat phot_* row for a photometry-only CIGALE fit.

        No spectrum and NO redshift: the row's z stays blank so CIGALE
        chi2-scans the redshifting module's z grid (photo-z; the grid must
        be hand-edited into the packaged pcigale.ini)."""
        row = {"target_id": str(target.get("id"))}
        for survey in self.surveys:
            m = (photometry or {}).get(survey, {}).get("total", {})
            if not isinstance(m, dict):
                m = {}
            row[f"phot_{survey}_flux_mjy"] = m.get("flux_mjy", np.nan)
            row[f"phot_{survey}_flux_err_mjy"] = m.get("flux_err_mjy",
                                                       np.nan)
        return row

    def _run_cigale_for_target(self, target, tdir, result_row,
                               cigale_mode="both"):
        """Single-target CIGALE fit; appends cigale_* columns in place.

        cigale_mode selects the input-row shape (see
        :func:`.cigale_integration.build_spectro_row`):

        - "both" (default): fixed z + broadband + spectrum — the validated
          combined method, unchanged.
        - "spectroscopy": fixed z + spectrum only (no broadband columns).
          Spectrum-only fits are less constrained than combined fits.
        - "photometry": broadband only, redshift left blank — CIGALE's
          photo-z over the hand-configured pcigale.ini z grid; the result
          carries the extra ``cigale_z_phot`` column(s).

        Uses the batch machinery in :mod:`.cigale_integration` with a
        one-row input. When processing MANY targets, prefer
        :meth:`run_cigale_batch` after the per-target stages — one
        ``pcigale run`` over one multi-row input file is much faster than
        N single-target runs (CIGALE reuses its model grid across rows).
        Never raises: on failure the product simply carries no
        cigale_* columns (a warning is logged).
        """
        tid = str(target["id"])
        try:
            from .cigale_integration import run_cigale_for_targets
            npz = None
            if cigale_mode != "photometry":
                npz = os.path.join(tdir, "spectroscopy", f"desi_{tid}.npz")
            work = os.path.join(tdir, "cigale")
            results = run_cigale_for_targets(
                [(tid, dict(result_row), npz)], work, mode=cigale_mode)
            cols = results.get(tid)
            if cols:
                result_row.update(cols)
                logger.info("%s: CIGALE columns appended (chi2_red=%.2f)",
                            tid, cols.get("cigale_chi2_red", np.nan))
            else:
                logger.warning("%s: CIGALE produced no result row", tid)
            return cols
        except Exception as exc:
            logger.error("CIGALE fit failed for %s: %s", tid, exc)
            return None

    # mode -> per-target product kind read/updated by run_cigale_batch
    _CIGALE_PRODUCT_KIND = {"photometry": "photometry",
                            "spectroscopy": "spectral",
                            "both": "combined"}

    def run_cigale_batch(self, targets, work_dir=None, conda_env="cigale",
                         force=False):
        """Batch CIGALE fit over targets with cached per-mode products.

        For every target whose product for ``self.mode`` exists under
        ``self.output_dir`` (``{id}_photometry`` for mode="photometry",
        ``{id}_spectral`` for "spectroscopy", ``{id}_combined`` for
        "both"; the two spectrum-using modes also need the cached DESI
        spectrum), builds ONE ``cigale_input.txt``, runs ONE ``pcigale
        run`` (much faster than per-target runs — the model grid is
        reused across rows), and rewrites each product with the
        ``cigale_*`` columns appended. Returns {target_id: cigale_cols}.

        Input rows follow the mode (see
        :func:`.cigale_integration.build_spectro_row`): no broadband
        columns for spectroscopy-only (spectrum-only fits are less
        constrained than combined fits); no redshift for photometry-only
        (CIGALE photo-z over the z grid hand-edited into the packaged
        pcigale.ini; best-fit z returned as ``cigale_z_phot``).

        IMPORTANT: this method is NEVER called automatically. With config
        ``cigale_batch: true`` :meth:`run` skips the per-target CIGALE
        fit, and your driver script MUST call ``run_cigale_batch(targets)``
        itself once after the whole run over all targets — the pipeline
        cannot detect a forgotten call.

        Targets whose product already carries cigale_* columns are
        skipped (checkpointing); pass ``force=True`` to refit them.
        """
        from .cigale_integration import (run_cigale_for_targets,
                                         read_combined_row)
        if work_dir is None:
            work_dir = os.path.join(self.output_dir, "cigale_batch")
        kind = self._CIGALE_PRODUCT_KIND[self.mode]
        need_spectrum = self.mode != "photometry"

        specs = []
        for t in targets:
            tid = str(t["id"])
            prod_dir = os.path.join(self.output_dir, tid, "products")
            npz = os.path.join(self.output_dir, tid, "spectroscopy",
                               f"desi_{tid}.npz")
            if self.mode == "both":
                # combined rows are read as plain string dicts from the csv
                prod = os.path.join(prod_dir, f"{tid}_combined.csv")
                if not os.path.exists(prod):
                    prod = None
            else:
                prod = self._cached_product(prod_dir, tid, kind)
            if prod is None or (need_spectrum and not os.path.exists(npz)):
                logger.warning("cigale batch: %s missing %s product%s; "
                               "skipped", tid, kind,
                               " or spectrum" if need_spectrum else "")
                continue
            if not force and self._product_has_cigale(prod):
                logger.info("cigale batch: %s already has cigale_* columns;"
                            " skipped (force=True to refit)", tid)
                continue
            if self.mode == "photometry":
                phot = self._load_cached_photometry(prod, self.surveys)
                row = self._cigale_row_photometry(t, phot or {})
            elif self.mode == "spectroscopy":
                row = self._load_cached_spectral(prod_dir, tid) or {}
                row["z"] = t.get("z")
            else:
                row = read_combined_row(prod)
            specs.append((tid, row, npz))

        results = run_cigale_for_targets(specs, work_dir,
                                         conda_env=conda_env,
                                         mode=self.mode)

        # append the cigale_* columns to each target's product
        for tid, cols in results.items():
            prod_dir = os.path.join(self.output_dir, tid, "products")
            self._append_cigale_columns(prod_dir, tid, kind, cols)
        return results

    def _run_deblending(self, ra, dec, images):
        """Try xdebpair → xmask → stub, returning a compatible result object."""
        if self.use_xdebpair:
            try:
                from .deblending import XdebPairAdapter
                result = XdebPairAdapter(ra, dec, images).run()
                logger.info("xdebpair: %s", result)
                return result
            except Exception as exc:
                logger.error("xdebpair failed (%s); falling back to xmask/stub", exc)

        return self._run_xmask(ra, dec)

    def _run_xmask(self, ra, dec):
        try:
            return self._XmaskPy(ra=ra, dec=dec, size_arcmin=self.download_size).run()
        except Exception as exc:
            logger.error("xmask failed (%s); falling back to stub", exc)
            return _StubXmask(ra, dec, self.download_size).run()

    # target CSV "type" values that mean "keep the components separate";
    # anything else (post_merger, typos, blank) counts as "other" = single.
    _SEPARATE_TYPES = ("merger", "pre_merger")

    def _apply_separation_policy(self, seg_result, target):
        """Reconcile xdebpair's raw component count with the user's intent.

        Applies the ``photometry.separation`` config (central/total/pair)
        and the OPTIONAL per-target ``type`` column of the input CSV.
        xdebpair always runs first (we need its component detection either
        way); this only decides how its masks are turned into apertures.

        Why classification wins over xdebpair (``pair`` mode): some
        galaxies that xdebpair splits into 2 components are, per the user's
        visual/catalog classification, a SINGLE galaxy (e.g. a post-merger
        with a double nucleus, or a clumpy disc). When the target carries a
        ``type`` value, that classification is trusted over xdebpair's
        pixel-level decision.

        Effective behavior (per-target ``type`` compared case-insensitively):

        separation="pair" (default — separate, but respect classification):
          * type is merger/pre_merger  -> keep xdebpair's N components.
          * type is post_merger or anything unrecognized -> force ONE
            component: the UNION of all detected masks (the "two"
            components are really one galaxy, so all its light is kept).
          * target has NO type key at all -> xdebpair's raw decision
            stands, no override.

        separation="central" (central galaxy only):
          * always exactly ONE component: the FIRST mask (gal1, xdebpair's
            primary/central detection). A detected companion's flux is
            DISCARDED, and the output table carries
            has_companion_not_measured=True plus a warning log so nobody
            misses that a companion exists but was not measured.

        separation="total" (integrated system light):
          * always exactly ONE aperture: the UNION of all detected masks.
            Unlike "central" the companion's flux IS included — one
            blended, system-total measurement, no per-component detail.

        The seg_result object is annotated in place with
        separation_policy / target_type / n_components_detected /
        has_companion_not_measured (consumed by make_galaxy_table).
        """
        masks = dict(getattr(seg_result, "masks", {}) or {})
        n_detected = len(masks)
        raw_type = target.get("type")
        ttype = str(raw_type).strip().lower() if raw_type is not None else None

        seg_result.separation_policy = self.separation
        seg_result.target_type = ttype if ttype else ""
        seg_result.n_components_detected = n_detected
        seg_result.has_companion_not_measured = False

        def _union():
            u = None
            for m in masks.values():
                if m is None:
                    continue
                u = m.copy() if u is None else (u | m)
            return u

        def _force_single(mask, reason):
            seg_result.masks = {"gal1": mask}
            seg_result.n_components = 1
            seg_result.is_merger = False
            # one aperture -> no close-pair blending concept
            seg_result.separation_arcsec = 0.0
            logger.info("separation policy: %s -> single component (%s; "
                        "xdebpair detected %d)", self.separation, reason,
                        n_detected)

        if self.separation == "central":
            if n_detected > 1:
                seg_result.has_companion_not_measured = True
                logger.warning(
                    "separation='central': xdebpair detected %d components "
                    "but only the central galaxy (gal1) is measured; the "
                    "companion's flux is NOT included "
                    "(has_companion_not_measured=True)", n_detected)
            first_key = next(iter(masks), "gal1")
            _force_single(masks.get(first_key), "central galaxy only")
            return seg_result

        if self.separation == "total":
            _force_single(_union(), "system-total aperture")
            return seg_result

        # separation == "pair"
        if ttype is None:
            # no "type" column in the target CSV: xdebpair decides
            return seg_result
        if ttype in self._SEPARATE_TYPES:
            return seg_result  # classified as (pre-)merger: keep components
        # post_merger / unrecognized classification: one galaxy, keep all
        # its light (classification wins over xdebpair's raw count)
        _force_single(_union(), f"type={ttype!r} treated as single")
        return seg_result

    def _run_aperture_detection(self, target, images):
        """Build the single-component seg result for the non-mask modes.

        aperture_mode="aperture": circular aperture of
        ``photometry.aperture_radius_arcsec`` at the target position.
        aperture_mode="sep_apertures": SEP-detected ellipse (a, b, theta)
        scaled by SEP_ELLIPSE_K (see photometry.detect_sep_ellipse).

        The aperture footprint is materialized as a boolean mask on the
        Legacy_r native grid (or the common grid when Legacy_r is missing)
        so plots and the galaxy table work exactly as in mask mode.
        """
        from .photometry import (detect_sep_ellipse, sep_ellipse_mask,
                                 circular_mask)
        from astropy.wcs.utils import proj_plane_pixel_scales

        ra, dec = target["ra"], target["dec"]
        ref = images.get("Legacy_r", {})
        ref_data, ref_wcs = ref.get("data"), ref.get("wcs")
        if ref_data is None or ref_wcs is None:
            ref_wcs = build_common_wcs(ra, dec, self.grid_pixscale,
                                       self.grid_size)
            ref_shape = (self.grid_size, self.grid_size)
            ps = self.grid_pixscale
        else:
            ref_shape = ref_data.shape
            ps = float(np.abs(proj_plane_pixel_scales(ref_wcs)).mean() * 3600.0)

        mask = None
        classification = self.aperture_mode
        if self.aperture_mode == "sep_apertures":
            ellipse = detect_sep_ellipse(
                images, ra, dec, ref_survey=self.sep_ref_survey,
                thresh_sigma=self.sep_thresh_sigma,
                max_match_arcsec=self.sep_max_match_arcsec)
            if ellipse is not None:
                mask = sep_ellipse_mask(ellipse, ref_wcs, ref_shape,
                                        k=self.sep_ellipse_k)
        if mask is None:  # circular aperture (also SEP-detection fallback)
            if self.aperture_mode == "sep_apertures":
                logger.warning("sep_apertures: SEP detection failed; "
                               "falling back to a circular aperture")
                classification = "sep_apertures_fallback_circular"
            cx, cy = ref_wcs.world_to_pixel_values(ra, dec)
            mask = circular_mask(ref_shape, (float(cx), float(cy)),
                                 self.aperture_radius_arcsec / ps)

        seg = _StubXmaskResult({"gal1": mask}, ref_wcs, 1, 0.0)
        seg.classification = classification
        return seg

    def _build_seds(self, photometry, masks_native):
        """Build SED point lists per component from the isolated photometry."""
        comp_names = list(masks_native.keys()) + ["total"]
        seds = {c: [] for c in comp_names}

        for survey in self.surveys:
            meta = SURVEY_DEFAULTS[survey]
            lam = meta["lambda_eff"]
            phot = photometry.get(survey, {})
            blend = bool(phot.get("blend_flag", False))

            for comp in comp_names:
                m = phot.get(comp)
                if not isinstance(m, dict):
                    continue
                seds[comp].append({
                    "survey": survey,
                    "lambda_eff": lam,
                    "flux_mjy": m["flux_mjy"],
                    "flux_err_mjy": m["flux_err_mjy"],
                    "blend_flag": blend,
                })

        return seds

    def _write_products(self, prod_dir, target_id, photometry, masks_native,
                        deblend_ratio=None, deblend_tphot=None, deblend_comb=None,
                        unwise_forced=None, galaxy_table=None,
                        spectral_result=None, combined_result=None):
        deblend_ratio = deblend_ratio or {}
        deblend_tphot = deblend_tphot or {}
        deblend_comb = deblend_comb or {}
        unwise_forced = unwise_forced or {}
        comp_names = list(masks_native.keys())
        rows = []
        for survey in self.surveys:
            phot = photometry.get(survey, {})
            meta = SURVEY_DEFAULTS[survey]
            row = {
                "survey": survey,
                "lambda_eff_um": meta["lambda_eff"],
                "psf_fwhm_arcsec": meta.get("psf_fwhm", np.nan),
                "blend_flag": int(phot.get("blend_flag", False)),
            }
            for comp in comp_names + ["total"]:
                m = phot.get(comp, {})
                if not isinstance(m, dict):
                    m = {}
                row[f"{comp}_flux_mjy"] = m.get("flux_mjy", np.nan)
                row[f"{comp}_flux_err_mjy"] = m.get("flux_err_mjy", np.nan)
                row[f"{comp}_npix"] = m.get("n_pix", 0)

            # deblended fluxes (only populated for blended bands)
            for comp in comp_names:
                for tag, dd in (("ratio", deblend_ratio), ("tphot", deblend_tphot),
                                ("comb", deblend_comb)):
                    dm = dd.get(survey, {}).get(comp, {})
                    if not isinstance(dm, dict):
                        dm = {}
                    row[f"{comp}_{tag}_flux_mjy"] = dm.get("flux_mjy", np.nan)
                    row[f"{comp}_{tag}_flux_err_mjy"] = dm.get("flux_err_mjy", np.nan)

            # which method the combined estimate used for this band
            comb_entry = deblend_comb.get(survey, {})
            method = "none"
            for v in comb_entry.values():
                if isinstance(v, dict) and v.get("method"):
                    method = v["method"]
                    break
            row["deblend_method"] = method

            # unWISE forced photometry (only W1/W2, only if matched)
            for comp in comp_names:
                uw = unwise_forced.get(survey, {}).get(comp, {})
                row[f"{comp}_unwise_flux_mjy"] = uw.get("flux_mjy", np.nan)
                row[f"{comp}_unwise_flux_err_mjy"] = uw.get("flux_err_mjy", np.nan)
                row[f"{comp}_unwise_match_sep_arcsec"] = uw.get("match_sep_arcsec", np.nan)

            rows.append(row)

        tbl = Table(rows)
        csv_path = os.path.join(prod_dir, f"{target_id}_photometry.csv")
        ecsv_path = os.path.join(prod_dir, f"{target_id}_photometry.ecsv")
        tbl.write(csv_path, format="csv", overwrite=True)
        tbl.write(ecsv_path, format="ascii.ecsv", overwrite=True)

        # clean per-component galaxy flux table (.csv + .ecsv)
        if galaxy_table is not None:
            try:
                save_galaxy_table(
                    galaxy_table,
                    os.path.join(prod_dir, f"{target_id}_galaxy_fluxes"),
                )
            except Exception as exc:
                logger.error("save galaxy table failed: %s", exc)

        # optional spectral-fit results (total fibre light of the system)
        self._write_spectral_products(prod_dir, target_id, spectral_result)

        # optional merged photometry+spectroscopy product (mode="both")
        self._write_combined_products(prod_dir, target_id, combined_result)

    def _write_spectral_products(self, prod_dir, target_id, spectral_result):
        """Write {target_id}_spectral.csv/.ecsv (no-op if result is None)."""
        if spectral_result is None:
            return
        try:
            spec_tbl = Table([spectral_result])
            spec_tbl.write(
                os.path.join(prod_dir, f"{target_id}_spectral.csv"),
                format="csv", overwrite=True)
            spec_tbl.write(
                os.path.join(prod_dir, f"{target_id}_spectral.ecsv"),
                format="ascii.ecsv", overwrite=True)
        except Exception as exc:
            logger.error("save spectral table failed: %s", exc)

    def _write_combined_products(self, prod_dir, target_id, combined_result):
        """Write {target_id}_combined.csv/.ecsv (no-op if result is None)."""
        if combined_result is None:
            return
        try:
            # None values (e.g. missing spectral classes) break Table();
            # normalize them to NaN/empty strings.
            clean = {k: (np.nan if v is None else v)
                     for k, v in combined_result.items()}
            tbl = Table([clean])
            tbl.write(
                os.path.join(prod_dir, f"{target_id}_combined.csv"),
                format="csv", overwrite=True)
            tbl.write(
                os.path.join(prod_dir, f"{target_id}_combined.ecsv"),
                format="ascii.ecsv", overwrite=True)
        except Exception as exc:
            logger.error("save combined table failed: %s", exc)

    def _build_deblend_seds(self, deblend_comb, comp_names):
        """Build {method: comp_seds} from the combined deblend dict.

        Groups blended-band component fluxes by their chosen method so the SED
        plot can overlay them (one comp_seds-style dict per method present).
        """
        deblend_seds = {}
        for survey, entry in (deblend_comb or {}).items():
            meta = SURVEY_DEFAULTS.get(survey, {})
            lam = meta.get("lambda_eff")
            for comp in comp_names:
                m = entry.get(comp)
                if not isinstance(m, dict):
                    continue
                method = m.get("method", "comb")
                seds = deblend_seds.setdefault(method, {})
                seds.setdefault(comp, []).append({
                    "survey": survey,
                    "lambda_eff": lam,
                    "flux_mjy": m.get("flux_mjy", np.nan),
                    "flux_err_mjy": m.get("flux_err_mjy", np.nan),
                    "blend_flag": True,
                })
        return deblend_seds

    def _make_plots(self, plot_dir, target_id, target, images, masks_native,
                    masks_reproj, target_wcs, photometry, comp_seds, xmask_result,
                    deblend_comb=None, unwise_forced=None, deblend_ratio=None,
                    deblend_tphot=None, galaxy_table=None):
        try:
            vp.plot_processing_grid(
                target_id, self.surveys, images, masks_native, masks_reproj,
                target_wcs, os.path.join(plot_dir, f"{target_id}_processing_grid.png"),
            )
        except Exception as exc:
            logger.error("processing grid plot failed: %s", exc)

        try:
            sed_for_plot = {k: v for k, v in comp_seds.items() if k != "circular"}
            deblend_seds = self._build_deblend_seds(
                deblend_comb, list(masks_native.keys()))
            vp.plot_sed(target_id, sed_for_plot,
                        os.path.join(plot_dir, f"{target_id}_SED.png"),
                        deblend_seds=deblend_seds or None,
                        unwise_forced=unwise_forced or None)
        except Exception as exc:
            logger.error("SED plot failed: %s", exc)

        try:
            vp.plot_summary_panel(
                target_id, target, self.surveys, photometry, comp_seds,
                xmask_result, os.path.join(plot_dir, f"{target_id}_summary.png"),
            )
        except Exception as exc:
            logger.error("summary plot failed: %s", exc)

        try:
            vp.plot_galaxy_panels(
                target_id, self.surveys, images,
                masks_reproj,
                os.path.join(plot_dir, f"{target_id}_galaxies.png"),
            )
        except Exception as exc:
            logger.error("galaxy panels plot failed: %s", exc)

        try:
            vp.plot_comprehensive(
                target_id, self.surveys, images, masks_reproj, comp_seds,
                photometry, xmask_result,
                deblend_ratio=deblend_ratio, deblend_tphot=deblend_tphot,
                deblend_comb=deblend_comb, unwise_forced=unwise_forced,
                galaxy_table=galaxy_table,
                out_path=os.path.join(plot_dir, f"{target_id}_comprehensive.png"),
            )
        except Exception as exc:
            logger.error("comprehensive plot failed: %s", exc)
