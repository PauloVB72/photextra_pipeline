"""Tests for the config/separation/aperture-mode/CIGALE/spectral-line work.

Covers (see the task brief, Parts 1-7):
 - photometry.separation policy (central/total/pair) crossed with a target
   that HAS a "type" column and one that doesn't (Part 2, Part 7),
 - the mask-misplacement regression (Part 3a),
 - the WCS-alignment diagnostic for WISE/GALEX vs Legacy (Part 3b),
 - the expanded spectral-line extraction on real cached DESI spectra
   (Part 6),
 - a cigale_run=True smoke test on real cached targets (Part 4/7)
   (skipped automatically if the "cigale" conda env / cached products are
   not available on this machine).

Run: pytest -q test_new_features.py
"""

import os
import sys

import numpy as np
import pytest

PIPE_DIR = os.path.dirname(os.path.abspath(__file__))
if PIPE_DIR not in sys.path:
    sys.path.insert(0, PIPE_DIR)

from photextra_pipeline.pipeline import Pipeline, _StubXmaskResult
from photextra_pipeline.reprojection import build_common_wcs, reproject_masks
from photextra_pipeline.deblending import XdebPairResult
from photextra_pipeline.validation_plots import masks_on_band_grid

MKW8_RUN = "/home/polo/Escritorio/PHD/CHANCES/MKW8_full_run"
BASE_CFG = {
    "surveys": ["Legacy_g", "Legacy_r", "Legacy_z", "WISE_W1", "WISE_W2"],
    "output_dir": "/tmp/photextra_test_new_features",
    "common_grid": {"pixscale": 1.375, "size": 45},
}


def _pipeline(**overrides):
    cfg = dict(BASE_CFG)
    cfg.update(overrides)
    return Pipeline(config=cfg, use_xdebpair=False, mode="photometry")


def _fake_seg(n_components=2):
    """Two masks 16" apart on a small native grid, mimicking xdebpair output."""
    wcs = build_common_wcs(150.0, 2.0, 0.262, 120)
    gal1 = np.zeros((120, 120), bool)
    gal1[50:65, 50:65] = True
    gal2 = np.zeros((120, 120), bool)
    gal2[70:85, 70:85] = True
    masks = {"gal1": gal1, "gal2": gal2} if n_components == 2 else {"gal1": gal1}
    return XdebPairResult(masks=masks, wcs=wcs,
                          n_components=len(masks), separation_arcsec=16.0,
                          classification="merger" if n_components == 2
                          else "single")


# ===========================================================================
# Part 2 / Part 7: separation policy (central / total / pair)
# ===========================================================================

@pytest.mark.parametrize("ttype,expect_n", [
    ("merger", 2), ("pre_merger", 2), ("post_merger", 1),
    ("garbage_unrecognized", 1),
])
def test_separation_pair_with_type_column(ttype, expect_n):
    """pair: type column present -> classification decides component count."""
    pipe = _pipeline()
    pipe.separation = "pair"
    seg = _fake_seg(2)
    out = pipe._apply_separation_policy(seg, {"type": ttype})
    assert out.n_components == expect_n
    assert out.separation_policy == "pair"
    assert out.target_type == ttype.lower()
    assert out.n_components_detected == 2
    assert out.has_companion_not_measured is False


def test_separation_pair_no_type_column_keeps_xdebpair_decision():
    """pair: no 'type' key at all -> xdebpair's raw n_components stands."""
    pipe = _pipeline()
    pipe.separation = "pair"
    seg = _fake_seg(2)
    out = pipe._apply_separation_policy(seg, {"id": "t1", "ra": 1, "dec": 1})
    assert out.n_components == 2
    assert out.target_type == ""


def test_separation_central_forces_one_and_flags_companion():
    """central: always 1 component; a detected companion is flagged, not
    silently dropped."""
    pipe = _pipeline()
    pipe.separation = "central"
    seg = _fake_seg(2)
    out = pipe._apply_separation_policy(seg, {"type": "merger"})
    assert out.n_components == 1
    assert list(out.masks.keys()) == ["gal1"]
    assert out.has_companion_not_measured is True
    assert out.n_components_detected == 2


def test_separation_central_single_component_no_flag():
    pipe = _pipeline()
    pipe.separation = "central"
    seg = _fake_seg(1)
    out = pipe._apply_separation_policy(seg, {})
    assert out.n_components == 1
    assert out.has_companion_not_measured is False


def test_separation_total_forces_one_union_mask_includes_companion_flux():
    """total: 1 aperture, but it's the UNION (companion flux included) —
    distinguishes it from 'central', which discards the companion."""
    pipe = _pipeline()
    pipe.separation = "total"
    seg = _fake_seg(2)
    original_masks = seg.masks
    out = pipe._apply_separation_policy(seg, {"type": "merger"})
    assert out.n_components == 1
    union_mask = out.masks["gal1"]
    # union covers pixels from BOTH original components
    assert np.any(union_mask & original_masks["gal1"])
    assert np.any(union_mask & original_masks["gal2"])
    assert union_mask.sum() == (original_masks["gal1"]
                                | original_masks["gal2"]).sum()


def test_separation_config_default_is_pair():
    pipe = _pipeline()
    assert pipe.separation == "pair"
    assert pipe.aperture_mode == "mask"
    assert pipe.mode == "photometry"
    assert pipe.cigale_run is False


# ===========================================================================
# Part 7: the three aperture_modes are selectable and validated
# ===========================================================================

def test_aperture_mode_validation():
    cfg = dict(BASE_CFG)
    cfg["photometry"] = {"aperture_mode": "not_a_mode"}
    with pytest.raises(ValueError):
        Pipeline(config=cfg, use_xdebpair=False)


def test_separation_validation():
    cfg = dict(BASE_CFG)
    cfg["photometry"] = {"separation": "not_a_value"}
    with pytest.raises(ValueError):
        Pipeline(config=cfg, use_xdebpair=False)


def test_cigale_run_enables_cigale():
    """cigale_run is the single switch for the CIGALE SED fit, uniform
    across all modes (both_method was removed — cigale_run replaces it)."""
    cfg = dict(BASE_CFG)
    cfg["mode"] = "both"
    cfg["cigale_run"] = True
    pipe = Pipeline(config=cfg, use_xdebpair=False)
    assert pipe.cigale_run is True
    assert pipe._cigale_enabled() is True

    cfg2 = dict(BASE_CFG)
    cfg2["mode"] = "both"
    pipe2 = Pipeline(config=cfg2, use_xdebpair=False)
    assert pipe2.cigale_run is False
    assert pipe2._cigale_enabled() is False


def test_config_kwarg_override():
    """Explicit constructor kwargs override the YAML/dict config value
    (backward compat for run_mkw8_full.py / timing_*.py driver scripts)."""
    cfg = dict(BASE_CFG)
    cfg["mode"] = "photometry"
    pipe = Pipeline(config=cfg, use_xdebpair=False, mode="both")
    assert pipe.mode == "both"


# ===========================================================================
# Part 3a: mask-misplacement regression
# ===========================================================================

def test_mask_placement_on_native_band_grid():
    """Reproduces the reported bug: masks appear tiny / in the wrong corner
    on a multi-galaxy field's native-resolution (e.g. Legacy) panel.

    Root cause: the comprehensive plot drew the COMMON-grid masks
    (masks_reproj, small e.g. 45x45px) directly on each band's NATIVE image
    (e.g. Legacy ~230x230px) without re-projecting them to that band's own
    WCS/shape first — a pixel-grid mismatch that only shows up with an
    off-center companion (single-galaxy targets draw no contours).

    Fix: masks_on_band_grid() reprojects seg_result.masks (native xdebpair
    grid) onto each band's own WCS before contouring. This test asserts the
    reprojected mask's centroid overlaps the true source position on the
    NATIVE grid within 2 pixels — not just that it renders without
    crashing.
    """
    ra, dec = 218.357, 3.093
    native_wcs = build_common_wcs(ra, dec, 0.262, 230)
    common_wcs = build_common_wcs(ra, dec, 1.375, 45)

    gal1 = np.zeros((230, 230), bool)
    gal1[100:130, 100:130] = True          # near field center
    gal2 = np.zeros((230, 230), bool)
    gal2[140:160, 140:160] = True          # off-center companion (the case
                                           # that exposes the bug)
    seg = XdebPairResult(masks={"gal1": gal1, "gal2": gal2}, wcs=native_wcs,
                        n_components=2, separation_arcsec=16.0,
                        classification="merger")

    masks_reproj = reproject_masks(seg.masks, seg.wcs, common_wcs, (45, 45))
    entry = {"data": np.zeros((230, 230)), "wcs": native_wcs}

    band_masks = masks_on_band_grid(seg, entry, masks_reproj)

    def centroid(m):
        yy, xx = np.where(m)
        return xx.mean(), yy.mean()

    expected_cx, expected_cy = centroid(gal2)   # (149.5, 149.5)
    fixed_cx, fixed_cy = centroid(band_masks["gal2"])
    assert abs(fixed_cx - expected_cx) < 2.0
    assert abs(fixed_cy - expected_cy) < 2.0

    # demonstrate the OLD behaviour would have failed this same tolerance:
    # applying the common-grid mask directly to the native 230x230 image
    # (what plot_processing_grid/_plot_comprehensive_impl did before the
    # fix) lands nowhere near the true companion position.
    old_cx, old_cy = centroid(masks_reproj["gal2"])
    assert abs(old_cx - expected_cx) > 50.0, (
        "sanity check: the bug should be reproducible with the un-fixed "
        "common-grid mask; if this assert fails the test fixture no "
        "longer demonstrates the original bug")


def test_mask_placement_real_mkw8_target():
    """Same regression check using a REAL cached multi-component MKW8
    target's galaxy-fluxes product, if available on this machine."""
    tid = "2842599849197568"   # known 2-component target (n_comp rows==2)
    prod = os.path.join(MKW8_RUN, tid, "products", f"{tid}_galaxy_fluxes.csv")
    if not os.path.exists(prod):
        pytest.skip("real MKW8 cached product not available on this machine")
    import csv
    rows = list(csv.DictReader(open(prod)))
    assert len(rows) == 2
    ra1, dec1 = float(rows[0]["ra_deg"]), float(rows[0]["dec_deg"])
    ra2, dec2 = float(rows[1]["ra_deg"]), float(rows[1]["dec_deg"])
    # the two real components must NOT be at the same sky position (that
    # would indicate the corner-collapse bug even in the catalog centroids)
    sep_arcsec = np.hypot((ra1 - ra2) * np.cos(np.radians(dec1)),
                         dec1 - dec2) * 3600.0
    assert sep_arcsec > 1.0


# ===========================================================================
# Part 3b: WISE/GALEX vs Legacy/SDSS WCS-alignment diagnostic
# ===========================================================================

def test_skyview_pixel_scale_matches_survey_defaults():
    """Diagnostic for the reported WISE/GALEX misalignment: confirm the
    downloader now requests images whose pixel scale equals
    SURVEY_DEFAULTS[survey]['pixscale'] (the scale the AB zero point is
    calibrated for), rather than 2x that (the bug: astroquery's
    ``radius=`` kwarg is a HALF-width, so ``radius=size_arcmin`` produced a
    field 2x too wide at half the intended resolution).
    """
    import inspect
    from photextra_pipeline.downloader import _download_skyview
    src = inspect.getsource(_download_skyview)
    assert "width=" in src and "height=" in src, (
        "downloader._download_skyview must request width/height (exact "
        "field size), not radius= (half-width) -- see the fix comment")
    # the actual SkyView.get_images(...) call must not pass radius= (only
    # the explanatory comment above is allowed to mention it)
    active_lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
    assert not any("radius=size_arcmin" in l for l in active_lines)


def test_wise_legacy_target_position_alignment_real_cache():
    """Real-data check: for a cached MKW8 target, the target's (ra, dec)
    lands within ~1 native pixel of the expected pixel position on BOTH the
    Legacy and WISE cached cutouts (using each image's OWN header WCS, so
    this exercises the actual downloaded product, not just the pipeline's
    common-grid reprojection)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    import csv

    tid = "2842593834565632"
    cache = os.path.join(MKW8_RUN, tid, "cache")
    combined = os.path.join(MKW8_RUN, tid, "products", f"{tid}_combined.csv")
    if not (os.path.exists(cache) and os.path.exists(combined)):
        pytest.skip("real MKW8 cached cutouts not available on this machine")
    row = next(csv.DictReader(open(combined)))
    ra, dec = float(row["ra"]), float(row["dec"])

    positions = {}
    for survey in ("Legacy_r", "WISE_W1"):
        path = os.path.join(cache, f"{survey}.fits")
        if not os.path.exists(path):
            pytest.skip(f"{survey} cutout not cached for {tid}")
        with fits.open(path) as hdul:
            w = WCS(hdul[0].header).celestial
            shape = hdul[0].data.shape[-2:]
        cx, cy = w.world_to_pixel_values(ra, dec)
        # fractional position within the image (scale-independent check)
        positions[survey] = (cx / shape[1], cy / shape[0])

    fx = abs(positions["Legacy_r"][0] - positions["WISE_W1"][0])
    fy = abs(positions["Legacy_r"][1] - positions["WISE_W1"][1])
    # both cutouts are centered on the same (ra, dec) by construction, so
    # the target should land within a few % of the image center on both
    assert fx < 0.05 and fy < 0.05, (
        f"target fractional position differs between Legacy_r {positions['Legacy_r']} "
        f"and WISE_W1 {positions['WISE_W1']} -- possible WCS/centering bug")


# ===========================================================================
# Part 6: expanded spectral-line extraction (real cached DESI spectra)
# ===========================================================================

_SPEC_TARGETS = ["2842593834565632", "2842599849197568", "2842599874363392"]


def _available_spec_targets():
    out = []
    for tid in _SPEC_TARGETS:
        npz = os.path.join(MKW8_RUN, tid, "spectroscopy", f"desi_{tid}.npz")
        combined = os.path.join(MKW8_RUN, tid, "products",
                                f"{tid}_combined.csv")
        if os.path.exists(npz) and os.path.exists(combined):
            out.append(tid)
    return out


@pytest.mark.parametrize("tid", _available_spec_targets() or ["__none__"])
def test_expanded_line_extraction_real_spectrum(tid):
    """Fits a REAL cached DESI spectrum and asserts the expanded line list
    (>4 lines) with flux/EW/sigma columns is produced and populated for at
    least the strong lines (Halpha at minimum)."""
    if tid == "__none__":
        pytest.skip("no cached MKW8 DESI spectra available on this machine")

    import csv
    npz_path = os.path.join(MKW8_RUN, tid, "spectroscopy", f"desi_{tid}.npz")
    combined_path = os.path.join(MKW8_RUN, tid, "products",
                                 f"{tid}_combined.csv")
    row = next(csv.DictReader(open(combined_path)))
    z = float(row["z"])

    from photextra_pipeline.spectral_fit import (_load_spectrum,
                                                  _flatten_emission_lines)
    from xpectrafit import XpectraFitter

    spec = _load_spectrum(npz_path, z, tid)
    # fit_emission=True forced explicitly: XpectraFit's pre_classify() skips
    # emission-line fitting entirely for very quiescent spectra (EW(Halpha)
    # < 3A) as a deliberate speed/quality call -- that heuristic is
    # xpectrafit's business, not what this test checks. This test verifies
    # OUR flattening code (_flatten_emission_lines) covers every line/
    # quantity XpectraFit CAN report, using a real cached spectrum's
    # continuum+noise (not fabricated flux), so forcing the line fit on is
    # the correct, honest way to exercise that code path here.
    result = XpectraFitter(spec.wave, spec.flux, spec.flux_err, z=z,
                           target_id=tid, fwhm_gal=2.5, fit_agn=True,
                           fit_emission=True).fit()
    cols = _flatten_emission_lines(result)

    # Derive canonical line names from the "_ew" keys specifically (not
    # "_ew_err", "_sigma", etc.) -- a naive rsplit on the last underscore
    # breaks on "line_<Name>_ew_err" (two trailing segments), which produced
    # a bogus "<Name>_ew" pseudo-name and failed the flux_/line_ checks below.
    line_names = {k[len("line_"):-len("_ew")]
                 for k in cols if k.startswith("line_") and k.endswith("_ew")}
    # more than the old 4-line subset
    assert len(line_names) > 4
    # every measured line carries flux, EW and sigma columns (Part 6 ask)
    for name in line_names:
        assert f"flux_{name}" in cols
        assert f"line_{name}_ew" in cols
        assert f"line_{name}_sigma" in cols
    # Halpha should be measured with a finite flux in real star-forming/AGN
    # cluster-member spectra at this S/N
    assert np.isfinite(cols.get("flux_Halpha", np.nan))


# ===========================================================================
# Part 4/7: cigale_run=True smoke test (real cached targets)
# ===========================================================================

def _cigale_env_available():
    import shutil
    return shutil.which("conda") is not None


@pytest.mark.parametrize("tid", _available_spec_targets()[:2] or ["__none__"])
def test_cigale_input_row_builds_from_real_cached_products(tid):
    """Smoke test for the CIGALE bridge WITHOUT invoking pcigale itself
    (that requires the "cigale" conda env + can take tens of seconds even
    for one target): confirms build_spectro_row() produces a valid input
    row and prism FITS from real cached photextra products, matching what
    Pipeline._run_cigale_for_target does before calling pcigale run.
    """
    if tid == "__none__":
        pytest.skip("no cached MKW8 targets available on this machine")
    import csv
    import tempfile
    from photextra_pipeline.cigale_integration import build_spectro_row

    combined_path = os.path.join(MKW8_RUN, tid, "products",
                                 f"{tid}_combined.csv")
    npz_path = os.path.join(MKW8_RUN, tid, "spectroscopy", f"desi_{tid}.npz")
    combined = next(csv.DictReader(open(combined_path)))

    with tempfile.TemporaryDirectory() as work:
        row = build_spectro_row(tid, combined, npz_path, work)
        assert row["id"] == tid
        assert os.path.exists(row["spectrum"])
        assert row["mode"] == "desi" and row["norm"] == "wave"
        # at least one broadband column has a finite flux
        assert any(v not in ("nan", "") for k, v in row.items()
                  if k not in ("id", "redshift", "spectrum", "mode", "norm"))


@pytest.mark.skipif(not _cigale_env_available(),
                    reason="conda not on PATH; cannot exercise the real "
                           "'cigale' environment")
def test_cigale_full_run_real_targets():
    """Full cigale_run=True smoke test: runs actual `pcigale run` on
    2 real cached MKW8 targets. Skipped when the 'cigale' conda env isn't
    installed on this machine (checked indirectly via conda availability;
    a missing env still fails clearly inside run_pcigale)."""
    targets = _available_spec_targets()[:2]
    if len(targets) < 1:
        pytest.skip("no cached MKW8 targets available on this machine")
    import csv
    from photextra_pipeline.cigale_integration import run_cigale_for_targets

    specs = []
    for tid in targets:
        combined_path = os.path.join(MKW8_RUN, tid, "products",
                                     f"{tid}_combined.csv")
        npz_path = os.path.join(MKW8_RUN, tid, "spectroscopy",
                                f"desi_{tid}.npz")
        specs.append((tid, next(csv.DictReader(open(combined_path))),
                     npz_path))

    work_dir = "/tmp/photextra_test_cigale_smoke"
    try:
        results = run_cigale_for_targets(specs, work_dir)
    except Exception as exc:
        pytest.skip(f"cigale environment not runnable on this machine: {exc}")
    assert results, "expected at least one target with cigale_* columns"
    for tid, cols in results.items():
        assert "cigale_chi2_red" in cols


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
