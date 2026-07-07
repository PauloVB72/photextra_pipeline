"""Unit tests for photextra_pipeline.spec_normalization.

Covers the four required guarantees:
 (a) with only 3 Legacy-like anchors, degree 2 is never selected;
 (b) a clean colour slope in the synthetic data recovers ~degree 1;
 (c) sigma-clipping removes a deliberately injected outlier band;
 (d) the anchor collection is generic over surveys -- a faked in-range
     "SDSS_i" band is picked up with no hardcoded band-list change.

Run: pytest -q test_spec_normalization.py
"""

import numpy as np

from photextra_pipeline.spec_normalization import (
    Anchor, collect_anchors, fit_spec_normalization,
    line_observed_wavelength_aa,
)

# Legacy g/r/z effective wavelengths in Angstrom (observed frame)
LAM = {"Legacy_g": 4720.0, "Legacy_r": 6410.0, "Legacy_z": 9060.0}


def _anchor(name, lam, phot, synth, phot_err=None, synth_err=None):
    if phot_err is None:
        phot_err = 0.005 * phot   # ~0.5% photometric error
    if synth_err is None:
        synth_err = 0.01 * synth
    return Anchor(name, lam, phot, phot_err, synth, synth_err)


def test_three_anchors_never_degree2():
    """(a) Guardrail: 3 anchors can never trigger degree 2 (needs >=5)."""
    # even with a strong curvature imprinted, only deg0/1 are candidates
    synth = {"Legacy_g": 1.0, "Legacy_r": 1.0, "Legacy_z": 1.0}
    factor = 2.0  # grey-ish with a mild slope
    anchors = []
    for b, lam in LAM.items():
        x = np.log(lam / 6000.0)
        phot = synth[b] * factor * np.exp(0.3 * x + 0.8 * x * x)
        anchors.append(_anchor(b, lam, phot, synth[b]))
    spec_norm, meta = fit_spec_normalization(anchors)
    assert meta["degree"] in (0, 1)
    assert meta["degree"] != 2
    assert "deg2" not in meta["models"]  # never even a candidate


def test_recover_degree1_slope():
    """(b) A clean colour slope is recovered as degree 1 with ~right slope."""
    lam0 = float(np.exp(np.mean(np.log(list(LAM.values())))))
    c0_true, c1_true = np.log(1.8), 0.6
    anchors = []
    for b, lam in LAM.items():
        synth = 1.0
        x = np.log(lam / lam0)
        phot = synth * np.exp(c0_true + c1_true * x)
        # tiny errors so the slope is well constrained -> passes BIC + CV gates
        anchors.append(_anchor(b, lam, phot, synth,
                               phot_err=1e-4 * phot, synth_err=1e-4))
    spec_norm, meta = fit_spec_normalization(anchors)
    assert meta["degree"] == 1, meta
    c = meta["coeffs"]
    assert abs(c[0] - c0_true) < 0.05
    assert abs(c[1] - c1_true) < 0.1
    # scale evaluated at the blue and red ends brackets the grey value
    assert spec_norm.scale_at(LAM["Legacy_g"]) < spec_norm.scale_at(LAM["Legacy_z"])


def test_sigma_clip_removes_outlier():
    """(c) An injected outlier band is sigma-clipped out of the fit."""
    # 5 clean bands on a flat grey ratio of 2.0, plus one wild outlier
    bands = {
        "Legacy_g": 4720.0, "Legacy_r": 6410.0, "Legacy_z": 9060.0,
        "SDSS_i": 7480.0, "SDSS_u": 3550.0,
    }
    anchors = []
    for b, lam in bands.items():
        anchors.append(_anchor(b, lam, 2.0, 1.0,
                               phot_err=1e-3 * 2.0, synth_err=1e-3))
    # outlier: ratio 8x instead of 2x. A realistic bad band has a larger
    # (lower-weight) error, so the weighted mean stays near the clean cluster
    # and the outlier sits many-sigma away -> clipped.
    anchors.append(_anchor("BAD_band", 5500.0, 8.0, 1.0,
                           phot_err=0.05 * 8.0, synth_err=0.05))
    spec_norm, meta = fit_spec_normalization(anchors)
    assert "BAD_band" in meta["names_clipped"]
    assert "BAD_band" not in meta["names_kept"]
    assert meta["n_clipped"] >= 1
    # grey level recovered near 2.0, not dragged up by the 8x outlier
    assert abs(spec_norm.scale_at(meta["lam0"]) - 2.0) < 0.2


def test_collect_anchors_generic_over_surveys():
    """(d) A faked in-range SDSS_i band is picked up with no code change."""
    surveys = ["GALEX_NUV", "Legacy_g", "Legacy_r", "Legacy_z", "SDSS_i",
               "WISE_W1"]
    lambda_eff_um = {
        "GALEX_NUV": 0.231,   # 2310 AA, below coverage -> excluded
        "Legacy_g": 0.472, "Legacy_r": 0.641, "Legacy_z": 0.906,
        "SDSS_i": 0.748,      # 7480 AA, in range -> included
        "WISE_W1": 3.4,       # 34000 AA, above coverage -> excluded
    }
    combined = {}
    spectral_result = {"spec_wave_min_aa": 3600.0, "spec_wave_max_aa": 9824.0}
    for s in surveys:
        combined[f"phot_{s}_flux_mjy"] = 2.0
        combined[f"phot_{s}_flux_err_mjy"] = 0.01
        spectral_result[f"synth_{s}_flux_mjy"] = 1.0
        spectral_result[f"synth_{s}_flux_err_mjy"] = 0.01
    anchors = collect_anchors(surveys, lambda_eff_um, combined,
                              spectral_result, 3600.0, 9824.0, "phot_")
    names = {a.name for a in anchors}
    assert "SDSS_i" in names           # generic pickup, no hardcoded list
    assert "GALEX_NUV" not in names    # out of (blue) coverage
    assert "WISE_W1" not in names      # out of (red) coverage
    assert names == {"Legacy_g", "Legacy_r", "Legacy_z", "SDSS_i"}


def test_missing_synth_flux_excludes_band():
    """A band lacking a synthetic flux is not eligible as an anchor."""
    surveys = ["Legacy_g", "Legacy_r"]
    lambda_eff_um = {"Legacy_g": 0.472, "Legacy_r": 0.641}
    combined = {"phot_Legacy_g_flux_mjy": 2.0, "phot_Legacy_g_flux_err_mjy": 0.01,
                "phot_Legacy_r_flux_mjy": 2.0, "phot_Legacy_r_flux_err_mjy": 0.01}
    spectral_result = {"synth_Legacy_g_flux_mjy": 1.0}  # r missing
    anchors = collect_anchors(surveys, lambda_eff_um, combined,
                              spectral_result, 3600.0, 9824.0, "phot_")
    assert {a.name for a in anchors} == {"Legacy_g"}


def test_empty_anchors_returns_none():
    spec_norm, meta = fit_spec_normalization([])
    assert spec_norm is None
    assert meta["degree"] == -1


def test_line_observed_wavelength():
    z = 0.0279
    assert abs(line_observed_wavelength_aa("Halpha", z) - 6562.8 * (1 + z)) < 1e-6
    assert np.isnan(line_observed_wavelength_aa("Unknown", z))


if __name__ == "__main__":
    import sys
    sys.exit(__import__("pytest").main([__file__, "-q"]))
