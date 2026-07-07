"""Wavelength-dependent spectrum-to-photometry normalization.

A fibre spectrum (DESI ~1.5", SDSS ~2.5") under-captures a galaxy's total
light relative to whole-system imaging photometry. To put synthetic
photometry / emission-line fluxes derived from the spectrum on the same
physical (total-light) scale as the imaging, the spectrum must be rescaled.

The classic approach (Zou et al. 2024, ApJ 961, 173) is a single grey
(wavelength-independent) factor ``F_phot / F_synth`` over one or a few
reference bands. That cannot capture a *wavelength-dependent* aperture /
colour mismatch (differential slit losses, PSF chromaticity, colour
gradients).

This module fits a smooth, wavelength-dependent correction instead::

    ln(scale(lambda)) = c0 + c1 * x + c2 * x**2 ,   x = ln(lambda / lambda0)

i.e. a degree-0/1/2 polynomial in log-wavelength, where ``lambda0`` is the
geometric mean of the usable anchor pivot wavelengths. Degree 0 reduces to a
(log-space) grey factor; degrees 1/2 add a colour slope / curvature. The
degree is chosen automatically and conservatively (BIC + leave-one-out CV
gating, sigma-clipping of anchor outliers), so the fit never over-reaches
when few anchors are available.

Method adapted from the CIGALE ``cigale2s`` branch's ``normalize_spec_to_phot``
(``pcigale/utils/read_prism.py``) for scientific provenance. This is an
original, standalone reimplementation using numpy only; no CIGALE code is
imported or vendored. The main deviation is that the anchor synthetic fluxes
are supplied pre-integrated (the pipeline already computes synthetic
broadband photometry through custom filters in ``xpectrafit``), rather than
being integrated from the raw spectrum inside this function.

The anchor collection is intentionally GENERIC over surveys: it iterates over
whatever imaging surveys are configured (Legacy today; SDSS / HSC / S-PLUS /
J-PLUS in the future) and admits any band whose effective wavelength falls
inside the observed-spectrum coverage and for which both a real total-aperture
flux and a synthetic flux exist. No hardcoded band list.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# --- degree-selection thresholds (mirroring the cigale2s reference) --------
MIN_POINTS_DEG1 = 3     # >=3 usable anchors to even attempt a slope
MIN_POINTS_DEG2 = 5     # >=5 usable anchors to attempt curvature
MIN_SPAN_DEG1 = 0.10    # x_span = ln(lam_max/lam_min); ~10% lever arm
MIN_SPAN_DEG2 = 0.40    # ~50% lever arm before curvature is allowed
MAX_COND = 1e8          # reject a degree if weighted design is near-degenerate
DELTA_BIC_MIN = 6.0     # BIC must improve by >=6 to upgrade degree
CV_IMPROVE_MIN = 0.05   # LOO-CV chi2 must drop by >=5% to upgrade degree
CURV2_ABS_MAX = 0.5     # cap on |c2| in ln-space
CLIP_SIGMA = 3.0        # sigma-clip anchors at 3 sigma in log-ratio space
MAX_CLIP_FRAC = 0.34    # never clip away more than 34% of the anchors


class Anchor:
    """One normalization anchor band: real vs synthetic flux at a pivot lambda.

    Fluxes in mJy, wavelength in Angstrom (observed frame, matching the
    spectrum). ``phot_err``/``synth_err`` may be NaN (handled downstream).
    """

    __slots__ = ("name", "lam_aa", "phot", "phot_err", "synth", "synth_err")

    def __init__(self, name, lam_aa, phot, phot_err, synth, synth_err):
        self.name = name
        self.lam_aa = float(lam_aa)
        self.phot = float(phot)
        self.phot_err = float(phot_err)
        self.synth = float(synth)
        self.synth_err = float(synth_err)

    def __repr__(self):
        return (f"Anchor({self.name!r}, lam={self.lam_aa:.1f}AA, "
                f"phot={self.phot:.4g}, synth={self.synth:.4g})")


class SpecNorm:
    """A fitted wavelength-dependent normalization ``scale(lambda)``.

    ``scale_at(lambda_aa)`` returns the multiplicative correction factor to
    apply to a synthetic flux (or emission-line flux) at observed-frame
    wavelength ``lambda_aa`` (Angstrom). ``degree == 0`` is a grey factor.
    """

    __slots__ = ("coeffs", "lam0", "degree")

    def __init__(self, coeffs, lam0, degree):
        self.coeffs = np.atleast_1d(np.asarray(coeffs, float))
        self.lam0 = float(lam0)
        self.degree = int(degree)

    def scale_at(self, lam_aa):
        """Multiplicative scale at wavelength(s) ``lam_aa`` (Angstrom)."""
        lam = np.asarray(lam_aa, float)
        x = np.log(np.clip(lam, 1e-300, None) / self.lam0)
        c = self.coeffs
        poly = c[0] * np.ones_like(x)
        if c.size > 1:
            poly = poly + c[1] * x
        if c.size > 2:
            poly = poly + c[2] * x * x
        out = np.exp(poly)
        return float(out) if np.ndim(out) == 0 else out


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def collect_anchors(surveys, lambda_eff_um, combined, spectral_result,
                    spec_wave_min_aa=None, spec_wave_max_aa=None,
                    phot_prefix="phot_"):
    """Gather usable normalization anchors, generic over configured surveys.

    A survey band becomes an anchor when, and only when, ALL hold:

    * it has an effective wavelength ``lambda_eff_um[survey]`` (micron);
    * that wavelength lies within the observed-spectrum coverage
      ``[spec_wave_min_aa, spec_wave_max_aa]`` (skipped if coverage is
      unknown -- the synth-flux presence check below still gates it);
    * a real total-aperture imaging flux ``{phot_prefix}{survey}_flux_mjy``
      is finite and > 0;
    * a synthetic flux ``synth_{survey}_flux_mjy`` (spectrum integrated
      through that band) is finite and > 0.

    No hardcoded band list: add a survey to the pipeline config and it
    participates automatically the moment its imaging + synthetic fluxes and
    an in-range ``lambda_eff`` are present.

    Parameters
    ----------
    surveys : iterable of str
        Configured imaging survey band names (``Pipeline.surveys``).
    lambda_eff_um : mapping
        ``{survey: lambda_eff_micron}`` (e.g. ``SURVEY_DEFAULTS[s]['lambda_eff']``).
    combined : mapping
        Flat dict carrying ``{phot_prefix}{survey}_flux_mjy`` columns.
    spectral_result : mapping
        Carries ``synth_{survey}_flux_mjy`` / ``_flux_err_mjy``.
    spec_wave_min_aa, spec_wave_max_aa : float, optional
        Observed-frame spectral coverage (Angstrom). If either is ``None`` /
        non-finite the corresponding bound check is skipped.
    phot_prefix : str
        ``"phot_"`` for total-aperture (primary) or ``"phot_fiber_"`` for the
        fiber-matched fallback.

    Returns
    -------
    list of :class:`Anchor`
    """
    lo = _to_float(spec_wave_min_aa)
    hi = _to_float(spec_wave_max_aa)
    anchors = []
    for survey in surveys:
        lam_um = lambda_eff_um.get(survey) if hasattr(lambda_eff_um, "get") \
            else None
        if lam_um is None:
            continue
        lam_aa = float(lam_um) * 1e4  # micron -> Angstrom
        if np.isfinite(lo) and lam_aa < lo:
            continue
        if np.isfinite(hi) and lam_aa > hi:
            continue
        fp = _to_float(combined.get(f"{phot_prefix}{survey}_flux_mjy"))
        fe = _to_float(combined.get(f"{phot_prefix}{survey}_flux_err_mjy"))
        fs = _to_float(spectral_result.get(f"synth_{survey}_flux_mjy"))
        se = _to_float(spectral_result.get(f"synth_{survey}_flux_err_mjy"))
        if not (np.isfinite(fp) and fp > 0 and np.isfinite(fs) and fs > 0):
            continue
        anchors.append(Anchor(survey, lam_aa, fp, fe, fs, se))
    return anchors


def _design(x, deg):
    x = np.asarray(x, float)
    if deg == 0:
        return np.ones((x.size, 1))
    if deg == 1:
        return np.column_stack([np.ones_like(x), x])
    return np.column_stack([np.ones_like(x), x, x * x])


def _fit_deg(deg, x, y, w, curv2_abs_max=CURV2_ABS_MAX):
    """Weighted least-squares fit of degree ``deg`` in log-lambda space.

    For degree 2, |c2| is capped at ``curv2_abs_max`` and (c0, c1) refit with
    the curvature held fixed. Returns ``(coeffs, cond)`` where ``cond`` is the
    condition number of the weighted design matrix.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.nan_to_num(np.asarray(w, float), nan=0.0, posinf=np.inf, neginf=0.0)
    w = np.clip(w, 0.0, np.inf)

    X = _design(x, deg)
    sw = np.sqrt(w)
    if not np.any(sw > 0):
        sw = np.ones_like(sw)

    Xw = X * sw[:, None]
    yw = y * sw
    c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    if deg == 2 and c.size == 3 and np.isfinite(c[2]) \
            and abs(c[2]) > float(curv2_abs_max):
        c2_fixed = np.sign(c[2]) * float(curv2_abs_max)
        y2 = y - c2_fixed * (x * x)
        X1w = _design(x, 1) * sw[:, None]
        c01, *_ = np.linalg.lstsq(X1w, y2 * sw, rcond=None)
        c = np.array([c01[0], c01[1], c2_fixed], dtype=float)
        Xw = _design(x, 2) * sw[:, None]

    cond = float(np.linalg.cond(Xw)) if Xw.shape[0] >= Xw.shape[1] else np.inf
    return c, cond


def fit_spec_normalization(anchors, allow_wave=True):
    """Fit ``scale(lambda)`` from a list of anchors; auto-select the degree.

    Parameters
    ----------
    anchors : list of :class:`Anchor`
    allow_wave : bool
        If ``False``, force degree 0 (grey). Useful to reproduce the old
        behaviour or when a wavelength-dependent fit is not wanted.

    Returns
    -------
    (SpecNorm or None, dict)
        The fitted normalization (``None`` when no usable anchor exists) and a
        diagnostics ``meta`` dict with keys ``degree``, ``coeffs``, ``lam0``,
        ``n_anchors`` (post-clip), ``n_clipped``, ``chi2``, ``bic``, ``cond``,
        ``names_kept``, ``names_clipped``, ``x_span``, and ``models`` (per
        candidate degree).
    """
    meta = {
        "degree": -1, "coeffs": [], "lam0": np.nan,
        "n_anchors": 0, "n_clipped": 0, "chi2": np.nan, "bic": np.nan,
        "cond": np.nan, "names_kept": [], "names_clipped": [],
        "x_span": np.nan, "models": {},
    }
    if not anchors:
        return None, meta

    names = np.array([a.name for a in anchors], dtype=object)
    lam = np.array([a.lam_aa for a in anchors], dtype=float)
    P = np.array([a.phot for a in anchors], dtype=float)
    Sig = np.array([a.phot_err for a in anchors], dtype=float)
    S = np.array([a.synth for a in anchors], dtype=float)
    Serr = np.array([a.synth_err for a in anchors], dtype=float)

    # errors -> non-negative; missing synth error treated as 0 (weight then
    # dominated by the photometric error, as in the reference).
    Serr = np.where(np.isfinite(Serr) & (Serr > 0), Serr, 0.0)
    # missing/zero phot error -> tiny relative error (keeps weight finite;
    # does not silently drop the band).
    rel_phot = np.where((Sig > 0) & np.isfinite(Sig), Sig / P, 0.0)
    N = int(P.size)

    # --- sigma-clip anchor outliers in log-ratio space --------------------
    n_clipped = 0
    clipped_names = []
    if N >= 4:
        ln_ratio = np.log(P) - np.log(S)
        var_ln = rel_phot ** 2 + (Serr / S) ** 2
        w_pre = 1.0 / np.clip(var_ln, 1e-30, np.inf)
        mu = np.average(ln_ratio, weights=w_pre)
        sig_abs = np.sqrt(1.0 / np.clip(w_pre, 1e-30, np.inf))
        keep = np.abs(ln_ratio - mu) <= CLIP_SIGMA * sig_abs
        if np.count_nonzero(keep) >= max(3, int(np.ceil((1.0 - MAX_CLIP_FRAC) * N))):
            clipped_names = list(names[~keep])
            n_clipped = int((~keep).sum())
            names, lam, P, Sig, S, Serr, rel_phot = (
                names[keep], lam[keep], P[keep], Sig[keep], S[keep],
                Serr[keep], rel_phot[keep])
            N = int(keep.sum())

    # --- reference wavelength and log-space design ------------------------
    lam0 = float(np.exp(np.mean(np.log(np.clip(lam, 1e-300, None)))))
    x = np.log(lam / lam0)
    y = np.log(P) - np.log(S)
    w_ln = 1.0 / np.clip(rel_phot ** 2 + (Serr / S) ** 2, 1e-30, np.inf)
    x_span = float(np.max(x) - np.min(x)) if N > 0 else 0.0

    def chi2_bic(c):
        scale = SpecNorm(c, lam0, len(c) - 1).scale_at(lam)
        scale = np.atleast_1d(scale)
        var = np.clip(Sig ** 2 + (scale * Serr) ** 2, 1e-30, np.inf)
        # if a phot error is missing, fall back to a relative-flux variance so
        # chi2 stays finite and meaningful
        bad = ~(np.isfinite(Sig) & (Sig > 0))
        if np.any(bad):
            var = var.copy()
            var[bad] = np.clip((0.02 * P[bad]) ** 2 + (scale[bad] * Serr[bad]) ** 2,
                               1e-30, np.inf)
        resid = P - scale * S
        chi2 = float(np.sum(resid ** 2 / var))
        k = int(len(c))
        bic = float(chi2 + k * np.log(max(N, 1)))
        return chi2, bic

    # --- candidate degrees ------------------------------------------------
    allowed = [0]
    if allow_wave:
        if N >= MIN_POINTS_DEG1 and x_span >= MIN_SPAN_DEG1:
            allowed.append(1)
        if N >= MIN_POINTS_DEG2 and x_span >= MIN_SPAN_DEG2:
            allowed.append(2)

    models = {}
    for d in allowed:
        c, cond = _fit_deg(d, x, y, w_ln)
        chi2_d, bic_d = chi2_bic(c)
        models[d] = {"c": c, "chi2": float(chi2_d), "bic": float(bic_d),
                     "cond": float(cond)}

    def loo_cv(deg):
        total = 0.0
        c2_signs = []
        for i in range(N):
            m = np.ones(N, dtype=bool)
            m[i] = False
            if np.count_nonzero(m) < (deg + 1):
                return np.inf, False
            ci, _ = _fit_deg(deg, x[m], y[m], w_ln[m])
            scale = float(np.atleast_1d(SpecNorm(ci, lam0, deg).scale_at(lam[i]))[0])
            var = max(Sig[i] ** 2 + (scale * Serr[i]) ** 2, 1e-30)
            if not (np.isfinite(Sig[i]) and Sig[i] > 0):
                var = max((0.02 * P[i]) ** 2 + (scale * Serr[i]) ** 2, 1e-30)
            resid = P[i] - scale * S[i]
            total += float(resid * resid / var)
            if deg == 2:
                c2_signs.append(np.sign(ci[2]))
        stable = True
        if c2_signs:
            npos = int(np.sum(np.array(c2_signs) >= 0))
            stable = (npos == 0) or (npos == len(c2_signs))
        return float(total), stable

    for d in allowed:
        cv, stable = loo_cv(d)
        models[d]["cv_chi2"] = float(cv)
        models[d]["stable"] = bool(stable)

    # --- conservative upgrade: BIC AND LOO-CV AND (deg2) stability + cond --
    chosen = min(allowed)
    for d in sorted(allowed)[1:]:
        prev = chosen
        if models[d]["cond"] > MAX_COND:
            continue
        if d == 2 and not models[d]["stable"]:
            continue
        delta_bic = models[prev]["bic"] - models[d]["bic"]
        pass_bic = delta_bic >= DELTA_BIC_MIN
        cv_prev, cv_new = models[prev]["cv_chi2"], models[d]["cv_chi2"]
        pass_cv = (np.isfinite(cv_prev) and np.isfinite(cv_new)
                   and cv_new <= (1.0 - CV_IMPROVE_MIN) * cv_prev)
        models[d]["delta_bic_vs_prev"] = float(delta_bic)
        if pass_bic and pass_cv:
            chosen = d

    coeffs = models[chosen]["c"]
    spec_norm = SpecNorm(coeffs, lam0, chosen)

    meta.update({
        "degree": int(chosen),
        "coeffs": [float(c) for c in np.atleast_1d(coeffs)],
        "lam0": float(lam0),
        "n_anchors": int(N),
        "n_clipped": int(n_clipped),
        "chi2": float(models[chosen]["chi2"]),
        "bic": float(models[chosen]["bic"]),
        "cond": float(models[chosen]["cond"]),
        "names_kept": list(names),
        "names_clipped": list(clipped_names),
        "x_span": float(x_span),
        "models": {
            f"deg{d}": {
                "coeffs": [float(ci) for ci in np.atleast_1d(models[d]["c"])],
                "chi2": float(models[d]["chi2"]),
                "bic": float(models[d]["bic"]),
                "cv_chi2": float(models[d].get("cv_chi2", np.nan)),
                "cond": float(models[d].get("cond", np.nan)),
                "stable": bool(models[d].get("stable", True)),
            } for d in allowed
        },
    })
    return spec_norm, meta


# observed-frame rest wavelengths (Angstrom) of the pipeline's tracked lines,
# used to evaluate the wavelength-dependent scale at each line's position.
LINE_REST_WAVELENGTHS_AA = {
    "Halpha": 6562.8,
    "Hbeta": 4861.3,
    "OIII5007": 5006.8,
    "NII6584": 6583.5,
}


def line_observed_wavelength_aa(line_name, z):
    """Observed-frame wavelength (AA) of a tracked emission line, or NaN."""
    rest = LINE_REST_WAVELENGTHS_AA.get(line_name)
    if rest is None:
        return np.nan
    try:
        return float(rest) * (1.0 + float(z))
    except (TypeError, ValueError):
        return np.nan
