"""Diagnostic plots: processing grid, SED, and summary data-sheet.

Publication-quality figures for the photextra_pipeline (galaxy-merger
photometry). Function signatures are kept identical to those called by
pipeline.py; all functions restore the default matplotlib style on exit.
"""

import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.colors import AsinhNorm
from astropy.visualization import ZScaleInterval, PercentileInterval
from astropy.wcs import WCS

from .downloader import SURVEY_DEFAULTS

logger = logging.getLogger(__name__)

# --- shared style ----------------------------------------------------------
_COMP_COLORS = {"gal1": "#ff5d5d", "gal2": "#5db4ff", "total": "#ffd24d"}
_COMP_LABELS = {"gal1": "galaxy 1", "gal2": "galaxy 2", "total": "total"}

# survey-family border colors
_FAMILY_COLORS = {"GALEX": "#b44fff", "Legacy": "#00c8c8", "WISE": "#ff8800"}

# wavelength regions (micron) for SED shading: (lo, hi, label, color)
_WAVE_BANDS = [
    (0.10, 0.40, "UV", "#b44fff"),
    (0.40, 1.00, "optical", "#00c8c8"),
    (1.00, 5.00, "NIR", "#ff8800"),
    (5.00, 40.0, "MIR", "#ff4d6d"),
]

_C_UM_HZ = 2.99792458e14  # speed of light in micron/s -> nu[Hz] = c/lambda[um]


def _family(survey):
    """Return the survey family key ('GALEX'/'Legacy'/'WISE') or None."""
    for fam in _FAMILY_COLORS:
        if survey.startswith(fam):
            return fam
    return None


def _family_color(survey):
    return _FAMILY_COLORS.get(_family(survey), "#888888")


def _norm(data):
    if data is None:
        return None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    try:
        vmin, vmax = ZScaleInterval().get_limits(finite)
    except Exception:
        vmin, vmax = np.nanpercentile(finite, [5, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    return {"vmin": vmin, "vmax": vmax}


def _style_border(ax, color, lw=2.0):
    """Draw a thin colored border around an axis."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(lw)


def _fmt_coords(ra, dec):
    try:
        return f"{float(ra):.5f}, {float(dec):.5f}"
    except (TypeError, ValueError):
        return "n/a"


def _compass(ax):
    """Draw a small N-up / E-left compass rose in the top-right of an axis."""
    # work in axes fraction coordinates
    x0, y0 = 0.88, 0.18  # origin of the rose
    ln = 0.10
    common = dict(transform=ax.transAxes, color="white", width=0.004,
                  head_width=0.025, head_length=0.025, length_includes_head=True,
                  zorder=10)
    # North: up
    ax.add_patch(FancyArrow(x0, y0, 0, ln, **common))
    # East: left
    ax.add_patch(FancyArrow(x0, y0, -ln, 0, **common))
    ax.text(x0, y0 + ln + 0.03, "N", transform=ax.transAxes, color="white",
            ha="center", va="bottom", fontsize=8, zorder=10)
    ax.text(x0 - ln - 0.03, y0, "E", transform=ax.transAxes, color="white",
            ha="right", va="center", fontsize=8, zorder=10)


# ===========================================================================
# 1. processing grid
# ===========================================================================
def plot_processing_grid(target_id, surveys, images, masks_native, masks_reproj,
                         target_wcs, out_path):
    """n_bands x 3 columns: Original | Convolved | Reprojected.

    The Reprojected column carries the common WCS (RA/Dec ticks); the other
    two columns are in pixel space and are kept tick-free.
    """
    plt.style.use("dark_background")
    n = max(len(surveys), 1)

    # Build a WCS object for the reprojected column if available.
    wcs = None
    if target_wcs is not None:
        try:
            wcs = target_wcs if isinstance(target_wcs, WCS) else WCS(target_wcs)
            wcs = wcs.celestial
        except Exception:
            wcs = None

    fig = plt.figure(figsize=(11.5, 3.2 * n))
    gs = fig.add_gridspec(n, 3, wspace=0.08, hspace=0.12,
                          left=0.10, right=0.93, top=0.94, bottom=0.03)

    col_titles = ["Original", "Convolved", "Reprojected"]
    axes = np.empty((n, 3), dtype=object)

    for i, survey in enumerate(surveys):
        entry = images.get(survey, {})
        panels = [entry.get("data"), entry.get("convolved"), entry.get("reprojected")]
        fam_color = _family_color(survey)
        meta = SURVEY_DEFAULTS.get(survey, {})
        psf = meta.get("psf_fwhm")

        last_im = None  # for the row colorbar
        for j, data in enumerate(panels):
            projection = wcs if (j == 2 and wcs is not None) else None
            ax = fig.add_subplot(gs[i, j], projection=projection)
            axes[i, j] = ax

            if i == 0:
                ax.set_title(col_titles[j], fontsize=12, color="white", pad=6)

            if projection is not None:
                # WCS axes: show RA/Dec but keep them compact
                ax.coords[0].set_axislabel("RA", fontsize=7)
                ax.coords[1].set_axislabel("Dec", fontsize=7)
                ax.coords[0].set_ticklabel(fontsize=6, exclude_overlapping=True)
                ax.coords[1].set_ticklabel(fontsize=6, exclude_overlapping=True)
                ax.coords[0].set_ticks(number=3)
                ax.coords[1].set_ticks(number=3)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if data is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="0.55", fontsize=11,
                        style="italic")
                _style_border(ax, fam_color)
                continue

            nrm = _norm(data)
            if nrm:
                im = ax.imshow(data, origin="lower", cmap="inferno", **nrm)
            else:
                im = ax.imshow(np.zeros_like(data), origin="lower", cmap="inferno")
            last_im = im

            if j == 2:
                for name, m in (masks_reproj or {}).items():
                    if m is None or not np.any(m):
                        continue
                    ax.contour(m.astype(float), levels=[0.5],
                               colors=[_COMP_COLORS.get(name, "w")], linewidths=1.3)
                # compass on the first reprojected panel only
                if i == 0:
                    _compass(ax)

            _style_border(ax, fam_color)

        # kernel inset on the convolved panel
        kernel = entry.get("kernel")
        if kernel is not None and axes[i, 1] is not None:
            inset = axes[i, 1].inset_axes([0.64, 0.64, 0.34, 0.34])
            inset.imshow(kernel, origin="lower", cmap="viridis")
            inset.set_xticks([])
            inset.set_yticks([])
            for sp in inset.spines.values():
                sp.set_color("white")
                sp.set_linewidth(0.8)
            inset.set_title("PSF kernel", fontsize=6, color="white", pad=2)

        # row colorbar on the right of the reprojected panel
        if last_im is not None and axes[i, 2] is not None:
            cax = axes[i, 2].inset_axes([1.04, 0.05, 0.035, 0.90])
            cb = fig.colorbar(last_im, cax=cax)
            cb.ax.tick_params(labelsize=6, colors="white")
            cb.outline.set_edgecolor("white")

        # row label: survey + PSF FWHM, color-coded by family
        label = survey if psf is None else f'{survey}\nPSF {psf:g}"'
        if axes[i, 0] is not None:
            axes[i, 0].text(-0.22, 0.5, label, transform=axes[i, 0].transAxes,
                            rotation=90, ha="center", va="center",
                            fontsize=10, color=fam_color, family="monospace")

    coord_str = ""
    if wcs is not None:
        try:
            ny, nx = wcs.array_shape if wcs.array_shape else (None, None)
            if nx and ny:
                sky = wcs.pixel_to_world(nx / 2.0, ny / 2.0)
                coord_str = f"  ({sky.ra.deg:.5f}, {sky.dec.deg:+.5f})"
        except Exception:
            coord_str = ""
    fig.suptitle(f"{target_id} — processing grid{coord_str}",
                 fontsize=15, color="white", y=0.985)

    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.style.use("default")


# ===========================================================================
# 2. SED
# ===========================================================================
# deblend method -> (marker, label tag)
_DEBLEND_MARKERS = {"ratio_prior": "s", "tphot": "*", "combined": "^",
                    "comb": "^"}


def plot_sed(target_id, comp_seds, out_path, deblend_seds=None, unwise_forced=None):
    """SED per component + total, log-log mJy vs micron, dark background.

    comp_seds: dict component -> list of {"lambda_eff", "flux_mjy",
    "flux_err_mjy", "survey", "blend_flag"} dicts.

    deblend_seds: optional dict method -> comp_seds (same format). Overlaid as
    dotted lines in the component color with method-specific markers.
    """
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # collect global wavelength range for the band shading
    all_lam = []
    for points in comp_seds.values():
        if points:
            all_lam += [p.get("lambda_eff") for p in points
                        if p.get("lambda_eff") is not None]
    lam_lo = min(all_lam) * 0.6 if all_lam else 0.1
    lam_hi = max(all_lam) * 1.6 if all_lam else 40.0

    # shade wavelength regions
    for lo, hi, name, col in _WAVE_BANDS:
        lo_c, hi_c = max(lo, lam_lo), min(hi, lam_hi)
        if hi_c <= lo_c:
            continue
        ax.axvspan(lo_c, hi_c, color=col, alpha=0.07, zorder=0)
        ax.text(np.sqrt(lo_c * hi_c), 0.97, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=col, alpha=0.8)

    # per-component totals (for the summary box / legend), indexed by survey
    band_totals = {}

    handles_labels = []
    for comp, points in comp_seds.items():
        if not points:
            continue
        lam = np.array([p.get("lambda_eff", np.nan) for p in points], dtype=float)
        flux = np.array([p.get("flux_mjy", np.nan) for p in points], dtype=float)
        err = np.array([p.get("flux_err_mjy", np.nan) for p in points], dtype=float)
        blend = np.array([bool(p.get("blend_flag", False)) for p in points])
        surveys = [p.get("survey") for p in points]

        good = np.isfinite(lam) & np.isfinite(flux) & (flux > 0)
        if not np.any(good):
            continue
        order = np.argsort(lam[good])
        color = _COMP_COLORS.get(comp, None)
        lg, fg, eg, bg = lam[good][order], flux[good][order], err[good][order], blend[good][order]

        # connecting line + filled markers for non-blended points
        line = ax.errorbar(lg, fg, yerr=eg, fmt="-", color=color, capsize=4,
                           lw=1.6, alpha=0.8, zorder=3)
        ax.scatter(lg[~bg], fg[~bg], color=color, s=42, marker="o",
                   edgecolors="black", linewidths=0.5, zorder=4)
        # blended points: open diamonds
        if np.any(bg):
            ax.scatter(lg[bg], fg[bg], facecolors="none", edgecolors=color,
                       s=70, marker="D", linewidths=1.6, zorder=5)

        # total flux at W1 if present
        w1 = next((p["flux_mjy"] for p in points
                   if p.get("survey") == "WISE_W1"
                   and np.isfinite(p.get("flux_mjy", np.nan))), None)
        lbl = _COMP_LABELS.get(comp, comp)
        if w1 is not None:
            lbl += f"  (W1={w1:.3g} mJy)"
        handles_labels.append((line, lbl))

        for s, f in zip(surveys, flux):
            if s is not None and np.isfinite(f):
                band_totals.setdefault(s, 0.0)
                if comp != "total":
                    band_totals[s] += f

    # --- deblended overlays (dotted, method-specific markers) ---------------
    if deblend_seds:
        for method, dseds in deblend_seds.items():
            if not isinstance(dseds, dict):
                continue
            marker = _DEBLEND_MARKERS.get(method, "P")
            for comp, points in dseds.items():
                if not points:
                    continue
                lam = np.array([p.get("lambda_eff", np.nan) for p in points],
                               dtype=float)
                flux = np.array([p.get("flux_mjy", np.nan) for p in points],
                                dtype=float)
                err = np.array([p.get("flux_err_mjy", np.nan) for p in points],
                               dtype=float)
                good = np.isfinite(lam) & np.isfinite(flux) & (flux > 0)
                if not np.any(good):
                    continue
                order = np.argsort(lam[good])
                lg, fg, eg = lam[good][order], flux[good][order], err[good][order]
                color = _COMP_COLORS.get(comp, "white")
                dline = ax.errorbar(lg, fg, yerr=eg, fmt=":", color=color,
                                    alpha=0.6, lw=1.4, capsize=3, marker=marker,
                                    markersize=9, markeredgecolor="black",
                                    markeredgewidth=0.5, zorder=6)
                lbl = f"{_COMP_LABELS.get(comp, comp)} [{method}]"
                handles_labels.append((dline, lbl))

    # blend marker proxy for legend
    blend_proxy = ax.scatter([], [], facecolors="none", edgecolors="white",
                             s=70, marker="D", linewidths=1.6, label="blended band")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lam_lo, lam_hi)
    ax.set_xlabel(r"$\lambda_{\rm eff}$ [$\mu$m]", fontsize=12)
    ax.set_ylabel("flux density [mJy]", fontsize=12)
    ax.set_title(f"{target_id} — spectral energy distribution", fontsize=14, pad=28)
    ax.grid(True, which="both", alpha=0.18)

    handles = [h for h, _ in handles_labels] + [blend_proxy]
    labels = [l for _, l in handles_labels] + ["blended band"]
    ax.legend(handles, labels, loc="lower center", fontsize=9, framealpha=0.25,
              ncol=1)

    # second x-axis on top: frequency in GHz
    def _lam_to_ghz(l):
        l = np.asarray(l, dtype=float)
        return np.where(l > 0, _C_UM_HZ / np.where(l > 0, l, np.nan) / 1e9, np.inf)

    def _ghz_to_lam(f):
        f = np.asarray(f, dtype=float)
        return np.where(f > 0, _C_UM_HZ / (np.where(f > 0, f, np.nan) * 1e9), np.inf)

    axf = ax.secondary_xaxis("top", functions=(_lam_to_ghz, _ghz_to_lam))
    axf.set_xlabel(r"$\nu$ [GHz]", fontsize=11)
    axf.tick_params(labelsize=8)

    # summary text box: total flux per band
    if band_totals:
        lines = ["band       flux[mJy]"]
        for s in sorted(band_totals, key=lambda k: SURVEY_DEFAULTS.get(k, {}).get("lambda_eff", 0)):
            lines.append(f"{s:<11s}{band_totals[s]:.3g}")
        ax.text(0.015, 0.015, "\n".join(lines), transform=ax.transAxes,
                va="bottom", ha="left", family="monospace", fontsize=7.5,
                bbox=dict(boxstyle="round", facecolor="#111111", edgecolor="#555555",
                          alpha=0.8))

    # unWISE forced photometry — cross markers per component
    _WISE_LAM = {"WISE_W1": 3.4, "WISE_W2": 4.6}
    if unwise_forced:
        uw_proxy = None
        for band, lam in _WISE_LAM.items():
            band_entry = unwise_forced.get(band, {})
            for comp, entry in band_entry.items():
                f = entry.get("flux_mjy", np.nan)
                e = entry.get("flux_err_mjy", np.nan)
                if not np.isfinite(f) or f <= 0:
                    continue
                color = _COMP_COLORS.get(comp, "white")
                uw_proxy = ax.errorbar(
                    lam, f, yerr=e if np.isfinite(e) else None,
                    fmt="x", color=color, ms=11, mew=2.5,
                    capsize=4, alpha=0.95, zorder=6,
                    label=f"{_COMP_LABELS.get(comp, comp)} [unWISE forced]",
                )
        if uw_proxy is not None:
            # add a single legend entry for the marker style
            ax.scatter([], [], marker="x", color="white", s=80, linewidths=2.5,
                       label="unWISE forced phot", zorder=6)

    # rebuild legend to include new entries
    ax.legend(loc="lower center", fontsize=9, framealpha=0.25, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.style.use("default")


def _iter_component_seds(comp_seds_tbl):
    """Yield (component_name, list_of_point_dicts)."""
    for comp, points in comp_seds_tbl.items():
        if not isinstance(points, (list, tuple)):
            continue
        yield comp, points


# ===========================================================================
# 3. summary data-sheet
# ===========================================================================
def plot_summary_panel(target_id, target, surveys, photometry, comp_seds_tbl,
                       xmask_result, out_path):
    """Dark data-sheet: mini SED, info block, styled flux table."""
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(11.5, 9))
    fig.suptitle(f"photextra_pipeline — {target_id}", fontsize=17, y=0.97)

    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25],
                          left=0.06, right=0.96, top=0.90, bottom=0.05,
                          wspace=0.22, hspace=0.28)

    components = list(_iter_component_seds(comp_seds_tbl))

    n_comp = getattr(xmask_result, "n_components", None)
    is_merger = bool(getattr(xmask_result, "is_merger", False))
    sep = getattr(xmask_result, "separation_arcsec", None)

    # --- top-left: compact SED ------------------------------------------
    ax_sed = fig.add_subplot(gs[0, 0])
    _mini_sed(ax_sed, components)

    # --- top-right: info block ------------------------------------------
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    sep_str = f'{sep:.2f}"' if isinstance(sep, (int, float)) else "n/a"
    info_lines = [
        f"target      : {target_id}",
        f"RA, Dec     : {_fmt_coords(target.get('ra'), target.get('dec'))}",
        f"surveys     : {len(surveys)}",
        f"n_components: {n_comp}",
        f"is_merger   : {is_merger}",
        f"separation  : {sep_str}",
    ]
    # per-band blend flags
    blend_bands = []
    for comp, points in components:
        for p in points:
            if p.get("blend_flag"):
                blend_bands.append(p.get("survey"))
    blend_bands = sorted(set(b for b in blend_bands if b))
    info_lines.append("blended     : " + (", ".join(blend_bands) if blend_bands else "none"))

    ax_info.text(0.02, 0.97, "image / detection info", va="top", ha="left",
                 fontsize=12, color="#ffd24d", weight="bold")
    ax_info.text(0.02, 0.84, "\n".join(info_lines), va="top", ha="left",
                 family="monospace", fontsize=11)

    if is_merger:
        ax_info.text(0.98, 0.99, " MERGER ", va="top", ha="right",
                     fontsize=12, color="white", weight="bold",
                     bbox=dict(boxstyle="round", facecolor="#cc1f1f",
                               edgecolor="white", pad=0.4))

    # --- bottom: flux table ---------------------------------------------
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")
    _flux_table(fig, ax_tbl, surveys, components)

    fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.style.use("default")


def _mini_sed(ax, components):
    """Compact SED rendered into an existing axis (no legend clutter)."""
    all_lam = []
    for _, points in components:
        all_lam += [p.get("lambda_eff") for p in points
                    if p.get("lambda_eff") is not None]
    lam_lo = min(all_lam) * 0.6 if all_lam else 0.1
    lam_hi = max(all_lam) * 1.6 if all_lam else 40.0

    for lo, hi, name, col in _WAVE_BANDS:
        lo_c, hi_c = max(lo, lam_lo), min(hi, lam_hi)
        if hi_c > lo_c:
            ax.axvspan(lo_c, hi_c, color=col, alpha=0.06, zorder=0)

    plotted = False
    for comp, points in components:
        if not points:
            continue
        lam = np.array([p.get("lambda_eff", np.nan) for p in points], dtype=float)
        flux = np.array([p.get("flux_mjy", np.nan) for p in points], dtype=float)
        blend = np.array([bool(p.get("blend_flag", False)) for p in points])
        good = np.isfinite(lam) & np.isfinite(flux) & (flux > 0)
        if not np.any(good):
            continue
        order = np.argsort(lam[good])
        color = _COMP_COLORS.get(comp, None)
        lg, fg, bg = lam[good][order], flux[good][order], blend[good][order]
        ax.plot(lg, fg, "-", color=color, lw=1.4, alpha=0.85, zorder=3,
                label=_COMP_LABELS.get(comp, comp))
        ax.scatter(lg[~bg], fg[~bg], color=color, s=22, zorder=4)
        if np.any(bg):
            ax.scatter(lg[bg], fg[bg], facecolors="none", edgecolors=color,
                       s=36, marker="D", linewidths=1.2, zorder=5)
        plotted = True

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lam_lo, lam_hi)
    ax.set_xlabel(r"$\lambda_{\rm eff}$ [$\mu$m]", fontsize=10)
    ax.set_ylabel("flux [mJy]", fontsize=10)
    ax.set_title("SED", fontsize=12)
    ax.grid(True, which="both", alpha=0.15)
    ax.tick_params(labelsize=8)
    if plotted:
        ax.legend(fontsize=8, framealpha=0.25)
    else:
        ax.text(0.5, 0.5, "no flux data", transform=ax.transAxes,
                ha="center", va="center", color="0.55", style="italic")


def _flux_table(fig, ax, surveys, components):
    """Styled flux table with alternating row colors and a blend-flag column."""
    comp_names = [c for c, _ in components]
    header = ["band", r"$\lambda$[$\mu$m]"] + [_COMP_LABELS.get(c, c) for c in comp_names] + ["blend"]

    rows = []
    cell_colors = []
    row_dark = "#1c1c1c"
    row_light = "#2b2b2b"

    for k, survey in enumerate(surveys):
        meta = SURVEY_DEFAULTS.get(survey, {})
        lam = meta.get("lambda_eff")
        row = [survey, f"{lam:g}" if lam is not None else "-"]
        blended = False
        for comp, points in components:
            match = next((p for p in points if p.get("survey") == survey), None)
            if match and np.isfinite(match.get("flux_mjy", np.nan)):
                row.append(f"{match['flux_mjy']:.3g}")
                if match.get("blend_flag"):
                    blended = True
            else:
                row.append("-")
        row.append("B" if blended else "-")
        rows.append(row)

        bg = row_light if (k % 2) else row_dark
        cell_colors.append([bg] * len(row))

    if not rows:
        ax.text(0.5, 0.5, "no photometry", transform=ax.transAxes,
                ha="center", va="center", color="0.55", style="italic")
        return

    table = ax.table(cellText=rows, colLabels=header, loc="center",
                     cellLoc="center", cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    n_cols = len(header)
    # style header + color-code component columns to match SED colors
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.get_text().set_fontfamily("monospace")
        if r == 0:
            cell.set_facecolor("#000000")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.set_height(cell.get_height() * 1.1)
            # component header colors
            comp_idx = c - 2
            if 0 <= comp_idx < len(comp_names):
                cell.get_text().set_color(
                    _COMP_COLORS.get(comp_names[comp_idx], "white"))
        else:
            comp_idx = c - 2
            if 0 <= comp_idx < len(comp_names):
                cell.get_text().set_color(
                    _COMP_COLORS.get(comp_names[comp_idx], "white"))
            else:
                cell.get_text().set_color("#dddddd")


# ===========================================================================
# 4. clean galaxy panels: gal1 | gal2 | combined (per band)
# ===========================================================================
def plot_galaxy_panels(target_id, surveys, images, masks_reproj, out_path):
    """Per-band clean cutouts: gal1 | gal2 | combined (no contours/overlays).

    Each panel shows the reprojected image with pixels outside the component
    mask set to NaN so only the galaxy light is visible against the background.
    The 'combined' column shows gal1+gal2 union mask together.
    """
    plt.style.use("dark_background")
    n = max(len(surveys), 1)

    # determine which components actually exist for this target
    comp_keys = [k for k in ["gal1", "gal2"] if k in (masks_reproj or {})]
    if not comp_keys:
        comp_keys = [k for k in (masks_reproj or {}) if k != "total"]
    # only add a 'combined' union column when there really are >=2 components;
    # for a single galaxy the combined column would just duplicate gal1 and its
    # "gal 1 + 2" label would falsely imply a second galaxy exists.
    cols = comp_keys + (["combined"] if len(comp_keys) >= 2 else [])
    col_labels = {
        "gal1": "galaxy 1" if len(comp_keys) >= 2 else "galaxy",
        "gal2": "galaxy 2",
        "combined": "gal 1 + 2",
    }

    ncols = len(cols)
    fig = plt.figure(figsize=(3.8 * ncols, 3.2 * n))
    gs = fig.add_gridspec(n, ncols, wspace=0.06, hspace=0.12,
                          left=0.10, right=0.98, top=0.94, bottom=0.03)

    # build combined mask (union of all components)
    combined_mask = None
    for m in (masks_reproj or {}).values():
        if m is not None and np.any(m):
            combined_mask = m.copy() if combined_mask is None else (combined_mask | m)

    def _masked_image(data, mask):
        """Return data with NaN outside mask; preserve background level."""
        if data is None or mask is None or not np.any(mask):
            return None
        out = np.full_like(data, np.nan, dtype=float)
        out[mask] = data[mask]
        return out

    for i, survey in enumerate(surveys):
        entry = images.get(survey, {})
        data = entry.get("reprojected")
        fam_color = _family_color(survey)
        meta = SURVEY_DEFAULTS.get(survey, {})
        psf = meta.get("psf_fwhm")

        # compute a shared norm for all columns of this row (same stretch)
        row_data = []
        for col in cols:
            if col == "combined":
                md = _masked_image(data, combined_mask)
            else:
                md = _masked_image(data, (masks_reproj or {}).get(col))
            row_data.append(md)

        # global ZScale on the combined / all-pixel data for consistent stretch
        nrm = _norm(data)

        for j, (col, md) in enumerate(zip(cols, row_data)):
            ax = fig.add_subplot(gs[i, j])

            if i == 0:
                ax.set_title(col_labels.get(col, col), fontsize=13,
                             color=_COMP_COLORS.get(col, "white") if col != "combined" else "#ffd24d",
                             pad=6)
            ax.set_xticks([])
            ax.set_yticks([])

            if md is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, color="0.45",
                        fontsize=11, style="italic")
                _style_border(ax, fam_color)
                continue

            if nrm:
                ax.imshow(md, origin="lower", cmap="inferno", **nrm)
            else:
                ax.imshow(md, origin="lower", cmap="inferno")

            _style_border(ax, fam_color)

        # row label: survey + PSF FWHM on the left
        label = survey if psf is None else f'{survey}\nPSF {psf:g}"'
        fig.get_axes()[i * ncols].text(
            -0.22, 0.5, label,
            transform=fig.get_axes()[i * ncols].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=10, color=fam_color, family="monospace",
        )

    fig.suptitle(f"{target_id} — galaxy components (clean cutouts)",
                 fontsize=15, color="white", y=0.985)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.style.use("default")


# ===========================================================================
# 5. comprehensive publication-quality figure
# ===========================================================================
# clean short labels per survey for rows / table columns
_SHORT_LABELS = {
    "GALEX_FUV": "FUV", "GALEX_NUV": "NUV",
    "Legacy_g": "g", "Legacy_r": "r", "Legacy_z": "z",
    "WISE_W1": "W1", "WISE_W2": "W2",
}
# canonical band order
_COMP_FIG_ORDER = ["GALEX_FUV", "GALEX_NUV", "Legacy_g", "Legacy_r",
                   "Legacy_z", "WISE_W1", "WISE_W2"]


def _img_norm(data):
    """Asinh-stretched norm using ZScale limits, falling back to percentile."""
    if data is None:
        return None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    try:
        vmin, vmax = ZScaleInterval().get_limits(finite)
    except Exception:
        vmin, vmax = None, None
    if vmin is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        try:
            vmin, vmax = PercentileInterval(98.5).get_limits(finite)
        except Exception:
            return None
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return None
    try:
        return AsinhNorm(vmin=vmin, vmax=vmax)
    except Exception:
        return None


def _img_cmap(survey):
    """gray_r for Legacy, inferno for GALEX/WISE."""
    return "gray_r" if survey.startswith("Legacy") else "inferno"


def _fmt_flux_cell(f, e):
    if not np.isfinite(f):
        return "---"
    if np.isfinite(e):
        return f"{f:.3f} ± {e:.3f}"
    return f"{f:.3f}"


def plot_comprehensive(target_id, surveys, images_conv, masks_reproj, comp_seds,
                       photometry, seg_result, deblend_ratio=None,
                       deblend_tphot=None, deblend_comb=None, unwise_forced=None,
                       galaxy_table=None, out_path=None):
    """One big dark-background data sheet: image grid + SED + flux table.

    Catches all exceptions internally and logs a warning rather than crashing
    the pipeline.
    """
    try:
        _plot_comprehensive_impl(
            target_id, surveys, images_conv, masks_reproj, comp_seds,
            photometry, seg_result, deblend_ratio, deblend_tphot, deblend_comb,
            unwise_forced, galaxy_table, out_path,
        )
    except Exception as exc:
        logger.warning("comprehensive plot failed: %s", exc, exc_info=True)
        try:
            plt.style.use("default")
        except Exception:
            pass


def masks_on_band_grid(seg_result, entry, fallback_masks):
    """Component masks reprojected onto ONE band's own pixel grid.

    BUG FIX (mask misplacement): the comprehensive figure displays each
    band's NATIVE image (``entry['data']``, e.g. Legacy at 0.262"/px,
    ~230x230 px) but the pipeline hands the plots ``masks_reproj`` on the
    small COMMON grid (e.g. 45x45 px at 1.375"/px). Contouring the
    common-grid masks on the native image drew them tiny in the bottom-left
    corner — visible only on Legacy/optical panels of multi-galaxy targets,
    because single targets draw no contours and the WISE grid nearly
    matches the common grid. The masks must be reprojected from their
    native grid (``seg_result.masks`` + ``seg_result.wcs``) to each band's
    own WCS/shape; ``fallback_masks`` (the common-grid set) is only correct
    when the band image is already on the common grid.
    """
    masks_native = getattr(seg_result, "masks", {}) or {}
    seg_wcs = getattr(seg_result, "wcs", None)
    data = entry.get("data") if isinstance(entry, dict) else None
    band_wcs = entry.get("wcs") if isinstance(entry, dict) else None
    if not masks_native or seg_wcs is None or band_wcs is None or data is None:
        return fallback_masks
    try:
        from .reprojection import reproject_masks as _reproj_masks
        return _reproj_masks(masks_native, seg_wcs, band_wcs, data.shape)
    except Exception as exc:
        logger.warning("mask reprojection to band grid failed (%s); "
                       "skipping contours", exc)
        return {}


def _plot_comprehensive_impl(target_id, surveys, images_conv, masks_reproj,
                             comp_seds, photometry, seg_result, deblend_ratio,
                             deblend_tphot, deblend_comb, unwise_forced,
                             galaxy_table, out_path):
    deblend_ratio = deblend_ratio or {}
    deblend_tphot = deblend_tphot or {}
    deblend_comb = deblend_comb or {}
    unwise_forced = unwise_forced or {}
    masks_reproj = masks_reproj or {}

    # ordered bands
    bands = [s for s in _COMP_FIG_ORDER if s in surveys]
    bands += [s for s in surveys if s not in _COMP_FIG_ORDER]
    n_band = max(len(bands), 1)

    n_comp = getattr(seg_result, "n_components", None)
    is_merger = bool(getattr(seg_result, "is_merger", False))
    sep = getattr(seg_result, "separation_arcsec", None)
    masks_native = getattr(seg_result, "masks", {}) or {}

    # component keys (gal1, gal2, ...) — singles have only gal1
    comp_keys = [k for k in masks_native.keys() if k != "total"]
    if not comp_keys:
        comp_keys = ["gal1"]
    single = len(comp_keys) < 2

    DARK = "#0d0d0d"
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 14), dpi=150, facecolor=DARK)

    # outer layout: title (thin) | body | table (bottom)
    outer = fig.add_gridspec(
        3, 1, height_ratios=[0.05, 1.0, 0.22],
        left=0.04, right=0.985, top=0.97, bottom=0.03, hspace=0.12,
    )

    # ----- title bar -----
    ax_title = fig.add_subplot(outer[0])
    ax_title.axis("off")
    cls = "MERGER" if is_merger else "SINGLE"
    sep_str = f'sep={sep:.1f}"' if isinstance(sep, (int, float)) else "sep=n/a"
    title = f"{target_id}    •    {cls}    •    {sep_str}    •    n={n_comp}"
    ax_title.text(0.0, 0.5, title, ha="left", va="center", fontsize=20,
                  color="white", weight="bold", family="monospace")

    # ----- body: image block (left) + SED (right) -----
    img_cols = 1 if single else 3
    body = outer[1].subgridspec(
        1, 2, width_ratios=[img_cols * 0.95, 2.0], wspace=0.10,
    )
    img_block = body[0].subgridspec(n_band, img_cols, wspace=0.04, hspace=0.06)

    col_comps = comp_keys[:2] if not single else comp_keys[:1]
    # for mergers columns are [gal1, gal2, combined]
    if single:
        col_defs = [(col_comps[0], None)]   # (label_comp, mask) - show full image
    else:
        col_defs = [(col_comps[0], masks_reproj.get(col_comps[0])),
                    (col_comps[1], masks_reproj.get(col_comps[1])),
                    ("combined", None)]

    col_titles = ([_COMP_LABELS.get(col_comps[0], col_comps[0])] if single
                  else [_COMP_LABELS.get(col_comps[0], col_comps[0]),
                        _COMP_LABELS.get(col_comps[1], col_comps[1]),
                        "combined"])

    for i, survey in enumerate(bands):
        entry = images_conv.get(survey, {}) if isinstance(images_conv, dict) else {}
        data = entry.get("data") if isinstance(entry, dict) else None
        fam_color = _family_color(survey)

        # masks for THIS band's pixel grid (see masks_on_band_grid — fixes
        # the tiny/misplaced mask contours on native-resolution panels)
        band_masks = masks_reproj
        if not single:
            band_masks = masks_on_band_grid(seg_result, entry, masks_reproj)
        short = _SHORT_LABELS.get(survey, survey)
        phot = photometry.get(survey, {}) if isinstance(photometry, dict) else {}
        blended = bool(phot.get("blend_flag", False))
        nrm = _img_norm(data)
        cmap = _img_cmap(survey)

        for j in range(img_cols):
            ax = fig.add_subplot(img_block[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("black")

            if i == 0:
                ax.set_title(col_titles[j], fontsize=12, pad=4,
                             color=(_COMP_COLORS.get(col_defs[j][0], "white")
                                    if col_defs[j][0] in _COMP_COLORS else "#ffd24d"))

            if data is None:
                ax.imshow(np.zeros((10, 10)), cmap="gray", vmin=0, vmax=1,
                          origin="lower")
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="0.5", style="italic",
                        fontsize=10)
                _style_border(ax, fam_color, lw=1.5)
                continue

            if nrm is not None:
                ax.imshow(data, origin="lower", cmap=cmap, norm=nrm)
            else:
                ax.imshow(data, origin="lower", cmap=cmap)

            # mask outlines (mergers only), on this band's own pixel grid
            if not single:
                for cname, ccolor in ((col_comps[0], "#ff5d5d"),
                                      (col_comps[1], "#5db4ff")):
                    m = (band_masks or {}).get(cname)
                    if m is not None and np.any(m):
                        try:
                            ax.contour(m.astype(float), levels=[0.5],
                                       colors=[ccolor], linewidths=1.1,
                                       alpha=0.9)
                        except Exception:
                            pass

            # blend indicator
            if blended:
                ax.text(0.97, 0.95, "⬢ BLENDED", ha="right", va="top",
                        transform=ax.transAxes, fontsize=7.5, color="#ffb000",
                        weight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#2a1a00",
                                  edgecolor="#ffb000", alpha=0.8, linewidth=0.6))

            _style_border(ax, fam_color, lw=1.5)

            # band label on the left-most column
            if j == 0:
                meta = SURVEY_DEFAULTS.get(survey, {})
                psf = meta.get("psf_fwhm")
                lbl = short if psf is None else f'{short}\nPSF {psf:g}"'
                ax.text(-0.16, 0.5, lbl, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=10, color=fam_color,
                        family="monospace")
            # family color bar on the right-most column
            if j == img_cols - 1:
                ax.add_patch(plt.Rectangle((1.01, 0.0), 0.03, 1.0,
                             transform=ax.transAxes, color=fam_color,
                             clip_on=False))

    # ----- SED panel (right, spans all rows) -----
    ax_sed = fig.add_subplot(body[1])
    _comprehensive_sed(ax_sed, bands, comp_seds, photometry, deblend_comb,
                       deblend_ratio, unwise_forced, comp_keys)

    # ----- info box -----
    cls_word = "merger" if is_merger else "single"
    if is_merger and isinstance(sep, (int, float)) and sep > 0:
        cls_word = "merger" if sep < 15 else "companion"
    info = (f"n_components={n_comp}, "
            f"sep={sep:.1f}\", " if isinstance(sep, (int, float))
            else f"n_components={n_comp}, sep=n/a, ")
    info += f"class={cls_word}"
    ax_sed.text(0.015, 0.02, info, transform=ax_sed.transAxes, fontsize=9,
                color="#cccccc", va="bottom", ha="left", family="monospace",
                bbox=dict(boxstyle="round", facecolor="#111111",
                          edgecolor="#444444", alpha=0.85))

    # ----- flux table (bottom) -----
    ax_tbl = fig.add_subplot(outer[2])
    ax_tbl.axis("off")
    _comprehensive_table(ax_tbl, bands, comp_keys, photometry, deblend_comb,
                         unwise_forced)

    if out_path is not None:
        fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    plt.style.use("default")


def _comprehensive_sed(ax, bands, comp_seds, photometry, deblend_comb,
                       deblend_ratio, unwise_forced, comp_keys=None):
    """Render the SED into the given axis.

    ``comp_keys`` is the authoritative list of components actually present for
    this target (e.g. ["gal1"] for a single, ["gal1", "gal2"] for a pair). The
    deblend overlay markers are drawn only for those components so a single
    target never gets a phantom gal2 marker.
    """
    ax.set_facecolor("#0d0d0d")
    lam_lo, lam_hi = 0.1, 6.0

    # background shading
    shade = [(0.1, 0.3, "#b44fff"), (0.4, 1.0, "#00c8c8"), (3.0, 5.0, "#ff8800")]
    for lo, hi, col in shade:
        ax.axvspan(lo, hi, color=col, alpha=0.08, zorder=0)

    handles_labels = []
    for comp, points in (comp_seds or {}).items():
        if not points:
            continue
        color = _COMP_COLORS.get(comp, "#cccccc")
        lam = np.array([p.get("lambda_eff", np.nan) for p in points], float)
        flux = np.array([p.get("flux_mjy", np.nan) for p in points], float)
        err = np.array([p.get("flux_err_mjy", np.nan) for p in points], float)
        blend = np.array([bool(p.get("blend_flag", False)) for p in points])
        good = np.isfinite(lam) & np.isfinite(flux) & (flux > 0)
        if not np.any(good):
            continue
        order = np.argsort(lam[good])
        lg, fg, eg, bg = (lam[good][order], flux[good][order],
                          err[good][order], blend[good][order])
        line = ax.errorbar(lg, fg, yerr=eg, fmt="-", color=color, lw=1.4,
                           alpha=0.6, capsize=3, zorder=3)
        # non-blended: filled circles
        if np.any(~bg):
            ax.scatter(lg[~bg], fg[~bg], color=color, s=48, marker="o",
                       edgecolors="black", linewidths=0.5, zorder=5)
        # blended mask points: open faded circles
        if np.any(bg):
            ax.scatter(lg[bg], fg[bg], facecolors="none", edgecolors=color,
                       s=55, marker="o", linewidths=1.3, alpha=0.4, zorder=4)
        handles_labels.append((line, _COMP_LABELS.get(comp, comp)))

    # comb deblended (filled diamonds) and ratio (x markers) — only for the
    # components that actually exist for this target (never a phantom gal2)
    if comp_keys:
        deblend_comps = [c for c in comp_keys if c != "total"]
    else:
        deblend_comps = [c for c in _COMP_COLORS if c != "total"]
    for survey in bands:
        meta = SURVEY_DEFAULTS.get(survey, {})
        lam = meta.get("lambda_eff")
        if lam is None:
            continue
        for comp in deblend_comps:
            color = _COMP_COLORS.get(comp, "#cccccc")
            cm = deblend_comb.get(survey, {}).get(comp) if isinstance(deblend_comb.get(survey), dict) else None
            if isinstance(cm, dict):
                f = cm.get("flux_mjy", np.nan)
                e = cm.get("flux_err_mjy", np.nan)
                if np.isfinite(f) and f > 0:
                    ax.errorbar(lam, f, yerr=e if np.isfinite(e) else None,
                                fmt="D", color=color, ms=8, capsize=3,
                                markeredgecolor="black", markeredgewidth=0.5,
                                zorder=7)
            rm = deblend_ratio.get(survey, {}).get(comp) if isinstance(deblend_ratio.get(survey), dict) else None
            if isinstance(rm, dict):
                f = rm.get("flux_mjy", np.nan)
                if np.isfinite(f) and f > 0:
                    ax.scatter(lam * 1.04, f, marker="x", color=color, s=55,
                               linewidths=1.8, zorder=6, alpha=0.85)

    # unWISE catalog matches (+ markers, offset)
    for survey, band_entry in (unwise_forced or {}).items():
        meta = SURVEY_DEFAULTS.get(survey, {})
        lam = meta.get("lambda_eff")
        if lam is None or not isinstance(band_entry, dict):
            continue
        for comp, entry in band_entry.items():
            if not isinstance(entry, dict):
                continue
            f = entry.get("flux_mjy", np.nan)
            e = entry.get("flux_err_mjy", np.nan)
            if np.isfinite(f) and f > 0:
                color = _COMP_COLORS.get(comp, "#cccccc")
                ax.errorbar(lam * 0.96, f, yerr=e if np.isfinite(e) else None,
                            fmt="+", color=color, ms=13, mew=2.2, capsize=3,
                            zorder=8)

    # PSF FWHM indicator bars at bottom
    try:
        ymin, ymax = ax.get_ylim()
    except Exception:
        ymin, ymax = 1e-3, 1.0

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lam_lo, lam_hi)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]", fontsize=12, color="white")
    ax.set_ylabel("flux density [mJy]", fontsize=12, color="white")
    ax.set_title("spectral energy distribution", fontsize=14, color="white",
                 pad=26)
    ax.grid(True, which="both", alpha=0.15)
    ax.tick_params(colors="white")

    # marker-style legend proxies
    proxies = [
        plt.Line2D([], [], marker="o", color="w", ls="none", label="direct (clean)",
                   mec="black", ms=8),
        plt.Line2D([], [], marker="o", mfc="none", mec="w", color="w", ls="none",
                   alpha=0.5, label="blended (mask)", ms=8),
        plt.Line2D([], [], marker="D", color="w", ls="none", label="comb deblend",
                   mec="black", ms=8),
        plt.Line2D([], [], marker="x", color="w", ls="none", label="ratio prior",
                   ms=9, mew=1.8),
        plt.Line2D([], [], marker="+", color="w", ls="none", label="unWISE", ms=11,
                   mew=2.0),
    ]
    comp_handles = [h for h, _ in handles_labels]
    comp_labels = [l for _, l in handles_labels]
    leg = ax.legend(comp_handles + proxies,
                    comp_labels + [p.get_label() for p in proxies],
                    loc="upper right", fontsize=8.5, framealpha=0.3, ncol=1)
    if leg is not None:
        leg.get_frame().set_edgecolor("#555555")

    # top frequency axis
    def _lam_to_ghz(l):
        l = np.asarray(l, float)
        return np.where(l > 0, _C_UM_HZ / np.where(l > 0, l, np.nan) / 1e9, np.inf)

    def _ghz_to_lam(f):
        f = np.asarray(f, float)
        return np.where(f > 0, _C_UM_HZ / (np.where(f > 0, f, np.nan) * 1e9), np.inf)

    try:
        axf = ax.secondary_xaxis("top", functions=(_lam_to_ghz, _ghz_to_lam))
        axf.set_xlabel(r"$\nu$ [GHz]", fontsize=11, color="white")
        axf.tick_params(labelsize=8, colors="white")
    except Exception:
        pass

    # PSF FWHM indicator bars near bottom of the panel (in axis-fraction y)
    ymin, ymax = ax.get_ylim()
    ybar = ymin * (ymax / ymin) ** 0.04 if ymin > 0 else ymin
    for survey in bands:
        meta = SURVEY_DEFAULTS.get(survey, {})
        lam = meta.get("lambda_eff")
        psf = meta.get("psf_fwhm")
        if lam is None or psf is None:
            continue
        # represent PSF as a fractional horizontal bar centered on lam
        half = lam * (psf / 30.0)  # purely indicative width
        ax.plot([max(lam - half, lam_lo), min(lam + half, lam_hi)],
                [ybar, ybar], color=_family_color(survey), lw=2.5, alpha=0.7,
                solid_capstyle="butt", zorder=2)


def _comprehensive_table(ax, bands, comp_keys, photometry, deblend_comb,
                         unwise_forced):
    """Render the bottom flux table (rows=components, cols=surveys)."""
    headers = ["comp"] + [_SHORT_LABELS.get(s, s) for s in bands]
    rows = []
    cell_colors = []

    for comp in comp_keys:
        row = [_COMP_LABELS.get(comp, comp)]
        crow = ["#101010"]
        for survey in bands:
            phot = photometry.get(survey, {}) if isinstance(photometry, dict) else {}
            blended = bool(phot.get("blend_flag", False))
            if blended:
                cm = deblend_comb.get(survey, {}).get(comp, {})
                cm = cm if isinstance(cm, dict) else {}
                f = cm.get("flux_mjy", np.nan)
                e = cm.get("flux_err_mjy", np.nan)
            else:
                m = phot.get(comp, {})
                m = m if isinstance(m, dict) else {}
                f = m.get("flux_mjy", np.nan)
                e = m.get("flux_err_mjy", np.nan)
            txt = _fmt_flux_cell(_as_float(f), _as_float(e))
            # unWISE dagger
            uw = unwise_forced.get(survey, {}).get(comp, {}) if isinstance(unwise_forced.get(survey), dict) else {}
            if isinstance(uw, dict) and np.isfinite(_as_float(uw.get("flux_mjy", np.nan))):
                txt += "†"
            row.append(txt)
            crow.append("#3d2800" if blended else "#1a1a1a")
        rows.append(row)
        cell_colors.append(crow)

    if not rows:
        ax.text(0.5, 0.5, "no photometry", transform=ax.transAxes,
                ha="center", va="center", color="0.5", style="italic")
        return

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center", cellColours=cell_colors)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    comp_colors = [_COMP_COLORS.get(c, "white") for c in comp_keys]
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.get_text().set_fontfamily("monospace")
        if r == 0:
            cell.set_facecolor("#000000")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            if c == 0:
                cell.get_text().set_color(comp_colors[r - 1]
                                          if r - 1 < len(comp_colors) else "white")
                cell.get_text().set_weight("bold")
            else:
                cell.get_text().set_color("#e8e8e8")

    ax.text(0.5, 1.06, "flux density [mJy]  (comb for blended, direct otherwise; "
            "† = unWISE match)", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9, color="#aaaaaa")


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan
