"""Generate paper figures for the photextra_pipeline software paper.

Reuses ONLY already-produced pipeline outputs (cached FITS cutouts,
per-target photometry CSVs, the MKW8 master table and the cached DESI
spectrum); nothing is re-downloaded and no production code is modified.
"""
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/polo/Escritorio/PHD/code/xdebpair")

HERE = os.path.dirname(os.path.abspath(__file__))
XDEB = "/home/polo/Escritorio/PHD/code/photextra_pipeline/test_output_xdeb"
MKW8 = "/home/polo/Escritorio/PHD/CHANCES/MKW8_full_run"

# --- paper style ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif",
})

# colourblind-safe (Paul Tol bright) fixed assignment
C_GAL1 = "#EE6677"   # component 1
C_GAL2 = "#4477AA"   # component 2
C_TOT = "#CCBB44"    # total
C_SPEC = "#888888"   # observed spectrum
C_NORM = "#4477AA"   # normalized spectrum
C_SYN = "#EE6677"    # synthetic photometry
C_IMG = "#CCBB44"    # imaging photometry
C_FIB = "#009988"    # fiber photometry

CASES = [("merger_D", "single"), ("merger_B", "companions"),
         ("merger_A", "merger")]


def _load_legacy(case):
    imgs = {}
    for b in ("Legacy_g", "Legacy_r", "Legacy_z"):
        with fits.open(f"{XDEB}/{case}/cache/{b}.fits") as h:
            imgs[b] = (h[0].data.astype(float), WCS(h[0].header))
    return imgs


def fig_deblend():
    """xdebpair source separation on the Legacy r band for three classes."""
    from xdebpair import XdebPair
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.55))
    for ax, (case, _) in zip(axes, CASES):
        imgs = _load_legacy(case)
        r, w = imgs["Legacy_r"]
        cube = np.array([imgs[b][0] for b in
                         ("Legacy_g", "Legacy_r", "Legacy_z")])
        res = XdebPair(use_nmf=False, verbose=False).fit(
            r, wcs=w, image_cube=cube)
        sep = res.separation_px * 0.262
        n = r.shape[0]
        single = res.classification == "single"
        nuc = np.atleast_2d(res.nuclei_yx)[: 1 if single else 2] \
            if res.nuclei_yx is not None else np.array([[n / 2, n / 2]])
        cy, cx = nuc.mean(axis=0)
        half = int(22.0 / 0.262 / 2)  # 22" wide zoom
        cy = int(np.clip(cy, half, n - half))
        cx = int(np.clip(cx, half, n - half))
        sly = slice(cy - half, cy + half)
        slx = slice(cx - half, cx + half)
        cut = r[sly, slx]
        norm = ImageNormalize(cut, interval=PercentileInterval(99.3),
                              stretch=AsinhStretch(0.02))
        ax.imshow(cut, origin="lower", cmap="Greys", norm=norm)
        colors = {"gal1": C_GAL1, "gal2": C_GAL2}
        for name, col in colors.items():
            if single and name == "gal2":
                continue
            m = res.masks.get(name)
            if m is None or not m.any():
                continue
            ax.contour(m[sly, slx].astype(float), levels=[0.5],
                       colors=[col], linewidths=1.1)
        for k, (y, x) in enumerate(nuc):
            ax.plot(x - (cx - half), y - (cy - half), "x",
                    color=[C_GAL1, C_GAL2][k], ms=5, mew=1.2)
        label = res.classification
        if sep > 0:
            label += f"  ($\\Delta\\theta = {sep:.1f}''$)"
        ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top",
                fontsize=8.5,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        # 5" scale bar
        bar = 5.0 / 0.262
        x0, y0 = cut.shape[1] * 0.70, cut.shape[0] * 0.06
        ax.plot([x0, x0 + bar], [y0, y0], color="k", lw=1.4)
        ax.text(x0 + bar / 2, y0 + 2.5, "5''", ha="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(wspace=0.03)
    fig.savefig(os.path.join(HERE, "fig_deblend.pdf"))
    plt.close(fig)
    print("fig_deblend done")


def fig_sed_comparison():
    """Per-component SEDs for the same three systems, from the pipeline
    photometry products (long-format CSV, one row per band)."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True,
                             sharex=True)
    for ax, (case, cls) in zip(axes, CASES):
        t = pd.read_csv(f"{XDEB}/{case}/products/{case}_photometry.csv")
        t = t.sort_values("lambda_eff_um")
        lam = t["lambda_eff_um"].values

        def _plot(col, ecol, color, label, marker, open_marker=False):
            f = t[col].values
            e = t[ecol].values
            good = np.isfinite(f) & (f > 0)
            ax.errorbar(lam[good], f[good], yerr=e[good], fmt=marker,
                        color=color, ms=4.5, lw=1.0, elinewidth=0.8,
                        capsize=1.5, label=label,
                        mfc="white" if open_marker else color, mew=1.0)
            ax.plot(lam[good], f[good], color=color, lw=0.9, alpha=0.45)

        _plot("total_flux_mjy", "total_flux_err_mjy", C_TOT,
              "total (system)", "o")
        if "gal2_flux_mjy" in t.columns and \
                np.isfinite(t["gal2_flux_mjy"]).any() and cls != "single":
            _plot("gal1_flux_mjy", "gal1_flux_err_mjy", C_GAL1,
                  "component 1", "s")
            _plot("gal2_flux_mjy", "gal2_flux_err_mjy", C_GAL2,
                  "component 2", "D")
        # blended bands marker
        bl = t["blend_flag"].astype(bool).values
        f = t["total_flux_mjy"].values
        gb = bl & np.isfinite(f) & (f > 0)
        ax.plot(lam[gb], f[gb], "o", ms=8, mfc="none", mec="k", mew=0.6,
                zorder=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel(r"$\lambda_{\rm eff}$ [$\mu$m]")
        ax.legend(loc="upper left", frameon=False, handlelength=1.2,
                  fontsize=7)
    axes[0].set_ylabel(r"$F_\nu$ [mJy]")
    axes[0].set_ylim(1e-5, 60)
    fig.subplots_adjust(wspace=0.04)
    fig.savefig(os.path.join(HERE, "fig_sed_comparison.pdf"))
    plt.close(fig)
    print("fig_sed_comparison done")


def fig_combined_example():
    """Photometry+spectroscopy combination for one MKW8 galaxy."""
    tid = "2842593834565632"
    d = np.load(f"{MKW8}/{tid}/spectroscopy/desi_{tid}.npz")
    wave, flux = d["wave"], d["flux"]  # AA, 1e-17 erg/s/cm2/AA
    tab = pd.read_csv(f"{MKW8}/MKW8_master_table.csv",
                      dtype={"target_id": str})
    row = tab[tab.target_id == tid].iloc[0]
    norm = row["norm_factor"]

    # f_lambda(1e-17 cgs) -> f_nu in mJy: fnu = flam * lam^2 / c
    c_aa = 2.99792458e18  # AA/s
    fnu_mjy = flux * 1e-17 * wave ** 2 / c_aa / 1e-26  # 1e-26 erg/s/cm2/Hz = mJy
    # light smoothing for display
    k = 15
    ker = np.ones(k) / k
    sm = np.convolve(fnu_mjy, ker, mode="same")

    lam_eff = {"GALEX_FUV": 0.153, "GALEX_NUV": 0.231, "Legacy_g": 0.472,
               "Legacy_r": 0.641, "Legacy_z": 0.906, "WISE_W1": 3.4,
               "WISE_W2": 4.6}
    synth_lam = {"Legacy_g": 0.472, "Legacy_r": 0.641, "Legacy_z": 0.906}

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    wum = wave / 1e4
    ax.plot(wum, sm, color=C_SPEC, lw=0.6,
            label="DESI spectrum (fibre)")
    ax.plot(wum, sm * norm, color=C_NORM, lw=0.6,
            label=rf"spectrum $\times\,{norm:.2f}$")

    for b, lam in lam_eff.items():
        f = row[f"phot_{b}_flux_mjy"]
        fe = row[f"phot_{b}_flux_err_mjy"]
        ff = row[f"phot_fiber_{b}_flux_mjy"]
        if np.isfinite(f) and f > 0:
            ax.errorbar(lam, f, yerr=fe, fmt="o", color=C_IMG, ms=5,
                        zorder=5)
        if np.isfinite(ff) and ff > 0:
            ax.errorbar(lam, ff, fmt="^", color=C_FIB, ms=4.5, zorder=5)
    for b, lam in synth_lam.items():
        fs = row[f"synth_{b}_flux_mjy"]
        fsc = row[f"scaled_{b}_flux_mjy"]
        if np.isfinite(fs):
            ax.plot(lam, fs, "s", mfc="none", mec=C_SYN, ms=6, mew=1.2,
                    zorder=6)
        if np.isfinite(fsc):
            ax.plot(lam, fsc, "D", color=C_SYN, ms=4.5, zorder=6)
    # proxy legend entries
    ax.plot([], [], "o", color=C_IMG, ms=5, label="imaging (total)")
    ax.plot([], [], "^", color=C_FIB, ms=4.5,
            label='imaging (fibre 1.5$^{\\prime\\prime}$)')
    ax.plot([], [], "s", mfc="none", mec=C_SYN, ms=6, mew=1.2,
            label="synthetic (pre-norm)")
    ax.plot([], [], "D", color=C_SYN, ms=4.5,
            label=r"synthetic $\times$ norm")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.13, 6.0)
    ax.set_ylim(3e-5, 3e2)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$F_\nu$ [mJy]")
    ax.legend(loc="lower right", frameon=False, fontsize=6.6,
              handlelength=1.2)
    fig.savefig(os.path.join(HERE, "fig_combined_example.pdf"))
    plt.close(fig)
    print("fig_combined_example done")


def fig_mkw8_stats():
    """MKW8 production-run statistics: aperture correction + WHAN classes."""
    tab = pd.read_csv(f"{MKW8}/MKW8_master_table.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    nf = tab["norm_factor"].values
    nf = nf[np.isfinite(nf) & (nf > 0)]
    bins = np.logspace(np.log10(0.1), np.log10(300), 36)
    ax1.hist(np.clip(nf, 0.1, 300), bins=bins, color=C_GAL2, alpha=0.85,
             edgecolor="white", lw=0.4)
    med = np.median(nf)
    ax1.axvline(med, color=C_GAL1, lw=1.2, ls="--")
    ax1.text(med * 1.25, ax1.get_ylim()[1] * 0.92,
             f"median = {med:.1f}", color=C_GAL1, fontsize=8, va="top")
    ax1.set_xscale("log")
    ax1.set_xlabel(r"aperture correction  $s(\lambda_0)$")
    ax1.set_ylabel("galaxies")

    order = ["SF", "sAGN", "wAGN", "retired", "passive", "unknown"]
    counts = tab["whan_class"].value_counts()
    vals = [counts.get(k, 0) for k in order]
    y = np.arange(len(order))
    ax2.barh(y, vals, color=C_GAL2, alpha=0.85, height=0.62)
    for yi, v in zip(y, vals):
        ax2.text(v + 3, yi, str(v), va="center", fontsize=8)
    ax2.set_yticks(y, order)
    ax2.invert_yaxis()
    ax2.set_xlabel("galaxies")
    ax2.set_xlim(0, max(vals) * 1.18)
    ax2.set_title("WHAN classification", fontsize=9)
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(os.path.join(HERE, "fig_mkw8_stats.pdf"))
    plt.close(fig)
    print("fig_mkw8_stats done")


if __name__ == "__main__":
    fig_deblend()
    fig_sed_comparison()
    fig_combined_example()
    fig_mkw8_stats()
