#!/usr/bin/env python3
"""Exploracion standalone de alternativas de aperture correction (fibra -> total).

NO forma parte del paquete photextra_pipeline. Compara, sobre un target ya
procesado, varios metodos de normalizacion de la fotometria sintetica del
espectro:

  a. baseline grey (banda r, apertura total)     -- metodo actual (Zou+2024)
  b. grey con banda g / banda z                  -- test de "greyness"
  c. curva lambda-dependiente fibra-vs-total     -- power law sobre Legacy g/r/z
     (anclada al espectro via phot_fiber_r/synth_r)
  c'. igual que c pero con TODAS las bandas      -- diagnostico (PSF-dominado)
  d. grey multi-banda ponderado por S/N          -- promedio de phot/synth en g,r,z

Uso:
    python explore_aperture_correction.py [ruta_csv] [ruta_png]
"""

import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_DEFAULT = ("/home/polo/Escritorio/PHD/CHANCES/test_output_sanity_normfix/"
               "39627860855490651/products/39627860855490651_combined.csv")
PNG_DEFAULT = ("/home/polo/Escritorio/PHD/CHANCES/test_output_sanity_normfix/"
               "39627860855490651/products/explore_aperture_correction.png")

# ---------------------------------------------------------------- lambda_eff
try:
    from photextra_pipeline.downloader import SURVEY_DEFAULTS
    LAMBDA_PHOT = {k: v["lambda_eff"] for k, v in SURVEY_DEFAULTS.items()}
except Exception:  # fallback: mismos valores que el paquete (micron)
    LAMBDA_PHOT = {
        "GALEX_FUV": 0.153, "GALEX_NUV": 0.231,
        "Legacy_g": 0.472, "Legacy_r": 0.641, "Legacy_z": 0.906,
        "WISE_W1": 3.4, "WISE_W2": 4.6, "WISE_W3": 12.0, "WISE_W4": 22.0,
    }

try:
    from photextra_pipeline.combined_plots import _EXTRA_LAMBDA_EFF
except Exception:
    _EXTRA_LAMBDA_EFF = {
        "SDSS_u": 0.3551, "SDSS_g": 0.4686, "SDSS_r": 0.6166,
        "SDSS_i": 0.7480, "SDSS_z": 0.8932,
    }

LAMBDA_ALL = {**LAMBDA_PHOT, **_EXTRA_LAMBDA_EFF}

PHOT_BANDS = ["GALEX_FUV", "GALEX_NUV", "Legacy_g", "Legacy_r", "Legacy_z",
              "WISE_W1", "WISE_W2"]
SYNTH_BANDS = ["SDSS_u", "SDSS_g", "Legacy_g", "SDSS_r", "Legacy_r",
               "SDSS_i", "SDSS_z", "Legacy_z"]
LEGACY_BANDS = ["Legacy_g", "Legacy_r", "Legacy_z"]


def get(row, col):
    v = row.get(col, np.nan)
    return float(v) if pd.notna(v) else np.nan


def main(csv_path=CSV_DEFAULT, png_path=PNG_DEFAULT):
    row = pd.read_csv(csv_path).iloc[0]

    phot = {b: get(row, f"phot_{b}_flux_mjy") for b in PHOT_BANDS}
    phot_err = {b: get(row, f"phot_{b}_flux_err_mjy") for b in PHOT_BANDS}
    fib = {b: get(row, f"phot_fiber_{b}_flux_mjy") for b in PHOT_BANDS}
    synth = {b: get(row, f"synth_{b}_flux_mjy") for b in SYNTH_BANDS}
    synth = {b: v for b, v in synth.items() if np.isfinite(v)}

    print(f"Target {row['target_id']}  z={row['z']:.4f}  "
          f"fuente={row['spectrum_source']}  fibra={row['fiber_diam_arcsec']}\"")
    print(f"norm_factor pipeline = {row['norm_factor']:.4f} "
          f"(norm_band={row['norm_band']}, aperture={row['norm_aperture']})\n")

    # ---------------- a/b: factores grey por banda (total/synth) ----------
    grey = {}
    for b in LEGACY_BANDS:
        if np.isfinite(phot[b]) and b in synth:
            grey[b] = phot[b] / synth[b]
    print("--- Factores grey phot_total/synth por banda ---")
    for b, f in grey.items():
        print(f"  {b:9s}: {f:.4f}")
    spread = (max(grey.values()) - min(grey.values())) / np.mean(list(grey.values()))
    print(f"  dispersion relativa g/r/z: {100 * spread:.1f}%\n")

    f_a = grey["Legacy_r"]

    # ---------------- d: grey multi-banda ponderado por S/N ---------------
    facs, wts = [], []
    for b in LEGACY_BANDS:
        if b in grey and np.isfinite(phot_err[b]) and phot_err[b] > 0:
            snr = phot[b] / phot_err[b]
            facs.append(grey[b])
            wts.append(snr**2)
    f_d = float(np.average(facs, weights=wts))
    print(f"--- (d) grey multi-banda ponderado S/N^2: {f_d:.4f} ---\n")

    # ---------------- c: curva lambda-dependiente fibra->total ------------
    # ratio de apertura por banda (imagen): total/fibra
    ratio = {b: phot[b] / fib[b] for b in PHOT_BANDS
             if np.isfinite(phot[b]) and np.isfinite(fib[b]) and fib[b] > 0}
    print("--- Cocientes de apertura phot_total/phot_fiber por banda ---")
    for b in PHOT_BANDS:
        if b in ratio:
            print(f"  {b:9s} (lam={LAMBDA_PHOT[b]:6.3f} um): {ratio[b]:8.2f}")
    print("  (GALEX/WISE: PSF 4.5-6.4\" >> fibra 1.5\" -> ratio dominado por "
          "PSF, no por gradiente fisico)\n")

    def powerlaw_fit(bands):
        lam = np.array([LAMBDA_PHOT[b] for b in bands])
        rat = np.array([ratio[b] for b in bands])
        slope, intercept = np.polyfit(np.log10(lam), np.log10(rat), 1)
        return slope, intercept

    def curve(lam_um, slope, intercept):
        return 10 ** (intercept + slope * np.log10(lam_um))

    # anclaje espectro<->fibra en banda r (comparables: misma apertura)
    anchor = fib["Legacy_r"] / synth["Legacy_r"]  # = norm_factor_fiber
    print(f"anclaje fibra/synth en r: {anchor:.4f} "
          f"(pipeline norm_factor_fiber={row['norm_factor_fiber']:.4f})")

    sl_leg, ic_leg = powerlaw_fit(LEGACY_BANDS)
    sl_all, ic_all = powerlaw_fit(list(ratio.keys()))
    print(f"(c)  power law Legacy g/r/z : ratio ~ lam^{sl_leg:+.3f}")
    print(f"(c') power law TODAS bandas : ratio ~ lam^{sl_all:+.3f}  "
          "[diagnostico, PSF-contaminado]\n")

    def f_c(lam_um, slope, intercept):
        return anchor * curve(lam_um, slope, intercept)

    # ---------------- factores por banda sintetica -------------------------
    methods = {
        "a) grey r (baseline)": lambda lam: f_a,
        "b) grey g": lambda lam: grey["Legacy_g"],
        "b) grey z": lambda lam: grey["Legacy_z"],
        "c) power law fibra/total (Legacy)": lambda lam: f_c(lam, sl_leg, ic_leg),
        "d) grey multi-banda pond. S/N": lambda lam: f_d,
    }
    print("--- Factor aplicado a cada banda sintetica ---")
    hdr = f"{'banda':10s}" + "".join(f"{m.split(')')[0]+')':>10s}" for m in methods)
    print(hdr)
    scaled = {m: {} for m in methods}
    for b in SYNTH_BANDS:
        if b not in synth:
            continue
        lam = LAMBDA_ALL[b]
        line = f"{b:10s}"
        for m, fn in methods.items():
            f = fn(lam)
            scaled[m][b] = synth[b] * f
            line += f"{f:10.3f}"
        print(line)

    # ---------------- plot -------------------------------------------------
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    fig.patch.set_facecolor("#1a1a19")
    ax.set_facecolor("#1a1a19")

    # fotometria real
    lam_p = [LAMBDA_PHOT[b] for b in PHOT_BANDS]
    ax.errorbar(lam_p, [phot[b] for b in PHOT_BANDS],
                yerr=[phot_err[b] for b in PHOT_BANDS],
                fmt="o", ms=9, color="#ffffff", mec="#1a1a19", mew=0.8,
                label="fotometria total (phot)", zorder=6)
    ax.plot(lam_p, [fib[b] for b in PHOT_BANDS], "D", ms=7,
            color="#898781", mec="#1a1a19", mew=0.8,
            label='fotometria fibra 1.5" (phot_fiber)', zorder=5)

    lam_s = np.array([LAMBDA_ALL[b] for b in synth])
    order = np.argsort(lam_s)
    sy = np.array([synth[b] for b in synth])
    ax.plot(lam_s[order], sy[order], "s--", ms=5, lw=1,
            color="#c3c2b7", alpha=0.7, label="sintetica sin escalar (synth)",
            zorder=4)

    colors = {"a) grey r (baseline)": "#3987e5",
              "b) grey g": "#199e70",
              "b) grey z": "#c98500",
              "c) power law fibra/total (Legacy)": "#9085e9",
              "d) grey multi-banda pond. S/N": "#e66767"}
    markers = {"a) grey r (baseline)": "o",
               "b) grey g": "v",
               "b) grey z": "^",
               "c) power law fibra/total (Legacy)": "P",
               "d) grey multi-banda pond. S/N": "X"}
    for m in methods:
        vals = np.array([scaled[m][b] for b in synth])
        ax.plot(lam_s[order], vals[order], markers[m] + "-", ms=7, lw=1.6,
                color=colors[m], mec="#1a1a19", mew=0.6, label=m, zorder=7)

    # curva continua del metodo c en el rango sintetico
    lam_grid = np.geomspace(lam_s.min(), lam_s.max(), 100)
    ax.plot(lam_grid, f_c(lam_grid, sl_leg, ic_leg) * np.interp(
        lam_grid, lam_s[order], sy[order]),
        lw=1, color="#9085e9", alpha=0.4, zorder=3)

    # inset: factor de correccion vs lambda (donde se ve la diferencia real)
    axi = ax.inset_axes([0.045, 0.63, 0.30, 0.31])
    axi.set_facecolor("#1a1a19")
    axi.set_xticks([0.4, 0.6, 0.8])
    for m in ["a) grey r (baseline)", "b) grey g", "b) grey z",
              "d) grey multi-banda pond. S/N"]:
        axi.axhline(methods[m](0.6), color=colors[m], lw=1.4)
    axi.plot(lam_grid, f_c(lam_grid, sl_leg, ic_leg),
             color=colors["c) power law fibra/total (Legacy)"], lw=1.6)
    for b in LEGACY_BANDS:  # factores grey medidos por banda
        axi.plot(LAMBDA_PHOT[b], grey[b], "o", ms=5, color="#ffffff",
                 mec="#1a1a19", mew=0.5, zorder=5)
    axi.set_title("factor de escala vs $\\lambda$ [$\\mu$m]", fontsize=8,
                  color="#c3c2b7")
    axi.set_xlim(lam_grid.min() * 0.95, lam_grid.max() * 1.05)
    axi.tick_params(colors="#898781", labelsize=7)
    axi.grid(True, lw=0.4, color="#2c2c2a")
    for s in axi.spines.values():
        s.set_color("#383835")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_{\rm eff}$ [$\mu$m]", color="#c3c2b7")
    ax.set_ylabel("flujo [mJy]", color="#c3c2b7")
    ax.set_title(f"Aperture correction alternatives - target {row['target_id']}"
                 f"  (norm_factor pipeline = {row['norm_factor']:.3f})",
                 color="#ffffff", fontsize=11)
    ax.tick_params(colors="#898781")
    ax.grid(True, which="major", lw=0.5, color="#2c2c2a")
    for s in ax.spines.values():
        s.set_color("#383835")
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.2)
    for t in leg.get_texts():
        t.set_color("#c3c2b7")
    fig.tight_layout()
    fig.savefig(png_path, facecolor=fig.get_facecolor())
    print(f"\nPlot guardado en: {png_path}")

    # ---------------- resumen numerico -------------------------------------
    print("\n--- RESUMEN ---")
    print(f"a) grey r (baseline)        : {f_a:.4f}")
    print(f"b) grey g                   : {grey['Legacy_g']:.4f}  "
          f"({100 * (grey['Legacy_g'] / f_a - 1):+.1f}% vs r)")
    print(f"b) grey z                   : {grey['Legacy_z']:.4f}  "
          f"({100 * (grey['Legacy_z'] / f_a - 1):+.1f}% vs r)")
    print(f"c) power law (en r)         : {f_c(LAMBDA_PHOT['Legacy_r'], sl_leg, ic_leg):.4f}"
          f"   pendiente lam^{sl_leg:+.3f}")
    print(f"   rango factor c (u..z)    : {f_c(0.3551, sl_leg, ic_leg):.3f} .. "
          f"{f_c(0.9060, sl_leg, ic_leg):.3f}")
    print(f"d) grey multi-banda S/N     : {f_d:.4f}  "
          f"({100 * (f_d / f_a - 1):+.1f}% vs r)")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(args if args else []))
