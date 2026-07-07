#!/usr/bin/env python3
"""Exploracion multi-target de aperture correction (fibra -> total).

Extiende explore_aperture_correction.py a los 9 targets reales disponibles
(MKW8 x8 + sanity_normfix x1). NO forma parte del paquete photextra_pipeline.

Para cada target calcula:
  a. grey r (baseline pipeline, Zou+2024)
  b. grey g / grey z
  c. power law lambda-dependiente fibra->total (solo Legacy g/r/z, anclada
     en fibra/synth r) -> exponente y factor evaluado en r
  d. grey multi-banda ponderado por S/N^2 (g,r,z)
mas fAGN, bpt_class/spectral_class y Dn4000 para correlacionar.

Salidas (en explore_aperture_multi/ junto a este script):
  - aperture_correction_multi_summary.csv
  - aperture_correction_multi_summary.png
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "explore_aperture_multi")

MKW8_DIR = "/home/polo/Escritorio/PHD/CHANCES/test_output_MKW8_8parallel_both_v2"
CSVS = sorted(glob.glob(os.path.join(MKW8_DIR, "*/products/*_combined.csv")))
CSVS.append("/home/polo/Escritorio/PHD/CHANCES/test_output_sanity_normfix/"
            "39627860855490651/products/39627860855490651_combined.csv")

LAMBDA = {"Legacy_g": 0.472, "Legacy_r": 0.641, "Legacy_z": 0.906}
LEGACY_BANDS = ["Legacy_g", "Legacy_r", "Legacy_z"]


def get(row, col):
    v = row.get(col, np.nan)
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def analyze(csv_path):
    row = pd.read_csv(csv_path).iloc[0]
    r = {"target_id": str(row["target_id"]),
         "z": get(row, "z"),
         "fAGN": get(row, "fAGN"),
         "spectral_class": row.get("spectral_class", ""),
         "bpt_class": row.get("bpt_class", ""),
         "Dn4000": get(row, "Dn4000"),
         "norm_factor_pipeline": get(row, "norm_factor")}

    phot, err, fib, synth = {}, {}, {}, {}
    for b in LEGACY_BANDS:
        phot[b] = get(row, f"phot_{b}_flux_mjy")
        err[b] = get(row, f"phot_{b}_flux_err_mjy")
        fib[b] = get(row, f"phot_fiber_{b}_flux_mjy")
        synth[b] = get(row, f"synth_{b}_flux_mjy")

    # grey factors por banda: phot_total / synth
    grey = {b: phot[b] / synth[b] for b in LEGACY_BANDS
            if np.isfinite(phot[b]) and np.isfinite(synth[b]) and synth[b] > 0}
    r["n_bandas_grey"] = len(grey)
    r["factor_a"] = grey.get("Legacy_r", np.nan)
    r["factor_b_g"] = grey.get("Legacy_g", np.nan)
    r["factor_b_z"] = grey.get("Legacy_z", np.nan)
    if len(grey) >= 2:
        v = list(grey.values())
        r["disp_grey_pct"] = 100 * (max(v) - min(v)) / np.mean(v)
    else:
        r["disp_grey_pct"] = np.nan

    # (d) grey ponderado S/N^2
    facs, wts = [], []
    for b, f in grey.items():
        if np.isfinite(err[b]) and err[b] > 0 and np.isfinite(phot[b]):
            facs.append(f)
            wts.append((phot[b] / err[b]) ** 2)
    r["factor_d"] = float(np.average(facs, weights=wts)) if facs else np.nan

    # (c) power law sobre cocientes total/fibra Legacy, anclada fibra/synth r
    ratio = {b: phot[b] / fib[b] for b in LEGACY_BANDS
             if np.isfinite(phot[b]) and np.isfinite(fib[b]) and fib[b] > 0}
    anchor = (fib["Legacy_r"] / synth["Legacy_r"]
              if np.isfinite(fib.get("Legacy_r", np.nan))
              and np.isfinite(synth.get("Legacy_r", np.nan))
              and synth["Legacy_r"] > 0 else np.nan)
    r["n_bandas_c"] = len(ratio)
    if len(ratio) >= 3 and np.isfinite(anchor) and all(v > 0 for v in ratio.values()):
        lam = np.array([LAMBDA[b] for b in ratio])
        rat = np.array(list(ratio.values()))
        slope, ic = np.polyfit(np.log10(lam), np.log10(rat), 1)
        r["exponente_c"] = slope
        r["factor_c_r"] = anchor * 10 ** (ic + slope * np.log10(LAMBDA["Legacy_r"]))
        r["factor_c_u"] = anchor * 10 ** (ic + slope * np.log10(0.3551))
        r["factor_c_z"] = anchor * 10 ** (ic + slope * np.log10(0.906))
    else:
        r["exponente_c"] = np.nan
        r["factor_c_r"] = np.nan
        r["factor_c_u"] = np.nan
        r["factor_c_z"] = np.nan
    return r


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = [analyze(p) for p in CSVS]
    df = pd.DataFrame(rows)

    # desviacion de c y d respecto a a (%)
    df["d_vs_a_pct"] = 100 * (df["factor_d"] / df["factor_a"] - 1)
    df["c_vs_a_pct"] = 100 * (df["factor_c_r"] / df["factor_a"] - 1)
    df["rango_c_pct"] = 100 * (df["factor_c_u"] - df["factor_c_z"]) / df["factor_a"]

    df = df.sort_values("fAGN", ascending=False).reset_index(drop=True)

    csv_out = os.path.join(OUTDIR, "aperture_correction_multi_summary.csv")
    df.to_csv(csv_out, index=False)

    cols = ["target_id", "fAGN", "bpt_class", "Dn4000", "factor_a", "factor_b_g",
            "factor_b_z", "factor_d", "disp_grey_pct", "exponente_c",
            "factor_c_r", "d_vs_a_pct", "c_vs_a_pct"]
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.float_format", "{:.3f}".format):
        print(df[cols].to_string(index=False))
    print(f"\nTabla guardada en: {csv_out}")

    # ------------------------------------------------------------- plot
    plt.style.use("dark_background")
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 7.5), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    fig.patch.set_facecolor("#1a1a19")
    x = np.arange(len(df))
    labels = [f"{t[-6:]}\nfAGN={f:.2f}\nDn={d:.2f}" for t, f, d in
              zip(df["target_id"], df["fAGN"], df["Dn4000"])]

    C_A, C_D, C_C = "#3987e5", "#e66767", "#9085e9"
    for axx in (ax, ax2):
        axx.set_facecolor("#1a1a19")
        axx.grid(True, lw=0.5, color="#2c2c2a")
        axx.tick_params(colors="#898781")
        for s in axx.spines.values():
            s.set_color("#383835")

    # panel superior: factores absolutos + rango de c (u..z)
    ax.vlines(x, df["factor_c_u"], df["factor_c_z"], color=C_C, lw=6,
              alpha=0.25, zorder=2, label="c) rango power law (u..z)")
    ax.plot(x, df["factor_a"], "o-", ms=9, lw=1.4, color=C_A,
            mec="#1a1a19", mew=0.8, zorder=5, label="a) grey r (baseline)")
    ax.plot(x, df["factor_d"], "X-", ms=9, lw=1.4, color=C_D,
            mec="#1a1a19", mew=0.8, zorder=6, label="d) grey pond. S/N$^2$")
    ax.plot(x, df["factor_c_r"], "P", ms=9, color=C_C,
            mec="#1a1a19", mew=0.8, zorder=7, label="c) power law en r")
    ax.set_ylabel("factor de correccion", color="#c3c2b7")
    ax.set_title("Aperture correction fibra$\\to$total: metodos a/c/d en 9 targets "
                 "(ordenados por fAGN)", color="#ffffff", fontsize=11)
    leg = ax.legend(loc="upper right", fontsize=8, framealpha=0.2)
    for t in leg.get_texts():
        t.set_color("#c3c2b7")

    # panel inferior: desviaciones relativas vs baseline + dispersion grey
    ax2.axhline(0, color="#898781", lw=0.8)
    ax2.plot(x, df["d_vs_a_pct"], "X-", ms=8, lw=1.2, color=C_D,
             mec="#1a1a19", mew=0.8, label="d vs a [%]")
    ax2.plot(x, df["c_vs_a_pct"], "P-", ms=8, lw=1.2, color=C_C,
             mec="#1a1a19", mew=0.8, label="c(r) vs a [%]")
    ax2.plot(x, df["disp_grey_pct"], "s--", ms=6, lw=1.0, color="#c3c2b7",
             alpha=0.8, label="dispersion grey g/r/z [%]")
    ax2.set_ylabel("desviacion [%]", color="#c3c2b7")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7, color="#c3c2b7")
    leg2 = ax2.legend(loc="upper right", fontsize=8, framealpha=0.2)
    for t in leg2.get_texts():
        t.set_color("#c3c2b7")

    png_out = os.path.join(OUTDIR, "aperture_correction_multi_summary.png")
    fig.savefig(png_out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Plot guardado en:  {png_out}")


if __name__ == "__main__":
    main()
