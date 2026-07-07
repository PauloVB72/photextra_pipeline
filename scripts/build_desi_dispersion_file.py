"""Build a CIGALE (cigale2s) spectral-resolution / dispersion file for DESI.

The cigale2s spectro engine reads an instrument dispersion file via
``pcigale/utils/read_prism.py::new_wavegrid(file_path)``.  That function wants an
ASCII/FITS table with three columns:

    WAVELENGTH  [micron]
    Dlambda     [micron / pixel]
    R           [lambda / Dlambda, resolving power]

and it builds a **fixed 512-point** wavelength grid, starting at the minimum
tabulated wavelength and stepping by ``lambda / (2.2 * R(lambda))`` per point.
The SAME 512-point grid is used both to register the pseudo-filters
(``pcigale-filters spec``) and to resample the observed spectrum at fit time, so
the two are automatically consistent as long as they read the same file.

DATA-DRIVEN part (as requested): the wavelength grid limits are taken directly
from a real cached DESI spectrum's own wavelength array, and the *native* DESI
per-pixel dispersion ``Dlambda_native[i] = wave[i+1]-wave[i]`` and native
``R_native = wave / Dlambda_native`` are computed from that array and reported.

IMPORTANT CONSTRAINT / ASSUMPTION (flagged):
``new_wavegrid`` hard-codes ``N_pix = 512``.  DESI's native pixel sampling is
~0.8 A/pix (R ~ 4500-12000), so a 512-point grid stepping at Dlambda_native/2.2
would span only ~185 A -- useless for a full-optical fit.  To cover the FULL
observed DESI range (~3600-9824 A) in the available 512 resolution elements we
therefore write a constant EFFECTIVE resolving power

    R_eff = (N_pix - 1) / (2.2 * ln(wave_max / wave_min))

into the dispersion file.  This makes ``new_wavegrid`` lay 512 log-uniform
resolution elements across the entire DESI range (R_eff ~ 230).  That is a
deliberate rebinning of the full spectrum onto 512 bins -- still ~50x denser
than the 10 custom continuum filters it replaces, and it captures the full
continuum shape + strong features, which is the diagnostic point of this run.
The native DESI R is preserved in a comment for the record.
"""
import argparse
from pathlib import Path

import numpy as np

N_PIX = 512          # hard-coded in read_prism.new_wavegrid
OVERSAMPLE = 2.2     # resolution-element = 2.2 pixels convention in new_wavegrid


def build(npz_path, out_path):
    d = np.load(npz_path)
    wave = np.asarray(d["wave"], float)          # [Angstrom], observed frame
    wave = np.sort(wave)
    wmin, wmax = float(wave.min()), float(wave.max())

    # Native DESI per-pixel dispersion (data-driven, for the record)
    dlam_native = np.diff(wave)                   # [A/pix]
    r_native = wave[:-1] / dlam_native
    print(f"DESI native grid: {wave.size} pix, {wmin:.1f}-{wmax:.1f} A, "
          f"median Dlambda={np.median(dlam_native):.3f} A/pix, "
          f"native R={np.median(r_native):.0f} (min {r_native.min():.0f}, "
          f"max {r_native.max():.0f})")

    # Effective R so that the fixed 512-pt grid spans the FULL DESI range.
    r_eff = (N_PIX - 1) / (OVERSAMPLE * np.log(wmax / wmin))
    print(f"Effective R for full-range coverage in {N_PIX} elements: "
          f"R_eff={r_eff:.2f}")

    # Verify the resulting grid span with new_wavegrid's exact recurrence.
    w = np.zeros(N_PIX)
    w[0] = wmin
    for i in range(1, N_PIX):
        w[i] = w[i - 1] + w[i - 1] / (OVERSAMPLE * r_eff)
    print(f"Resulting 512-pt grid span: {w[0]:.1f}-{w[-1]:.1f} A "
          f"(DESI max {wmax:.1f})")

    # Write the dispersion table (micron units, constant R_eff).
    lam_um = np.linspace(wmin, wmax, 64) / 1e4          # micron
    dlam_um = lam_um / r_eff                            # micron/pix
    rcol = np.full_like(lam_um, r_eff)

    out_path = Path(out_path)
    with open(out_path, "w") as f:
        f.write("# DESI dispersion file for cigale2s new_wavegrid()\n")
        f.write(f"# built from {Path(npz_path).name}; native median R="
                f"{np.median(r_native):.0f}; R_eff={r_eff:.3f} "
                f"(512-element full-range rebinning)\n")
        f.write("WAVELENGTH Dlambda R\n")
        for lam, dl, r in zip(lam_um, dlam_um, rcol):
            f.write(f"{lam:.8f} {dl:.8e} {r:.6f}\n")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="a cached DESI .npz spectrum")
    ap.add_argument("out", help="output dispersion ASCII file")
    args = ap.parse_args()
    build(args.npz, args.out)
