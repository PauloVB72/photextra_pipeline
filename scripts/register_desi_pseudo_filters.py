"""Register DESI resolution-element pseudo-filters in the pcigale filter DB.

cigale2s fits a spectrum by treating each resolution element as a narrow
"pseudo-filter".  configuration.py picks them up by name prefix ``mode + '_'``
(here ``desi_``), so the observed spectrum resampled onto the 512-point grid is
matched, element by element, against model fluxes synthesised through these
pseudo-filters.

This mirrors ``pcigale_filters.add_spec`` exactly (same tophat construction,
same 1e20/(c*int) normalisation so model integration yields mJy), except:
  * the filters are named ``desi_Band_XXX`` (mode-namespaced) instead of the
    hard-coded ``prism_Band_XXX``, so we do NOT clobber any NIRSpec prism set;
  * the grid comes from our DESI dispersion file (build_desi_dispersion_file.py)
    via the same ``new_wavegrid`` used by read_prism at fit time -> guaranteed
    consistent pivots.
"""
import argparse

import numpy as np
import scipy.constants as cst

from pcigale.data import SimpleDatabase as Database
from pcigale_filters import new_wavegrid


def register(disp_file, mode="desi"):
    spec_wave, Dlambda, R = new_wavegrid(disp_file)   # 512-pt grid [Angstrom]
    n_digits = len(str(len(spec_wave)))
    print(f"grid: {len(spec_wave)} elements, "
          f"{spec_wave.min():.1f}-{spec_wave.max():.1f} A, R~{np.median(R):.1f}")

    db = Database("filters", writable=True)
    n = 0
    for i_wave, (lambda_i, Dlambda_i, R_i) in enumerate(zip(spec_wave, Dlambda, R)):
        filtname = f"{mode}_Band_{str(i_wave).zfill(n_digits)}"
        desc = f"# Pseudo-filter for DESI resolution element {i_wave}"
        # tophat spanning one resolution element (same shape as add_spec)
        wl = np.array([lambda_i - lambda_i / R_i / 2.0 - Dlambda_i / 1000.,
                       lambda_i - lambda_i / R_i / 3.0,
                       lambda_i + lambda_i / R_i / 3.0,
                       lambda_i + lambda_i / R_i / 2.0 + Dlambda_i / 1000.])
        tr = np.array([0.0, 1.0, 1.0, 0.0])
        wl *= 0.1  # Angstrom -> nm
        pivot = np.sqrt(np.trapz(tr, wl) / np.trapz(tr / wl ** 2, wl))
        # 1e20/(c*int(tr/wl^2)) so integrating W/m2/nm gives mJy directly
        tr *= 1e20 / (cst.c * np.trapz(tr / wl ** 2, wl))
        db.add({"name": filtname},
               {"wl": wl, "tr": tr, "pivot": pivot, "desc": desc})
        n += 1
    db.close()
    print(f"registered {n} '{mode}_Band_*' pseudo-filters")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("disp_file")
    ap.add_argument("--mode", default="desi")
    args = ap.parse_args()
    register(args.disp_file, args.mode)
