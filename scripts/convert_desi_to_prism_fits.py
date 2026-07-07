"""Convert a cached DESI .npz spectrum into the FITS structure that
cigale2s ``read_prism()`` expects.

read_prism opens the file and reads ``hdu_spec[1].data`` with fields:
    WAVELENGTH  [micron]   (multiplied by 1e4 -> Angstrom, observed frame)
    FLUX        [mJy]      (F_nu, compared directly to photometric mJy)
    FLUX_ERROR  [mJy]

The cached DESI npz stores (from photextra_pipeline.spectrum_acquisition via
SPARCL / DESI DR1 coadd):
    wave  [Angstrom, observed frame]
    flux  [1e-17 erg/s/cm2/A, F_lambda]
    ivar  [1 / (1e-17 erg/s/cm2/A)^2]
    z     [redshift]

We convert F_lambda -> F_nu in mJy:
    F_nu[mJy] = F_lambda[erg/s/cm2/A] * wave[A]^2 / c[A/s] * 1e26
with F_lambda = flux * 1e-17.  The per-pixel error is 1/sqrt(ivar) in the same
F_lambda units, converted with the identical per-pixel factor so the
signal-to-noise per pixel is preserved exactly.

Note: read_prism internally renormalises the spectrum onto the broadband
photometry (normalize_spec_to_phot), so the absolute flux scale is not critical;
what matters -- and what this conversion gets right -- is the F_lambda->F_nu
(x lambda^2) *shape* change and a consistent flux/error unit.
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

C_AA_S = 2.99792458e18   # speed of light in Angstrom / s


def convert(npz_path, out_path):
    d = np.load(npz_path)
    wave = np.asarray(d["wave"], float)     # Angstrom
    flux = np.asarray(d["flux"], float)     # 1e-17 erg/s/cm2/A
    ivar = np.asarray(d["ivar"], float)

    order = np.argsort(wave)
    wave, flux, ivar = wave[order], flux[order], ivar[order]

    # F_lambda in cgs
    flam = flux * 1e-17                                  # erg/s/cm2/A
    with np.errstate(divide="ignore"):
        flam_err = np.where(ivar > 0, 1.0 / np.sqrt(ivar) * 1e-17, np.nan)

    # F_lambda -> F_nu[mJy]
    conv = wave ** 2 / C_AA_S * 1e26                     # (erg/s/cm2/A) -> mJy
    fnu = flam * conv                                    # mJy
    fnu_err = flam_err * conv                            # mJy

    good = np.isfinite(fnu) & np.isfinite(fnu_err) & (fnu_err > 0)
    wave, fnu, fnu_err = wave[good], fnu[good], fnu_err[good]

    col_wave = fits.Column(name="WAVELENGTH", format="D", array=wave / 1e4)  # micron
    col_flux = fits.Column(name="FLUX", format="D", array=fnu)              # mJy
    col_ferr = fits.Column(name="FLUX_ERROR", format="D", array=fnu_err)    # mJy
    hdu = fits.BinTableHDU.from_columns([col_wave, col_flux, col_ferr])
    hdu.header["TUNIT1"] = "micron"
    hdu.header["TUNIT2"] = "mJy"
    hdu.header["TUNIT3"] = "mJy"
    hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
    out_path = Path(out_path)
    hdulist.writeto(out_path, overwrite=True)
    print(f"wrote {out_path}: {wave.size} pix, {wave.min():.1f}-{wave.max():.1f} A, "
          f"F_nu median={np.median(fnu):.4g} mJy, median SNR={np.median(fnu/fnu_err):.1f}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("out")
    args = ap.parse_args()
    convert(args.npz, args.out)
