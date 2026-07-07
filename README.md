# photextra_pipeline

A multiband photometry and spectroscopy pipeline for merger and non-merger galaxies. It downloads image cutouts (GALEX/Legacy/WISE), separates individual components in interacting systems (via xdebpair), measures photometry for each component or for the entire system, fits DESI/SDSS spectra (stellar continuum + emission lines using XpectraFit), and combines photometric and spectroscopic data into a single SED for each target using either CIGALE (native spectroscopic fitting) or our proprietary aperture normalization method.

```text
photextra
```

Runs the pipeline with a welcome banner, execution summary, and a terminal
progress bar.

## Installation

```bash
git clone <this repository>
cd photextra_pipeline
pip install -e .
```

Required dependencies:

`numpy`, `scipy`, `astropy`, `matplotlib`, `pyyaml`, `sep`,
`reproject`, `astroquery`, `photutils`, `rich`.

Additionally, two companion packages (not available on PyPI) must be cloned
separately and installed (or made available through `sys.path`):

- [`xdebpair`](../xdebpair) — source separation for interacting galaxy pairs.
- [`xpectrafit`](../xpectrafit) — spectral fitting (pPXF + E-MILES, AGN,
  emission lines, BPT/WHAN classification).

For `both_method: cigale` (SED fitting with CIGALE using the full spectrum,
rather than broadband photometry only), a dedicated `cigale` environment with
the `cigale_spec` fork is also required. See
[`docs/cigale_tutorial/TUTORIAL.pdf`](docs/cigale_tutorial/TUTORIAL.pdf) for
complete installation instructions, registration of custom filters (CHANCES
continuum + real DECam/Legacy filters), and integration with this pipeline.

## Quick Start (CLI)

```bash
photextra config_xmask.yaml --targets my_targets.csv --limit 5
```

`my_targets.csv` must contain the columns:

```
ID,RA,DEC,REDSHIFT
```

An optional `type` column may also be included (`merger`,
`pre_merger`, or `post_merger`)—see the `separation` section below.

## Using as a Python Library

```python
from photextra_pipeline import Pipeline

pipe = Pipeline(config="config_xmask.yaml", mode="both")
pipe.run({"id": "12345", "ra": 180.1, "dec": 12.3, "z": 0.03})
```

## Configuration (`config_xmask.yaml`)

```yaml
mode: photometry            # photometry | spectroscopy | both
both_method: own            # own | cigale (only applies when mode: both)

photometry:
  aperture_mode: mask       # mask (xdebpair) | aperture (circular) | sep_apertures (SEP ellipses)
  separation: pair          # central | total | pair
  aperture_radius_arcsec: 5.0

spectroscopy:
  fit_agn: true
  spectrum_fwhm: 2.5
  survey_filters: [Legacy, SDSS]

use_xdebpair: true
surveys: [GALEX_FUV, GALEX_NUV, Legacy_g, Legacy_r, Legacy_z, WISE_W1, WISE_W2, WISE_W3, WISE_W4]
download_size: 1
common_grid: {reference: WISE_W4, pixscale: 1.375, size: 45}
output_dir: ./test_output
```

Explicit keyword arguments passed when creating `Pipeline(...)` (such as
`mode=`, `use_xdebpair=`, or `both_method=`) override the values defined in
the YAML configuration. This ensures that existing scripts (e.g.,
`run_mkw8_full.py`) continue to work without modification.

### `photometry.aperture_mode`

- **`mask`** (default): component separation using `xdebpair`, with automatic
  fallback to `xmask` if available, and finally to an internal single-component
  stub if neither package is installed.

- **`aperture`**: simple circular aperture photometry.

- **`sep_apertures`**: Kron elliptical apertures derived from source detection
  with `SEP`. This mode does not require `xdebpair` and provides a fast
  alternative without segmentation masks.

### `photometry.separation`

This option only applies when `aperture_mode: mask` is selected.

`xdebpair` is always executed first. Afterwards, the final behavior depends on
the selected mode:

- **`pair`**: measures each component separately. If the target catalog
  contains a `type` column, it controls the final decision:
  - `merger` or `pre_merger` → keep separate components.
  - `post_merger` (or any other value) → force a single component, even if
    `xdebpair` detected two.
  - If no `type` column exists, the raw `xdebpair` classification is used.

- **`central`**: always measures only the primary galaxy. If a companion is
  detected, its flux is ignored, while the output includes the flag
  `has_companion_not_measured=True` together with a warning in the execution
  log.

- **`total`**: merges all detected components into a single aperture,
  measuring the total flux of the system without decomposition.

### `both_method`

- **`own`** (default): proprietary aperture normalization method that scales
  fiber spectroscopy to total photometry using an S/N²-weighted combination of
  Legacy bands, extending the methodology of Zou et al. (2024).

- **`cigale`**: runs CIGALE using the complete observed spectrum
  (`use_spectro=True`) instead of the internal normalization method. See the
  CIGALE tutorial for installation details.

## Emission-Line Measurements (Spectroscopy)

`spectral_fit.py` extracts the following parameters for each emission line:

- `flux`
- `flux_err`
- `ew` (equivalent width)
- `ew_err`
- `sigma` (velocity dispersion, km s⁻¹)
- `v_kms` (velocity shift)
- `snr`

These quantities are measured for all 22 narrow emission lines and the 6 broad
AGN lines supported by XpectraFit. Previous versions extracted only fluxes for
four emission lines.

For very quiescent spectra (EW(Hα) < 3 Å), XpectraFit intentionally skips
emission-line fitting to reduce unnecessary computation. This behavior is a
built-in optimization rather than a software bug.

## Repository Structure

```text
photextra_pipeline/       # main package
  pipeline.py              # pipeline orchestration (Pipeline.run)
  downloader.py            # cutout download (GALEX/Legacy/WISE)
  deblending.py            # xdebpair interface
  deblend_photometry.py    # component photometry / xmask fallback
  photometry.py            # flux measurements (mask/aperture/SEP)
  convolution.py           # PSF matching across surveys
  reprojection.py          # common WCS grid + reprojection
  spectrum_acquisition.py  # DESI/SDSS spectrum download and caching
  spectral_fit.py          # spectral fitting (XpectraFit) + emission lines
  spec_normalization.py    # fiber-to-total flux normalization
  cigale_integration.py    # interface to CIGALE (both_method: cigale)
  output_table.py          # final output catalog
  validation_plots.py      # diagnostic plots
  cli.py                   # `photextra` command-line interface

scripts/                   # CIGALE utilities (filters, DESI→FITS conversion)
docs/cigale_tutorial/      # CIGALE installation, filters, tutorial PDF
test_*.py                  # pytest test suite
```

## Tests

```bash
python -m pytest test_new_features.py -q
```

The test suite currently contains **23 tests**, covering:

- all three photometry `aperture_mode` options;
- every combination of `separation` mode and galaxy classification;
- mask-related regression tests;
- WCS alignment across different surveys;
- emission-line extraction using cached real spectra;
- smoke tests for `both_method: cigale`.

All tests currently pass.
