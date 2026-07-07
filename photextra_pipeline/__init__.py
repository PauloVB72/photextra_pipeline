"""photextra_pipeline: multiband aperture photometry for galaxies/mergers."""

import os

# Pin BLAS/OpenMP thread pools to 1 thread BEFORE numpy/scipy are imported
# anywhere in this package.  The pPXF/xpectrafit matrices are too small to
# benefit from multithreaded BLAS (measured slower unpinned).  These env vars
# are read once at BLAS thread-pool init, so they must be set before the
# first numpy import in the process (xpectrafit/__init__.py does the same,
# but numpy is imported by this package well before xpectrafit).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
del _v

from .pipeline import Pipeline
from .downloader import SURVEY_DEFAULTS, hostphot_downloader
from .deblending import XdebPairAdapter, XdebPairResult

__version__ = "0.1.0"

__all__ = ["Pipeline", "SURVEY_DEFAULTS", "hostphot_downloader",
           "XdebPairAdapter", "XdebPairResult"]
