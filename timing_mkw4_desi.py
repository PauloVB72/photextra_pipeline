"""Timing script: 10 MKW4 members, DESI spectra (via SPARCL), photometry+spectroscopy.

Read-only w.r.t. the pipeline source: only instruments existing functions
(monkeypatch for photometric download timing, direct calls for spectroscopy).
Produces plots for the photometry (mask) path via Pipeline._make_plots and
for the spectroscopy path via XpectraResult.plot().
"""
import csv
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from photextra_pipeline import pipeline as pl_mod
from photextra_pipeline.pipeline import Pipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

CSV_PATH = "/home/polo/Escritorio/PHD/CHANCES/MKW4_cluster_members.csv"
OUT_DIR = "test_output_MKW4_timing"
N_TARGETS = 10

with open(CSV_PATH) as fh:
    rows = list(csv.DictReader(fh))[:N_TARGETS]

targets = [{"id": r["ID"], "ra": float(r["RA"]), "dec": float(r["DEC"]),
            "z": float(r["REDSHIFT"])} for r in rows]

# ---------------------------------------------------------------- photometry
photo_config = {
    "surveys": ["GALEX_FUV", "GALEX_NUV", "Legacy_g", "Legacy_r", "Legacy_z", "WISE_W1", "WISE_W2"],
    "output_dir": OUT_DIR,
    "common_grid": {"pixscale": 1.375, "size": 45},
    "download_size": 1,
}

_real_downloader = pl_mod.hostphot_downloader
_download_time = {}


def _timed_downloader(ra, dec, surveys, size, cache_dir):
    t0 = time.time()
    out = _real_downloader(ra, dec, surveys, size, cache_dir)
    _download_time["last"] = time.time() - t0
    return out


pl_mod.hostphot_downloader = _timed_downloader

phot_results = []
pipeline_phot = Pipeline(config=photo_config, mode="photometry")
for t in targets:
    row = {"id": t["id"], "phot_status": "failed",
           "phot_download_s": np.nan, "phot_processing_s": np.nan}
    try:
        t0 = time.time()
        pipeline_phot.run(dict(t))
        total = time.time() - t0
        dl = _download_time.get("last", np.nan)
        row.update(phot_status="ok", phot_download_s=round(dl, 2),
                    phot_processing_s=round(total - dl, 2))
    except Exception as exc:
        row["phot_error"] = str(exc)[:200]
        logging.error("photometry failed for %s: %s", t["id"], exc)
    phot_results.append(row)
    print(f"[phot] {t['id']}: {row}")

pl_mod.hostphot_downloader = _real_downloader

# --------------------------------------------------------------- spectroscopy
from xpectrafit import XpectraFitter
from xpectrafit.core.spectrum import Spectrum
from xpectrafit.io.downloader import download_desi_sparcl

spec_results = []
spec_plot_dir = os.path.join(OUT_DIR, "spectroscopy_plots")
os.makedirs(spec_plot_dir, exist_ok=True)

for t in targets:
    row = {"id": t["id"], "spec_status": "failed",
           "spec_download_s": np.nan, "spec_processing_s": np.nan}
    try:
        t0 = time.time()
        recs = download_desi_sparcl(targetids=[int(t["id"])],
                                     output_dir=os.path.join(OUT_DIR, "desi_cache"))
        dl = time.time() - t0
        row["spec_download_s"] = round(dl, 2)
        if not recs:
            row["spec_error"] = "no DESI record for this targetid"
            spec_results.append(row)
            print(f"[spec] {t['id']}: {row}")
            continue

        rec = recs[0]
        spec = Spectrum.from_ivar(rec["wave"], rec["flux"], rec["ivar"],
                                   z=t["z"], target_id=str(t["id"]))

        t1 = time.time()
        result = XpectraFitter(spec.wave, spec.flux, spec.flux_err, z=t["z"],
                               ra=t["ra"], dec=t["dec"], target_id=str(t["id"]),
                               fwhm_gal=2.2, fit_agn=True,
                               survey_filters=["Legacy", "SDSS"]).fit()
        proc = time.time() - t1
        row.update(spec_status=getattr(result, "status", "ok"),
                    spec_processing_s=round(proc, 2))

        try:
            result.plot(os.path.join(spec_plot_dir, f"{t['id']}_spectrum.pdf"))
        except Exception as exc:
            logging.warning("plot failed for %s: %s", t["id"], exc)

    except Exception as exc:
        row["spec_error"] = str(exc)[:200]
        logging.error("spectroscopy failed for %s: %s", t["id"], exc)
    spec_results.append(row)
    print(f"[spec] {t['id']}: {row}")

# ------------------------------------------------------------------- report
import csv as _csv
merged = []
for t, p, s in zip(targets, phot_results, spec_results):
    row = {"id": t["id"], "ra": t["ra"], "dec": t["dec"], "z": t["z"]}
    row.update(p)
    row.update(s)
    merged.append(row)

out_csv = os.path.join(OUT_DIR, "timing_table.csv")
os.makedirs(OUT_DIR, exist_ok=True)
fieldnames = list(merged[0].keys())
with open(out_csv, "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(merged)

print("\n=== wrote", out_csv, "===")
for row in merged:
    print(row)
