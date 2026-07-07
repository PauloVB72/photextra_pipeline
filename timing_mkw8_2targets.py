"""Timing run: 2 MKW8 members x 3 modes (photometry/spectroscopy/both).

Times each pipeline.run() with perf_counter, and separates stage times via
monkeypatch seams:
  - image download : pl_mod.hostphot_downloader
  - spectrum acquisition : spectrum_acquisition.ensure_spectrum
  - spectral fit : spectral_fit.run_spectral_fit
Processing time is (total - measured stages) for that mode.
Writes results to a JSON so a separate step can build the HTML report.
"""
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/home/polo/Escritorio/PHD/code/photextra_pipeline")
sys.path.insert(0, "/home/polo/Escritorio/PHD/code/xdebpair")

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s: %(message)s")

from photextra_pipeline import pipeline as pl_mod
from photextra_pipeline import spectrum_acquisition as spec_acq
from photextra_pipeline import spectral_fit as spec_fit
from photextra_pipeline.pipeline import Pipeline

TARGETS = [
    {"id": "39627860855490651", "ra": 217.58312266961883, "dec": 3.0914309347536766, "z": 0.0312236620139741},
    {"id": "39627854845054783", "ra": 219.44362353598808, "dec": 2.832713286105544,  "z": 0.0262800576656983},
]
MODES = ["photometry", "spectroscopy", "both"]
BASE_OUT = "/home/polo/Escritorio/PHD/CHANCES/test_output_MKW8_2targets"

# ---- monkeypatch seams -----------------------------------------------------
_stage = {}  # reset per run

_real_downloader = pl_mod.hostphot_downloader
_real_ensure = spec_acq.ensure_spectrum
_real_fit = spec_fit.run_spectral_fit


def _timed_downloader(*a, **k):
    t0 = time.perf_counter()
    out = _real_downloader(*a, **k)
    _stage["download_s"] = _stage.get("download_s", 0.0) + (time.perf_counter() - t0)
    return out


def _timed_ensure(*a, **k):
    t0 = time.perf_counter()
    out = _real_ensure(*a, **k)
    _stage["spec_acq_s"] = _stage.get("spec_acq_s", 0.0) + (time.perf_counter() - t0)
    return out


def _timed_fit(*a, **k):
    t0 = time.perf_counter()
    out = _real_fit(*a, **k)
    _stage["spec_fit_s"] = _stage.get("spec_fit_s", 0.0) + (time.perf_counter() - t0)
    return out


pl_mod.hostphot_downloader = _timed_downloader
spec_acq.ensure_spectrum = _timed_ensure
spec_fit.run_spectral_fit = _timed_fit
# _run_spectroscopy does `from .spectral_fit import run_spectral_fit` at call
# time, so patch that module attribute too (the source object is looked up).

results = []
grand_t0 = time.perf_counter()

for tgt in TARGETS:
    for mode in MODES:
        out_dir = os.path.join(BASE_OUT, mode)
        cfg = {
            "surveys": ["GALEX_FUV", "GALEX_NUV", "Legacy_g", "Legacy_r", "Legacy_z", "WISE_W1", "WISE_W2"],
            "output_dir": out_dir,
            "common_grid": {"pixscale": 1.375, "size": 45},
            "download_size": 1,
        }
        _stage.clear()
        row = {"target_id": tgt["id"], "mode": mode, "output_dir": out_dir,
               "status": "FAILED", "total_s": None,
               "download_s": None, "spec_acq_s": None,
               "spec_fit_s": None, "processing_s": None, "error": ""}
        try:
            pipe = Pipeline(config=dict(cfg), use_xdebpair=True, mode=mode)
            t0 = time.perf_counter()
            pipe.run(dict(tgt))
            total = time.perf_counter() - t0
            dl = _stage.get("download_s")
            sa = _stage.get("spec_acq_s")
            sf = _stage.get("spec_fit_s")
            measured = sum(x for x in (dl, sa, sf) if x)
            row.update(status="OK", total_s=round(total, 2),
                       download_s=round(dl, 2) if dl is not None else None,
                       spec_acq_s=round(sa, 2) if sa is not None else None,
                       spec_fit_s=round(sf, 2) if sf is not None else None,
                       processing_s=round(total - measured, 2))
        except Exception as exc:
            row["error"] = str(exc)[:300]
            logging.error("FAILED %s/%s: %s", tgt["id"], mode, exc)
        print(f"[{tgt['id']} | {mode:12s}] {row['status']:6s} "
              f"total={row['total_s']} dl={row['download_s']} "
              f"acq={row['spec_acq_s']} fit={row['spec_fit_s']} "
              f"proc={row['processing_s']} {row['error']}")
        results.append(row)

grand = time.perf_counter() - grand_t0
print(f"\n=== grand total {grand:.1f}s ({grand/60:.1f} min) ===")

os.makedirs(BASE_OUT, exist_ok=True)
with open(os.path.join(BASE_OUT, "timing_results.json"), "w") as fh:
    json.dump({"results": results, "grand_total_s": round(grand, 2)}, fh, indent=2)
print("wrote", os.path.join(BASE_OUT, "timing_results.json"))
