"""Parallel test: 6 MKW8 targets, mode='both' only, 6 workers (ProcessPoolExecutor).

Measures per-target wall time and the total batch wall time, to compare
throughput of running N targets concurrently (one process per core, BLAS
pinned to 1 thread each) vs strictly sequential.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/home/polo/Escritorio/PHD/code/photextra_pipeline")
sys.path.insert(0, "/home/polo/Escritorio/PHD/code/xdebpair")

BASE_OUT = "/home/polo/Escritorio/PHD/CHANCES/test_output_MKW8_6parallel_both"

TARGETS = [
    {"id": "39627860914209728", "ra": 221.04303924793248, "dec": 3.046006012432269, "z": 0.0304705561294698},
    {"id": "39627878970687794", "ra": 218.56510362660336, "dec": 3.83134367363413,  "z": 0.0288377687469585},
    {"id": "39627878970687929", "ra": 218.57126607708875, "dec": 3.7242357264935486, "z": 0.0295195258737501},
    {"id": "2842599903723520",  "ra": 221.5409963856697,  "dec": 3.0577761567740365, "z": 0.0284285775027491},
    {"id": "39627878966496687", "ra": 218.46527887933453, "dec": 3.775127599988316,  "z": 0.0277214740170827},
    {"id": "39627878966496974", "ra": 218.4771514666284,  "dec": 3.713309295632941,  "z": 0.0282482537619546},
]

N_WORKERS = 6

# standard 7-survey band set for all test/production runs
SURVEYS = ["GALEX_FUV", "GALEX_NUV", "Legacy_g", "Legacy_r", "Legacy_z",
           "WISE_W1", "WISE_W2"]
DOWNLOAD_SIZE = 1


def _run_one(tgt):
    import logging
    import time as _time
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    from photextra_pipeline.pipeline import Pipeline

    cfg = {
        "surveys": SURVEYS,
        "output_dir": BASE_OUT,
        "common_grid": {"pixscale": 1.375, "size": 45},
        "download_size": DOWNLOAD_SIZE,
    }
    row = {"target_id": tgt["id"], "status": "FAILED", "total_s": None, "error": ""}
    t0 = _time.perf_counter()
    try:
        pipe = Pipeline(config=dict(cfg), deblend=True, mode="both")
        pipe.run(dict(tgt))
        row["status"] = "OK"
    except Exception as exc:
        row["error"] = str(exc)[:300]
    row["total_s"] = round(_time.perf_counter() - t0, 2)
    return row


if __name__ == "__main__":
    os.makedirs(BASE_OUT, exist_ok=True)
    grand_t0 = time.perf_counter()

    # Stage 1: pre-fetch all imaging at low concurrency (rate-limit safe),
    # so the parallel fit workers below only read from cache.
    from photextra_pipeline.downloader import prefetch_images
    pf_t0 = time.perf_counter()
    pf_failures = prefetch_images(TARGETS, SURVEYS, DOWNLOAD_SIZE, BASE_OUT,
                                  max_workers=2)
    pf_s = time.perf_counter() - pf_t0
    print(f"=== prefetch stage {pf_s:.1f}s, {len(pf_failures)} target(s) "
          f"with failed bands ===", flush=True)

    # Stage 2: CPU-bound parallel fit stage
    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_run_one, t): t for t in TARGETS}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"target_id": t["id"], "status": "FAILED", "total_s": None,
                     "error": str(exc)[:300]}
            print(f"[{r['target_id']}] {r['status']} total={r['total_s']}s {r['error']}",
                  flush=True)
            results.append(r)

    grand = time.perf_counter() - grand_t0
    print(f"\n=== 6-parallel batch grand total {grand:.1f}s ({grand/60:.1f} min) ===",
          flush=True)
    with open(os.path.join(BASE_OUT, "timing_6parallel.json"), "w") as fh:
        json.dump({"results": results, "grand_total_s": round(grand, 2),
                   "prefetch_s": round(pf_s, 2),
                   "prefetch_failures": pf_failures,
                   "surveys": SURVEYS,
                   "n_workers": N_WORKERS}, fh, indent=2)
    print("RESUME DONE", flush=True)
