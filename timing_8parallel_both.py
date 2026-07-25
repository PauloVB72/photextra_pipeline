"""Parallel test: 8 MKW8 targets, mode='both' only, 8 workers (ProcessPoolExecutor).

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

BASE_OUT = "/home/polo/Escritorio/PHD/CHANCES/test_output_MKW8_8parallel_both_v2"

TARGETS = [
    {"id": "39627878966494092", "ra": 218.34276405795933, "dec": 3.7429593657794262, "z": 0.0292386953370364},
    {"id": "39627879000052206", "ra": 220.554217238498,   "dec": 3.741250580066781,  "z": 0.02538269685082},
    {"id": "39627860876460470", "ra": 218.76896187726413, "dec": 2.8825699813057795, "z": 0.0292453864035358},
    {"id": "39627872968643444", "ra": 220.54472027706893, "dec": 3.525600928461484,  "z": 0.0307666140860241},
    {"id": "39627878966496974", "ra": 218.4771514666284,  "dec": 3.713309295632941,  "z": 0.0282482537619546},
    {"id": "39627866907871914", "ra": 218.66018168783265, "dec": 3.1871343789583757, "z": 0.0287434293475448},
    {"id": "39627884989516896", "ra": 217.93234416584272, "dec": 4.0090446439647325, "z": 0.0289002622815852},
    {"id": "39627866933038260", "ra": 220.18063633655413, "dec": 3.3740161722432607, "z": 0.0319579897619362},
]

N_WORKERS = 8

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

    # --- Stage 1: pre-fetch all imaging at low concurrency ----------------
    # Downloads hit a shared external rate limit (Legacy Survey returned 429
    # at 8 concurrent workers), so they run BEFORE the parallel fit stage,
    # capped at 2 concurrent requests. The fit workers then read from cache.
    from photextra_pipeline.downloader import prefetch_images
    pf_t0 = time.perf_counter()
    pf_failures = prefetch_images(TARGETS, SURVEYS, DOWNLOAD_SIZE, BASE_OUT,
                                  max_workers=2)
    pf_s = time.perf_counter() - pf_t0
    print(f"=== prefetch stage {pf_s:.1f}s, {len(pf_failures)} target(s) "
          f"with failed bands ===", flush=True)

    # --- Stage 2: CPU-bound parallel fit stage (reads from cache) ---------
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
    print(f"\n=== 8-parallel batch grand total {grand:.1f}s ({grand/60:.1f} min) ===",
          flush=True)
    with open(os.path.join(BASE_OUT, "timing_8parallel.json"), "w") as fh:
        json.dump({"results": results, "grand_total_s": round(grand, 2),
                   "prefetch_s": round(pf_s, 2),
                   "prefetch_failures": pf_failures,
                   "surveys": SURVEYS,
                   "n_workers": N_WORKERS}, fh, indent=2)
    print("RESUME DONE", flush=True)
