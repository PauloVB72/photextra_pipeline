"""Run photextra on 5 CHANCES major-merger targets and report timing."""

import sys, time, logging
sys.path.insert(0, "/home/polo/Escritorio/PHD/code/photextra_pipeline")
sys.path.insert(0, "/home/polo/Escritorio/PHD/code/xdebpair")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from photextra_pipeline.pipeline import Pipeline

TARGETS = [
    {"id": "422953_1208",  "ra": 31.647589,  "dec": 16.206546},
    {"id": "263074_2959",  "ra": 348.352485, "dec": -11.978438},
    {"id": "499412_1386",  "ra": 357.336435, "dec": 30.446392},
    {"id": "504976_3115",  "ra": 180.042354, "dec": 31.800835},
    {"id": "498203_3262",  "ra": 7.356562,   "dec": 30.556970},
]

CFG = {
    "surveys": ["GALEX_FUV", "GALEX_NUV", "Legacy_g", "Legacy_r", "Legacy_z",
                "WISE_W1", "WISE_W2"],
    "output_dir": "test_output_CHANCES_mm5",
    "common_grid": {"pixscale": 1.375, "size": 45},
    "download_size": 1.5,
}

pipe = Pipeline(CFG, use_xdebpair=True)

t0_all = time.perf_counter()
timings = {}

for tgt in TARGETS:
    t0 = time.perf_counter()
    try:
        pipe.run(tgt)
        status = "OK"
    except Exception as exc:
        status = f"FAILED: {exc}"
    elapsed = time.perf_counter() - t0
    timings[tgt["id"]] = (elapsed, status)
    print(f"  {tgt['id']:20s}  {elapsed:6.1f}s  {status}")

total = time.perf_counter() - t0_all
print(f"\n{'='*50}")
print(f"Total: {total:.1f}s  ({total/60:.1f} min)")
for tid, (t, s) in timings.items():
    print(f"  {tid:20s}  {t:6.1f}s  {s}")
