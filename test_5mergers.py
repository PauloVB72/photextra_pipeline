"""Five-merger batch test. Coordinates of nearby galaxy pairs (sep ~2-5")."""

import logging

from photextra_pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

targets = [
    {"id": "merger_A", "ra": 352.37031, "dec": 1.75630},
    {"id": "merger_B", "ra": 150.08654, "dec": 2.21320},
    {"id": "merger_C", "ra": 36.55012,  "dec": -4.41890},
    {"id": "merger_D", "ra": 215.13380, "dec": 12.78450},
    {"id": "merger_E", "ra": 9.81245,   "dec": -0.99213},
]


def main():
    pipeline = Pipeline(config="config_xmask.yaml")
    for target in targets:
        try:
            result = pipeline.run(target)
            print(f"{target['id']}: ok -> {result['output_dir']}")
        except Exception as exc:
            print(f"{target['id']}: FAILED -> {exc}")


if __name__ == "__main__":
    main()
