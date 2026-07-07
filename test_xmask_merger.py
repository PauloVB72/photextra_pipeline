"""Single-merger smoke test."""

import logging

from photextra_pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    target = {"id": "merger_test", "ra": 352.37031, "dec": 1.75630}
    pipeline = Pipeline(config="config_xmask.yaml")
    result = pipeline.run(target)
    print("Output dir:", result["output_dir"])


if __name__ == "__main__":
    main()
