"""Fast smoke test for the photextra CLI (no network, no real pipeline run).

Checks: cli imports, --help works, the banner renders, the targets-CSV
loader parses, and the rich progress wrapper drives pipe.run() per target
(with a stub Pipeline). Run:  python3 test_cli_smoke.py
"""

import io
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_import_and_help():
    from photextra_pipeline.cli import main  # noqa: F401
    out = subprocess.run(
        [sys.executable, "-m", "photextra_pipeline.cli", "--help"],
        capture_output=True, text=True, check=True)
    assert "photextra" in out.stdout and "--targets" in out.stdout


def test_banner_renders():
    from rich.console import Console
    from photextra_pipeline.cli import print_banner
    buf = io.StringIO()
    print_banner(Console(file=buf, force_terminal=True, width=100))
    text = buf.getvalue()
    assert "PHOTEXTRA" in text and "Bienvenido" in text


def test_progress_wraps_pipeline_run():
    from rich.console import Console
    from photextra_pipeline.cli import run_with_progress, load_targets_csv

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write("ID,RA,DEC,REDSHIFT,type\n"
                 "G1,150.1,2.2,0.05,merger\n"
                 "G2,150.2,2.3,0.06,\n")
        csv_path = fh.name
    try:
        targets = load_targets_csv(csv_path)
        assert targets[0]["id"] == "G1" and targets[0]["z"] == 0.05
        assert "type" not in targets[1]

        calls = []

        class FakePipe:  # stands in for Pipeline; run() signature identical
            def run(self, t):
                calls.append(t["id"])
                if t["id"] == "G2":
                    raise RuntimeError("boom")

        buf = io.StringIO()
        ok, fail, failures = run_with_progress(
            FakePipe(), targets, Console(file=buf, force_terminal=True))
        assert calls == ["G1", "G2"]
        assert (ok, fail) == (1, 1) and failures[0][0] == "G2"
    finally:
        os.unlink(csv_path)


if __name__ == "__main__":
    test_import_and_help()
    test_banner_renders()
    test_progress_wraps_pipeline_run()
    print("CLI smoke test: all OK")
