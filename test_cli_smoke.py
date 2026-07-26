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


def test_set_parsing_and_apply():
    """--set: dotted paths, YAML-typed values, order, typo warning."""
    import io
    from rich.console import Console
    from photextra_pipeline.cli import (parse_set, apply_set_overrides,
                                        key_warning)

    assert parse_set("mode=both") == (["mode"], "both")
    assert parse_set("deblend=false") == (["deblend"], False)
    assert parse_set("common_grid.size=60") == (["common_grid", "size"], 60)
    assert parse_set("sep.thresh_sigma=2.0") == (["sep", "thresh_sigma"], 2.0)
    assert parse_set("surveys=[Legacy_g,Legacy_r]") == (
        ["surveys"], ["Legacy_g", "Legacy_r"])
    for bad in ("mode", "=both"):
        try:
            parse_set(bad)
            raise AssertionError(f"should have rejected {bad!r}")
        except ValueError:
            pass

    config = {"mode": "photometry", "photometry": {"separation": "pair"}}
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    applied = apply_set_overrides(
        config,
        ["mode=both", "photometry.aperture_mode=aperture",
         "sep.ellipse_k=3.0"],   # sep: section does not exist yet
        console)
    assert config["mode"] == "both"
    assert config["photometry"] == {"separation": "pair",
                                    "aperture_mode": "aperture"}
    assert config["sep"] == {"ellipse_k": 3.0}
    assert applied[0] == "mode = both"

    # known keys are silent, typos get a suggestion (but are still applied)
    assert key_warning(["photometry", "aperture_mode"]) is None
    assert "photometry.aperture_mode" in key_warning(["photomety",
                                                      "aperture_mode"])


def test_save_roundtrip():
    """--save writes the EFFECTIVE config and never clobbers without --force."""
    import yaml
    from photextra_pipeline.cli import (save_config, default_save_path,
                                        apply_set_overrides)

    tmpdir = tempfile.mkdtemp()
    src = os.path.join(tmpdir, "config_x.yaml")
    with open(src, "w") as fh:
        fh.write("mode: photometry\nphotometry:\n  separation: pair\n")

    assert default_save_path(src) == os.path.join(tmpdir, "config_x.new.yaml")

    config = yaml.safe_load(open(src))
    apply_set_overrides(config, ["mode=both", "sep.ellipse_k=3.0"])
    out = default_save_path(src)
    ok, _ = save_config(config, out, src, force=False)
    assert ok
    back = yaml.safe_load(open(out))
    assert back["mode"] == "both" and back["sep"]["ellipse_k"] == 3.0
    assert back["photometry"]["separation"] == "pair"

    # existing file (and the source config) are protected without --force
    ok, msg = save_config(config, out, src, force=False)
    assert not ok and "--force" in msg
    ok, msg = save_config(config, src, src, force=False)
    assert not ok
    assert yaml.safe_load(open(src))["mode"] == "photometry"  # untouched
    ok, _ = save_config(config, src, src, force=True)
    assert ok and yaml.safe_load(open(src))["mode"] == "both"


def test_show_config_cli():
    """--show-config prints the effective config and runs nothing."""
    tmpdir = tempfile.mkdtemp()
    cfg = os.path.join(tmpdir, "c.yaml")
    with open(cfg, "w") as fh:
        fh.write("mode: photometry\nsurveys: [Legacy_r]\noutput_dir: /tmp/o\n")
    out = subprocess.run(
        [sys.executable, "-m", "photextra_pipeline.cli", cfg,
         "--set", "mode=both", "--show-config"],
        capture_output=True, text=True, check=True)
    assert "mode: both" in out.stdout


if __name__ == "__main__":
    test_import_and_help()
    test_banner_renders()
    test_progress_wraps_pipeline_run()
    test_set_parsing_and_apply()
    test_save_roundtrip()
    test_show_config_cli()
    print("CLI smoke test: all OK")
