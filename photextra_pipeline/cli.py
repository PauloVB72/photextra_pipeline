"""photextra — terminal entry point (presentation layer only).

This module is purely cosmetic/UX: it prints a welcome banner, summarizes
the loaded config, and drives :class:`~photextra_pipeline.pipeline.Pipeline`
per target inside a rich progress bar. It contains NO science logic and
does not modify Pipeline's behavior in any way — scripts that import
``Pipeline`` directly (run_mkw8_full.py etc.) are completely unaffected.

Usage::

    photextra                        # uses ./config_xmask.yaml
    photextra path/to/config.yaml
    photextra config.yaml --targets members.csv --limit 5

Targets are taken from (in order of precedence):
  1. ``--targets`` CSV on the command line (columns ID, RA, DEC,
     REDSHIFT, optional type — same format as the driver scripts),
  2. a ``targets_csv`` key in the YAML config,
  3. an inline ``targets:`` list of {id, ra, dec, z} dicts in the config.
"""

import argparse
import csv
import logging
import os
import sys

import yaml

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich._spinners import SPINNERS
from rich.progress import (Progress, SpinnerColumn, BarColumn,
                           TaskProgressColumn, TimeElapsedColumn,
                           TimeRemainingColumn, TextColumn)

DEFAULT_CONFIG = "config_xmask.yaml"

# ---------------------------------------------------------------------------
# banner
# ---------------------------------------------------------------------------

# Hand-rolled block letters (pyfiglet not installed; kept small on purpose).
_BANNER_LINES = [
    "█▀█ █ █ █▀█ ▀█▀ █▀▀ ▀▄▀ ▀█▀ █▀█ █▀█",
    "█▀▀ █▀█ █ █  █  █▀▀  █   █  █▀▄ █▀█",
    "▀   ▀ ▀ ▀▀▀  ▀  ▀▀▀ ▀ ▀  ▀  ▀ ▀ ▀ ▀",
]

# Two interlocked / merging spiral galaxies (S-shaped tidal interaction):
# navy tendrils sweep between two glowing cores, medium-blue spiral arms,
# scattered white stars. Base style per line = medium blue (arm body); the
# navy tendrils, yellow cores and white stars are set via inline markup.
_GALAXY_LINES = [
    # Fila 1 (Punta exterior del brazo derecho)
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[navy_blue]⣀⣤⣴⣲⣶⣶⣦⣄⠀⠀⠀", ""),
    # Fila 2
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[deep_sky_blue1]⣠⣾⣿⣿⣽⡷⠾⣾⣭⣟⣷⡄⠀", ""),
    # Fila 3
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[medium_purple1]⣼⣿⣵⣿⠟⠁⠀⠀⠀⠈⢻⣿⣿⡀", ""),
    # Fila 4 (Acercándose al núcleo)
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[magenta1]⣸⢯⡿⡿⢁⣤⣶⣶⣄⠀⠀⠀⢻⣿⡇", ""),
    # Fila 5 (Borde superior del núcleo)
    ("⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[white]⢠⣿⢺⣽⣳⣿⢛⣽⣯⣿⡆⠀⠀⠀⠛⠃", ""),
    # Fila 6 (Comienza brazo izquierdo, centro derecha brillante)
    ("⠀⠀⠀⠀[navy_blue]⢠⣶⣟⣿⣿⠆⠀[deep_sky_blue1]⢀⣀⣤⣤⣶⢶⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀[white]⢸⣿⡁⣿⣿⠓⡍⢍⠻⣿⣇⠀⠀⠀⠀⠀", ""),
    # Fila 7
    ("⠀⠀⠀[deep_sky_blue1]⢀⡿⣹⡏⣠⣤⡴⣶⡛[medium_purple1]⣻⠽⠿⠿[white]⠿⠿⣿⣿⡿⣷⣄⣀⠀⠀⠀[white]⢸⣿⣷⡘⢇⠀⠀⢑⢱⣽⣿⠀⠀⠀⠀⠀", ""),
    # Fila 8 (Centro absoluto)
    ("⠀⠀⠀[medium_purple1]⢸⣿⣻⣿⣿⢋⠾⠋[magenta1]⡾⢿⡦⣴⣶[white]⣯⡿⢻⣟⣛⣀⣙⢛⣛⣻⠿⠯⡿⢿[magenta1]⡻⢶⡡⢀⠈⣾⣧⡿⠀⠀⠀⠀⠀", ""),
    # Fila 9
    ("⠀⠀⠀[magenta1]⠈⣿⣎⢿⡟⠸⠢⢫⢻[white]⣷⣮⢉⣽⢶⣿⠯⠛⠛⠋⠉⢳⣿⣟[magenta1]⠿⣷⢶[medium_purple1]⣬⣛⣓⣊⡽[deep_sky_blue1]⢰⡿⡽⠃⠀⠀⠀⠀⠀", ""),
    # Fila 10
    ("⠀⠀⠀⠀[medium_purple1]⠹⣿⣶⣉⠤⡄⠀[magenta1]⠐⡻⣿⣯⣞⠉⠀⠀⠀⠀⠀⠀⠀[medium_purple1]⠙⢯⣗⡦⣍⣻⣿⡷[deep_sky_blue1]⢟⣵⣿⠟⠁⠀⠀⠀⠀⠀⠀", ""),
    # Fila 11
    ("⠀⠀⠀⠀⠀[deep_sky_blue1]⠙⢿⣟⡷⠶⣓⡇⡢⢻⣧⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[navy_blue]⠈⠙⠛⠓⠒⠚⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀", ""),
    # Fila 12 (Cola del brazo izquierdo)
    ("[navy_blue]⢀⣤⡀⠀⠀⠀⠀[deep_sky_blue1]⠙⠻⢿⣿⣾⢷⢸⣯⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", ""),
    # Fila 13
    ("[navy_blue]⠈⣿⣻⣷⣤⣀⣀⣀⣠⣤⣞⣷⣫[deep_sky_blue1]⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", ""),
    # Fila 14
    ("⠀[navy_blue]⠈⠻⢷⣽⣛⣿⡷⣷⣞⣯⣾⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", ""),
    # Fila 15 (Punta exterior del brazo izquierdo)
    ("⠀⠀⠀⠀[navy_blue]⠉⠙⠛⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀", "")
]
# banner gradient colors, one per line
_BANNER_COLORS = ("bold magenta", "bold medium_purple", "bold deep_sky_blue1")

WELCOME_MSG = ("Welcome to PHOTEXTRA — photometry multiwavelength pipeline"
               "& spectroscopy to morphology on galaxies")


def print_banner(console=None):
    """Print the PHOTEXTRA welcome banner (banner + spiral galaxy + message)."""
    console = console or Console()
    body = Text()
    for line, color in zip(_BANNER_LINES, _BANNER_COLORS):
        body.append(line + "\n", style=color)
    body.append("\n")
    for line, style in _GALAXY_LINES:
        body.append_text(Text.from_markup(line, style=style))
        body.append("\n")
    body.append("\n")
    body.append(WELCOME_MSG, style="italic cyan")
    console.print(Panel(body, border_style="medium_purple",
                        title="[bold yellow]* PHOTEXTRA *[/]",
                        subtitle="[dim]PhD. UGR P.Vasquez-Bustos [/]",
                        expand=False))


# ---------------------------------------------------------------------------
# config / targets loading (thin wrappers, no science)
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_targets_csv(path, limit=None):
    """CSV -> list of target dicts (same columns as the driver scripts)."""
    targets = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            low = {k.strip().lower(): v for k, v in row.items()}
            tgt = {"id": str(low["id"]).strip(),
                   "ra": float(low["ra"]),
                   "dec": float(low["dec"])}
            z = low.get("redshift", low.get("z"))
            if z not in (None, ""):
                tgt["z"] = float(z)
            if str(low.get("type", "")).strip():
                tgt["type"] = str(low["type"]).strip()
            targets.append(tgt)
    if limit:
        targets = targets[:limit]
    return targets


def resolve_targets(config, args):
    """--targets CSV > config targets_csv > config inline targets list."""
    if args.targets:
        return load_targets_csv(args.targets, args.limit)
    if config.get("targets_csv"):
        return load_targets_csv(config["targets_csv"], args.limit)
    targets = config.get("targets") or []
    if args.limit:
        targets = targets[:args.limit]
    return list(targets)


def print_config_summary(console, config, config_path, targets):
    """Human-readable 'what is about to run' summary from the config."""
    phot = config.get("photometry") or {}
    tbl = Table(show_header=False, box=None, padding=(0, 1))
    tbl.add_column(style="bold cyan", justify="right")
    tbl.add_column(style="white")
    tbl.add_row("config", str(config_path))
    tbl.add_row("mode", str(config.get("mode", "photometry")))
    tbl.add_row("both_method", str(config.get("both_method", "own")))
    tbl.add_row("aperture_mode", str(phot.get("aperture_mode", "mask")))
    tbl.add_row("separation", str(phot.get("separation", "pair")))
    tbl.add_row("surveys", ", ".join(config.get("surveys") or []))
    tbl.add_row("output_dir", str(config.get("output_dir", "")))
    tbl.add_row("targets", f"{len(targets)}")
    console.print(Panel(tbl, title="[bold]Run summary[/]",
                        border_style="deep_sky_blue1", expand=False))


# ---------------------------------------------------------------------------
# progress-wrapped run
# ---------------------------------------------------------------------------

# 4-frame rotating spiral arms around a yellow nucleus. Pure ASCII: renders
# reliably in any terminal (unlike emoji, which can misalign the bar).
SPINNERS["galaxy"] = {
    "interval": 160,
    "frames": ["-*@*-", "\\*@*/", "|*@*|", "/*@*\\"],
}


def run_with_progress(pipe, targets, console):
    """Drive pipe.run(target) per target under a rich progress bar.

    Purely a wrapper: each target still goes through the unmodified
    Pipeline.run(). Failures are caught per target so one bad galaxy
    never kills the run (same policy as the driver scripts).
    """
    n_ok = n_fail = 0
    failures = []
    progress = Progress(
        SpinnerColumn(spinner_name="galaxy", style="bold magenta"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="medium_purple",
                  finished_style="deep_sky_blue1"),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed}/{task.total}[/]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("processing", total=len(targets))
        for tgt in targets:
            tid = str(tgt.get("id"))
            progress.update(task, description=f"[bold]{tid}[/]")
            try:
                pipe.run(tgt)
                n_ok += 1
            except Exception as exc:  # keep going; report at the end
                n_fail += 1
                failures.append((tid, str(exc)))
                progress.console.print(
                    f"  [red]x[/] {tid} failed: {exc}")
            progress.advance(task)
    return n_ok, n_fail, failures


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="photextra",
        description=("PHOTEXTRA — multiband photometry + spectroscopy "
                     "pipeline for merging galaxies."))
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                    help=f"YAML config path (default: {DEFAULT_CONFIG})")
    ap.add_argument("--targets", default=None, metavar="CSV",
                    help="target list CSV (columns ID, RA, DEC, REDSHIFT, "
                         "optional type); overrides the config's "
                         "targets_csv / targets keys")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="only process the first N targets")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    console = Console()
    print_banner(console)

    if not os.path.exists(args.config):
        console.print(f"[bold red]Config not found:[/] {args.config}")
        return 1
    config = load_config(args.config)

    try:
        targets = resolve_targets(config, args)
    except (KeyError, ValueError) as exc:
        console.print(f"[bold red]Could not read targets:[/] {exc}")
        return 1

    print_config_summary(console, config, args.config, targets)

    if not targets:
        console.print(
            "[yellow]No targets to run.[/] Provide them with "
            "[bold]--targets file.csv[/], or add a [bold]targets_csv[/] "
            "path (or an inline [bold]targets:[/] list) to the config.")
        return 0

    # quiet INFO chatter so the progress bar stays readable; warnings/errors
    # still show. Pipeline's own logging behavior is untouched elsewhere.
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    from .pipeline import Pipeline
    pipe = Pipeline(config=config)

    n_ok, n_fail, failures = run_with_progress(pipe, targets, console)

    style = "bold green" if n_fail == 0 else "bold yellow"
    console.print(f"[{style}]Done:[/] {n_ok} ok, {n_fail} failed "
                  f"-> {config.get('output_dir', '')}")
    for tid, err in failures:
        console.print(f"  [red]{tid}[/]: {err}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
