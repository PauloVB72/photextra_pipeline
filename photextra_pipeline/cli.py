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
    photextra config.yaml --targets members.csv --dry-run
    photextra config.yaml --mode both --no-deblend --output-dir /tmp/out
    photextra config.yaml --set sep.thresh_sigma=2.0 --show-config
    photextra config.yaml --edit          # abre el YAML en $EDITOR

Targets are taken from (in order of precedence):
  1. ``--targets`` CSV on the command line (columns ID, RA, DEC,
     REDSHIFT, optional type — same format as the driver scripts),
  2. a ``targets_csv`` key in the YAML config,
  3. an inline ``targets:`` list of {id, ra, dec, z} dicts in the config.

CLI flags are all OPTIONAL: every run-time knob still lives in the YAML
config, and a flag simply OVERWRITES that key in the already-loaded config
dict before Pipeline is constructed (see :func:`apply_overrides`). Nothing
here reaches into pipeline internals — the config dict is Pipeline's own
documented public input.

CONFIG EDITING (three layers, in increasing scope):
  1. the named flags (--mode, --deblend, ...) cover the handful of knobs
     changed every day: discoverable in --help, typo-proof, tab-completable;
  2. ``--set dotted.path=value`` covers EVERY key, present and future,
     without ever adding another argparse flag (values go through
     ``yaml.safe_load`` so lists/bools/numbers are typed naturally);
  3. ``--edit`` opens the YAML itself in $EDITOR when many things change
     at once (and it is the only way to write/keep comments).
``--set`` is applied AFTER the named flags, so it always wins. All three
are in-memory only except --edit; ``--save`` is the explicit, opt-in way to
persist an effective config to disk. The input YAML is still never modified
behind the user's back.

SCOPE NOTE (per-bin timing): the progress bar reports per-TARGET times
(current target, running average, ETA) and NOT per spectral bin. That is a
deliberate decision, not an oversight: ``Pipeline.run(target)`` is a single
opaque call with no callback/progress hooks, so per-bin (or per-stage)
timing would require instrumenting ``pipeline.py`` itself, which would
break this module's presentation-only boundary. If per-bin progress is ever
wanted, the right fix is to add an optional ``progress_cb`` hook to
Pipeline first, and only then consume it from here.
"""

import argparse
import csv
import difflib
import logging
import os
import shutil
import subprocess
import sys
import time

import yaml

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich._spinners import SPINNERS
from rich.progress import (Progress, SpinnerColumn, BarColumn,
                           TaskProgressColumn, TimeElapsedColumn,
                           TimeRemainingColumn, TextColumn, ProgressColumn)

DEFAULT_CONFIG = "config_xmask.yaml"

# Template written by --init. Mirrors config_xmask.yaml's documented
# defaults (mode: photometry, aperture_mode: mask, deblend: true, the
# standard 9-band survey list) with placeholder targets/output_dir left
# commented out for the user to fill in.
_INIT_TEMPLATE = """\
# =============================================================================
# photextra_pipeline configuration (generado por: photextra --init)
# =============================================================================
mode: photometry            # photometry | spectroscopy | both

cigale_run: false
cigale_batch: false

photometry:
  aperture_mode: mask       # mask | aperture | sep_apertures
  separation: pair          # pair | central | total
  aperture_radius_arcsec: 5.0

spectroscopy:
  fit_agn: true
  spectrum_fwhm: 2.5
  survey_filters: [Legacy, SDSS]

deblend: true

surveys:
  - GALEX_FUV
  - GALEX_NUV
  - Legacy_g
  - Legacy_r
  - Legacy_z
  - WISE_W1
  - WISE_W2
  - WISE_W3
  - WISE_W4

download_size: 1

common_grid:
  reference: WISE_W4
  pixscale: 1.375
  size: 45

output_dir: ./test_output

# --- targets ---
# Opcion A: CSV con columnas ID, RA, DEC, REDSHIFT, type (opcional)
# targets_csv: members.csv
# Opcion B: lista inline
# targets:
#   - {id: ejemplo, ra: 150.0, dec: 2.0, z: 0.05, type: merger}
"""

try:  # version for -V/--version (package __init__ is imported anyway)
    from . import __version__ as _VERSION
except Exception:  # pragma: no cover - defensive only
    _VERSION = "unknown"

# Valid choices for the CLI overrides. Duplicated here ON PURPOSE as plain
# strings instead of importing pipeline.VALID_* : this module must not grow
# import-time coupling to the science layer, and argparse only needs them to
# print --help and reject typos (Pipeline re-validates them anyway).
MODES = ("photometry", "spectroscopy", "both")
APERTURE_MODES = ("mask", "aperture", "sep_apertures")
SEPARATIONS = ("central", "total", "pair")

# ---------------------------------------------------------------------------
# known config keys (documentation + typo hints for --set)
# ---------------------------------------------------------------------------
# Mirror of every key Pipeline actually reads out of the config dict (all of
# them are consumed in Pipeline.__init__; nothing else in the package touches
# self.config), plus the two CLI-only target keys. Kept as plain data here,
# NOT imported from pipeline.py, for the same reason as MODES above: this
# module must not gain import-time coupling to the science layer.
#
# It is used for (a) --list-keys, (b) a *non-blocking* fuzzy typo warning on
# --set. It is deliberately NOT a schema: an unknown key is warned about and
# then applied anyway, because the config is genuinely open-ended (new keys
# land in pipeline.py before this table is updated) and Pipeline does its own
# validation. Descriptions are one-phrase versions of config_xmask.yaml's.
KNOWN_KEYS = {
    "mode": "photometry | spectroscopy | both",
    "cigale_run": "run the additional CIGALE SED fit (needs the cigale env)",
    "cigale_batch": "skip the per-target CIGALE fit (driver calls the batch)",
    "deblend": "attempt xdebpair source separation (else single-aperture stub)",
    "use_xdebpair": "legacy alias of deblend (still honoured)",
    "surveys": "list of imaging bands to download and measure",
    "download_size": "cutout size in arcmin",
    "survey_filters": "top-level fallback for spectroscopy.survey_filters",
    "output_dir": "where all per-target products/plots are written",
    "targets_csv": "path to the target list CSV (CLI only)",
    "targets": "inline list of {id, ra, dec, z, type} dicts (CLI only)",
    "photometry.aperture_mode": "mask | aperture | sep_apertures",
    "photometry.separation": "pair | central | total (mask mode only)",
    "photometry.aperture_radius_arcsec": "circular aperture radius, arcsec",
    "sep.thresh_sigma": "SEP detection threshold in background-rms units",
    "sep.ellipse_k": "Kron-like scale of the measured (a,b) half-axes",
    "sep.max_match_arcsec": "max distance to accept a SEP source as the target",
    "sep.ref_survey": "band the SEP ellipse is detected on",
    "spectroscopy.fit_agn": "fit an AGN component in XpectraFit",
    "spectroscopy.spectrum_fwhm": "instrumental FWHM of the spectrum, Angstrom",
    "spectroscopy.survey_filters": "survey families for synthetic photometry",
    "common_grid.reference": "documentation only; grid is set by pixscale/size",
    "common_grid.pixscale": "analysis-grid pixel scale, arcsec/pix",
    "common_grid.size": "analysis-grid size, pixels",
}

# Top-level keys that hold a nested mapping (--set may descend into them).
_NESTED_SECTIONS = ("photometry", "sep", "spectroscopy", "common_grid")


# ---------------------------------------------------------------------------
# --set: generic dotted-path overrides
# ---------------------------------------------------------------------------

def parse_set(assignment):
    """``"a.b=value"`` -> ``(["a", "b"], parsed_value)``.

    The value is run through ``yaml.safe_load`` so the user types plain YAML:
    ``mode=both`` (str), ``deblend=false`` (bool), ``common_grid.size=60``
    (int), ``surveys=[Legacy_g,Legacy_r]`` (list). Anything yaml cannot parse
    (e.g. an unquoted path with a colon) is kept as the raw string.
    """
    if "=" not in assignment:
        raise ValueError(f"--set espera clave=valor, recibi: {assignment!r}")
    path, _, raw = assignment.partition("=")
    path = path.strip()
    if not path:
        raise ValueError(f"--set sin clave: {assignment!r}")
    keys = [p for p in path.split(".") if p]
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    if isinstance(value, str):
        value = value.strip()
    return keys, value


def key_warning(keys):
    """Non-blocking typo hint for a --set path, or None if it looks fine.

    Only reports when the path is unknown AND a close known key exists (or the
    path descends into a non-nested top-level key). Never blocks: a genuinely
    new/valid key must keep working without editing this file.
    """
    dotted = ".".join(keys)
    if dotted in KNOWN_KEYS:
        return None
    # a deeper path under a known section (e.g. sep.foo.bar) still counts as
    # unknown, but only warn when we can propose something concrete.
    match = difflib.get_close_matches(dotted, KNOWN_KEYS, n=1, cutoff=0.6)
    if match:
        return (f"clave desconocida '{dotted}', "
                f"¿quisiste decir '{match[0]}'?")
    if len(keys) > 1 and keys[0] not in _NESTED_SECTIONS:
        near = difflib.get_close_matches(keys[0], _NESTED_SECTIONS, n=1,
                                         cutoff=0.6)
        hint = f", ¿quisiste decir '{near[0]}'?" if near else ""
        return (f"'{keys[0]}' no es una seccion anidada conocida "
                f"({', '.join(_NESTED_SECTIONS)}){hint}")
    return f"clave desconocida '{dotted}' (se aplica igual)"


def apply_set_overrides(config, assignments, console=None):
    """Apply every ``--set`` in order; returns the "key = value" list.

    Intermediate dicts are created as needed, so ``--set sep.ellipse_k=3``
    works even on a config with no ``sep:`` section at all. Unknown keys are
    warned about (see :func:`key_warning`) but still applied.
    """
    applied = []
    for assignment in assignments or []:
        keys, value = parse_set(assignment)
        if console is not None:
            warn = key_warning(keys)
            if warn:
                console.print(f"[yellow]advertencia:[/] {warn}")
        node = config
        for k in keys[:-1]:
            nxt = node.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                node[k] = nxt
            node = nxt
        node[keys[-1]] = value
        applied.append(f"{'.'.join(keys)} = {value}")
    return applied


def print_key_list(console):
    """--list-keys: every config key photextra knows about, with --set paths."""
    tbl = Table(box=None, padding=(0, 2))
    tbl.add_column("clave (--set ...)", style="bold cyan")
    tbl.add_column("que hace", style="white")
    for key, desc in KNOWN_KEYS.items():
        tbl.add_row(key, desc)
    console.print(Panel(tbl, expand=False, border_style="deep_sky_blue1",
                        title="[bold]Claves de configuracion[/]",
                        subtitle="[dim]uso: --set clave=valor[/]"))
    console.print("[dim]La lista es informativa: --set acepta cualquier "
                  "clave, incluso las que no estan aqui.[/]")


# ---------------------------------------------------------------------------
# --show-config / --save / --edit
# ---------------------------------------------------------------------------

def dump_config(config):
    """Effective config -> YAML text (block style, original key order)."""
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True,
                          default_flow_style=False)


def print_effective_config(console, config):
    """Pretty-print the FULL config that will be handed to Pipeline."""
    body = Text(dump_config(config).rstrip())
    console.print(Panel(body, title="[bold]Config efectivo[/]",
                        subtitle="[dim]YAML + flags + --set[/]",
                        border_style="medium_purple", expand=False))


def save_config(config, path, source_path, force):
    """Write the effective config to ``path``; returns (ok, message).

    Overwriting the config that was just READ is the one dangerous case (it
    would silently bake a throwaway --set into the user's reference file and
    DROP every comment, since yaml.safe_dump cannot keep them), so it needs an
    explicit --force. Any other existing file is also protected by --force.
    """
    same = os.path.abspath(path) == os.path.abspath(source_path or "")
    if os.path.exists(path) and not force:
        why = (" (es el config de entrada; se perderian los comentarios)"
               if same else "")
        return False, (f"Ya existe: {path}{why} — usa --force para "
                       f"sobreescribir")
    header = ("# Config efectivo generado por photextra --save\n"
              f"# base: {source_path}\n"
              "# NOTA: los comentarios del YAML original no se conservan.\n")
    with open(path, "w") as fh:
        fh.write(header + dump_config(config))
    return True, f"Config guardado: {path}"


def default_save_path(config_path):
    """``config_xmask.yaml`` -> ``config_xmask.new.yaml`` (never in place)."""
    stem, ext = os.path.splitext(config_path)
    return f"{stem}.new{ext or '.yaml'}"


def editor_command():
    """$VISUAL / $EDITOR, else the first of nano/vi actually installed."""
    for env in ("VISUAL", "EDITOR"):
        value = os.environ.get(env)
        if value:
            return value
    for cand in ("nano", "vi", "vim"):
        if shutil.which(cand):
            return cand
    return None


def edit_config(path, console):
    """--edit: open the config file itself in the user's editor."""
    editor = editor_command()
    if not editor:
        console.print("[bold red]No encontre un editor[/] (define $EDITOR, "
                      "o instala nano/vi)")
        return 1
    console.print(f"[dim]abriendo[/] {path} [dim]con[/] {editor}")
    try:
        rc = subprocess.call(f'{editor} "{path}"', shell=True)
    except OSError as exc:
        console.print(f"[bold red]No pude abrir el editor:[/] {exc}")
        return 1
    if rc != 0:
        console.print(f"[yellow]El editor salio con codigo {rc}[/]")
    try:  # re-read so an accidental YAML syntax error is caught NOW
        load_config(path)
    except yaml.YAMLError as exc:
        console.print(f"[bold red]YAML invalido tras editar:[/] {exc}")
        return 1
    console.print(f"[bold green]Guardado.[/] Ahora corre: "
                  f"photextra {path} --dry-run")
    return 0

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

WELCOME_MSG = ("Bienvenido a PHOTEXTRA — pipeline de fotometria multibanda "
               "y espectroscopia para morfologia de galaxias")


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
    tbl.add_row("cigale_run", str(config.get("cigale_run", False)))
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

# --- spinners ---------------------------------------------------------------
# "galaxy": the original 4-frame rotating arms around a nucleus. Pure ASCII,
# kept as a fallback/comparison option (--spinner galaxy).
SPINNERS["galaxy"] = {
    "interval": 160,
    "frames": ["-*@*-", "\\*@*/", "|*@*|", "/*@*\\"],
}

# "spiral" (DEFAULT): a two-armed spiral galaxy actually spinning clockwise.
# Built from rich's own 10-frame braille arc (the "dots" spinner), whose 3 lit
# dots sweep clockwise through 10 angular positions around one cell. Here TWO
# copies of that arc flank a bright nucleus, phase-shifted by half a cycle
# (i + 5 of 10 = 180 degrees apart), so the pair rotates rigidly around the
# core: arm up-left/arm down-right -> up/down -> up-right/down-left -> ...
# i.e. a two-arm pinwheel completing a revolution every 10*80 ms = 0.8 s.
# Braille + ASCII only, exactly 3 cells wide and constant width in every
# frame: no emoji, so the progress-bar columns can never misalign (same
# constraint as the original "galaxy" spinner).
_ARC = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # rich's "dots": 3-dot arc sweeping clockwise
SPINNERS["spiral"] = {
    "interval": 80,
    "frames": [_ARC[i] + "*" + _ARC[(i + len(_ARC) // 2) % len(_ARC)]
               for i in range(len(_ARC))],
}

DEFAULT_SPINNER = "spiral"


class TargetTimeColumn(ProgressColumn):
    """Per-TARGET timing: elapsed on the current target + running average.

    Deliberately per target and not per spectral bin — Pipeline.run() is an
    opaque single call from here (see the module docstring's SCOPE NOTE).

    ``_t0`` (perf_counter at the start of the current target) and ``_avg``
    (mean seconds/target over the finished ones) are pushed into the task
    fields by :func:`run_with_progress`. Because rich re-renders on its own
    refresh thread, the "act" number keeps ticking live even while
    ``pipe.run()`` blocks the main thread.
    """

    @staticmethod
    def _fmt(seconds):
        if seconds is None:
            return "--"
        if seconds < 60:
            return f"{seconds:4.1f}s"
        m, s = divmod(int(seconds), 60)
        return f"{m:d}m{s:02d}s"

    def render(self, task):
        t0 = task.fields.get("_t0")
        cur = None if t0 is None else time.perf_counter() - t0
        avg = task.fields.get("_avg")
        return Text.from_markup(
            f"[dim]act[/] [bold cyan]{self._fmt(cur)}[/] "
            f"[dim]prom[/] [bold cyan]{self._fmt(avg)}[/]")


def run_with_progress(pipe, targets, console, spinner=DEFAULT_SPINNER,
                      times=None):
    """Drive pipe.run(target) per target under a rich progress bar.

    Purely a wrapper: each target still goes through the unmodified
    Pipeline.run(). Failures are caught per target so one bad galaxy
    never kills the run (same policy as the driver scripts).

    Timing is measured with ``time.perf_counter()`` AROUND the pipe.run()
    call, i.e. from the outside only. That is why the bar reports per-target
    (per-galaxy) times and not per spectral bin: see the module docstring.

    Returns ``(n_ok, n_fail, failures)``. The return signature is kept at
    three items on purpose (test_cli_smoke.py unpacks exactly three); the
    per-target times are appended to the optional ``times`` list instead, so
    the caller can print an overall average without a breaking change.
    """
    n_ok = n_fail = 0
    failures = []
    times = [] if times is None else times
    progress = Progress(
        SpinnerColumn(spinner_name=spinner, style="bold magenta"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20, complete_style="medium_purple",
                  finished_style="deep_sky_blue1"),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.completed}/{task.total}[/]"),
        TargetTimeColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("procesando", total=len(targets),
                                 _t0=None, _avg=None)
        for tgt in targets:
            tid = str(tgt.get("id"))
            t0 = time.perf_counter()
            progress.update(task, description=f"[bold]{tid}[/]", _t0=t0)
            try:
                pipe.run(tgt)
                n_ok += 1
            except Exception as exc:  # keep going; report at the end
                n_fail += 1
                failures.append((tid, str(exc)))
                progress.console.print(
                    f"  [red]x[/] {tid} failed: {exc}")
            dt = time.perf_counter() - t0
            times.append(dt)
            progress.advance(task)
            progress.update(task, _t0=None,
                            _avg=sum(times) / len(times))
    return n_ok, n_fail, failures


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

_EPILOG = """\
ejemplos:
  photextra --init                            # genera config_xmask.yaml de ejemplo
  photextra --init mi_config.yaml             # con otro nombre
  photextra                                   # usa ./config_xmask.yaml
  photextra config.yaml --targets members.csv --limit 5
  photextra config.yaml --targets members.csv --dry-run
  photextra config.yaml --mode both --cigale --output-dir runs/mkw8
  photextra config.yaml --no-deblend --aperture-mode aperture -q
  photextra --spinner galaxy                  # spinner alternativo

editar la configuracion (3 formas, de menor a mayor alcance):
  photextra config.yaml --show-config          # ver el config efectivo
  photextra --list-keys                        # ver TODAS las claves
  photextra config.yaml --set sep.thresh_sigma=2.0 --show-config
  photextra config.yaml --set surveys=[Legacy_g,Legacy_r] --set mode=both
  photextra config.yaml --edit                 # abrir el YAML en $EDITOR
  photextra config.yaml --set mode=both --save mi_config.yaml

Todas las opciones son opcionales: sin flags el YAML manda. Cada flag
SOBREESCRIBE la clave equivalente del config solo para esta corrida (el
archivo YAML nunca se modifica, salvo con --edit o --save).
"""


def build_parser():
    ap = argparse.ArgumentParser(
        prog="photextra",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=("PHOTEXTRA — multiband photometry + spectroscopy "
                     "pipeline for merging galaxies."),
        epilog=_EPILOG)
    ap.add_argument("config", nargs="?", default=DEFAULT_CONFIG,
                    help=f"YAML config path (default: {DEFAULT_CONFIG})")
    ap.add_argument("-V", "--version", action="version",
                    version=f"photextra {_VERSION}",
                    help="print the photextra version and exit")

    g_t = ap.add_argument_group("targets")
    g_t.add_argument("--targets", default=None, metavar="CSV",
                     help="target list CSV (columns ID, RA, DEC, REDSHIFT, "
                          "optional type); overrides the config's "
                          "targets_csv / targets keys")
    g_t.add_argument("--limit", type=int, default=None, metavar="N",
                     help="only process the first N targets")

    g_r = ap.add_argument_group(
        "run overrides (each one overrides the same key in the YAML config)")
    g_r.add_argument("--mode", choices=MODES, default=None,
                     help="what to run: photometry (imaging only), "
                          "spectroscopy (XpectraFit spectral fit only) or "
                          "both (photometry + spectrum, combined). "
                          "Config key: mode")
    g_r.add_argument("--output-dir", default=None, metavar="DIR",
                     help="write all products under DIR instead of the "
                          "config's output_dir")
    g_r.add_argument("--surveys", default=None, metavar="LIST",
                     help="comma-separated survey/band list to use "
                          "(e.g. Legacy_g,Legacy_r,WISE_W1); overrides the "
                          "config's surveys key")
    g_r.add_argument("--deblend", action=argparse.BooleanOptionalAction,
                     default=None,
                     help="run xdebpair source separation (--no-deblend "
                          "forces the single-aperture fallback, much "
                          "faster). Config key: deblend / use_xdebpair")
    g_r.add_argument("--aperture-mode", choices=APERTURE_MODES, default=None,
                     help="photometry aperture strategy. "
                          "Config key: photometry.aperture_mode")
    g_r.add_argument("--separation", choices=SEPARATIONS, default=None,
                     help="which component of a pair to measure. "
                          "Config key: photometry.separation")
    g_r.add_argument("--cigale", action=argparse.BooleanOptionalAction,
                     default=None,
                     help="run the additional CIGALE SED fit (needs the "
                          "cigale conda env). Config key: cigale_run")
    g_r.add_argument("--cigale-batch", action=argparse.BooleanOptionalAction,
                     default=None,
                     help="skip the per-target CIGALE fit; the driver must "
                          "call run_cigale_batch() afterwards. "
                          "Config key: cigale_batch")

    g_c = ap.add_argument_group(
        "config editing (cualquier clave, sin flags nuevos)")
    g_c.add_argument("--set", dest="sets", action="append", default=None,
                     metavar="CLAVE=VALOR",
                     help="override ANY config key by dotted path, e.g. "
                          "--set mode=both, "
                          "--set photometry.aperture_mode=aperture, "
                          "--set common_grid.pixscale=1.5, "
                          "--set surveys=[Legacy_g,Legacy_r]. The value is "
                          "parsed as YAML (so true/false/numbers/lists get "
                          "their real type). Repeatable; applied in order, "
                          "AFTER the named flags above (so --set wins). "
                          "In-memory only unless you add --save")
    g_c.add_argument("--list-keys", action="store_true",
                     help="print every known config key (with its --set path "
                          "and a one-line description) and exit")
    g_c.add_argument("--show-config", action="store_true",
                     help="print the FULL effective config (YAML + flags + "
                          "--set) as YAML and exit, without running anything "
                          "(--dry-run also shows it)")
    g_c.add_argument("--edit", action="store_true",
                     help="open the config file in $EDITOR (fallback: "
                          "$VISUAL, nano, vi), check its YAML on save, and "
                          "exit. The only way to change many keys at once "
                          "and to keep/write comments")
    g_c.add_argument("--save", nargs="?", const="", default=None,
                     metavar="PATH",
                     help="write the effective config (base YAML + all "
                          "overrides) to PATH and exit. Without PATH it "
                          "writes <config>.new.yaml; it never overwrites an "
                          "existing file (nor the input config, whose "
                          "comments would be lost) unless --force is given")

    g_o = ap.add_argument_group("output / behaviour")
    g_o.add_argument("--dry-run", action="store_true",
                     help="parse the config, resolve the target list and "
                          "print the summary, then exit WITHOUT running "
                          "anything (use it to check a big list before "
                          "committing to a multi-hour run)")
    g_o.add_argument("-q", "--quiet", action="store_true",
                     help="only log errors (quietest useful level)")
    g_o.add_argument("-v", "--verbose", action="count", default=0,
                     help="-v shows INFO logs, -vv shows DEBUG logs "
                          "(default: warnings only, so the bar stays clean)")
    g_o.add_argument("--spinner", default=DEFAULT_SPINNER, metavar="NAME",
                     help=f"progress-bar spinner (default: "
                          f"{DEFAULT_SPINNER}, a spinning two-armed spiral "
                          f"galaxy; 'galaxy' is the older ASCII one, plus "
                          f"any rich built-in such as dots/arc/moon). "
                          f"Use --list-spinners to see them all")
    g_o.add_argument("--list-spinners", action="store_true",
                     help="print the available spinner names and exit")
    g_o.add_argument("--init", nargs="?", const=DEFAULT_CONFIG,
                     default=None, metavar="PATH",
                     help="write a default config template to PATH "
                          f"(default: {DEFAULT_CONFIG}) and exit; refuses "
                          "to overwrite an existing file unless --force is "
                          "also given")
    g_o.add_argument("--force", action="store_true",
                     help="with --init or --save, overwrite PATH if it "
                          "already exists")
    return ap


def apply_overrides(config, args):
    """Write the CLI flags into the loaded config dict (flag wins if given).

    Only touches documented top-level / photometry config keys — the same
    ones a user would edit in the YAML by hand. Returns the list of
    "key = value" strings that were overridden, for display.
    """
    applied = []

    def top(key, value):
        if value is not None:
            config[key] = value
            applied.append(f"{key} = {value}")

    top("mode", args.mode)
    top("output_dir", args.output_dir)
    top("cigale_run", args.cigale)
    top("cigale_batch", args.cigale_batch)
    if args.deblend is not None:
        config["deblend"] = args.deblend
        # legacy alias also honoured by Pipeline; keep both consistent so the
        # older key cannot silently win.
        if "use_xdebpair" in config:
            config["use_xdebpair"] = args.deblend
        applied.append(f"deblend = {args.deblend}")
    if args.surveys:
        config["surveys"] = [s.strip() for s in args.surveys.split(",")
                             if s.strip()]
        applied.append(f"surveys = {', '.join(config['surveys'])}")
    if args.aperture_mode or args.separation:
        phot = dict(config.get("photometry") or {})
        if args.aperture_mode:
            phot["aperture_mode"] = args.aperture_mode
            applied.append(f"photometry.aperture_mode = {args.aperture_mode}")
        if args.separation:
            phot["separation"] = args.separation
            applied.append(f"photometry.separation = {args.separation}")
        config["photometry"] = phot
    return applied


def log_level_from(args):
    """--quiet / -v / -vv -> logging level (default WARNING)."""
    if args.quiet:
        return logging.ERROR
    if args.verbose >= 2:
        return logging.DEBUG
    if args.verbose == 1:
        return logging.INFO
    return logging.WARNING


def print_spinner_list(console):
    console.print("[bold]Spinners disponibles[/] (--spinner NAME):")
    console.print(f"  [bold magenta]{DEFAULT_SPINNER}[/] (por defecto), "
                  f"[bold magenta]galaxy[/], y los de rich:")
    names = sorted(n for n in SPINNERS if n not in ("spiral", "galaxy"))
    console.print("  " + ", ".join(names), highlight=False)


def _fmt_hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def main(argv=None):
    args = build_parser().parse_args(argv)
    console = Console()

    if args.list_spinners:
        print_spinner_list(console)
        return 0

    if args.list_keys:  # pure documentation; no config needed
        print_key_list(console)
        return 0

    if args.init is not None:
        if os.path.exists(args.init) and not args.force:
            console.print(f"[bold red]Ya existe:[/] {args.init} "
                          f"(usa --force para sobreescribir)")
            return 1
        with open(args.init, "w") as fh:
            fh.write(_INIT_TEMPLATE)
        console.print(f"[bold green]Config creado:[/] {args.init}")
        console.print("[dim]Edita targets_csv/targets y output_dir, "
                      "despues corre:[/] "
                      f"photextra {args.init} --targets tus_targets.csv")
        return 0

    # --edit works on the FILE, so it runs before anything is loaded and
    # before the banner (it hands the terminal straight to the editor).
    if args.edit:
        if not os.path.exists(args.config):
            console.print(f"[bold red]Config not found:[/] {args.config} "
                          f"(crealo con: photextra --init {args.config})")
            return 1
        return edit_config(args.config, console)

    print_banner(console)

    if args.spinner not in SPINNERS:
        console.print(f"[bold red]Spinner desconocido:[/] {args.spinner} "
                      f"(prueba --list-spinners)")
        return 1

    if not os.path.exists(args.config):
        console.print(f"[bold red]Config not found:[/] {args.config}")
        return 1
    config = load_config(args.config)
    # named flags first, then --set: the generic layer always wins, so
    # `--mode both --set mode=photometry` ends up on photometry.
    applied = apply_overrides(config, args)
    try:
        applied += apply_set_overrides(config, args.sets, console)
    except ValueError as exc:
        console.print(f"[bold red]--set invalido:[/] {exc}")
        return 1

    # --save: config management, so it writes and EXITS (never starts a
    # multi-hour run as a side effect of asking for a file).
    if args.save is not None:
        path = args.save or default_save_path(args.config)
        ok, msg = save_config(config, path, args.config, args.force)
        console.print(f"[bold {'green' if ok else 'red'}]{msg}[/]")
        return 0 if ok else 1

    if args.show_config:
        print_effective_config(console, config)
        return 0

    try:
        targets = resolve_targets(config, args)
    except (KeyError, ValueError) as exc:
        console.print(f"[bold red]Could not read targets:[/] {exc}")
        return 1

    print_config_summary(console, config, args.config, targets)
    if applied:
        console.print("[dim]overrides CLI:[/] " + "; ".join(applied))

    # dry-run means "show me exactly what would run", so it prints the FULL
    # effective config, not just the summary panel: that is the only way to
    # verify a --set actually landed where it was meant to. Printed BEFORE
    # the empty-targets early return, so `--dry-run` still shows the config
    # while the target list is still being set up.
    if args.dry_run:
        print_effective_config(console, config)

    if not targets:
        console.print(
            "[yellow]No targets to run.[/] Provide them with "
            "[bold]--targets file.csv[/], or add a [bold]targets_csv[/] "
            "path (or an inline [bold]targets:[/] list) to the config.")
        return 0

    if args.dry_run:
        preview = ", ".join(str(t.get("id")) for t in targets[:8])
        if len(targets) > 8:
            preview += f", ... (+{len(targets) - 8})"
        console.print(f"[dim]targets:[/] {preview}")
        console.print(f"[bold cyan]--dry-run:[/] {len(targets)} targets "
                      f"resueltos, nada ejecutado.")
        return 0

    # quiet INFO chatter so the progress bar stays readable; warnings/errors
    # still show. Pipeline's own logging behavior is untouched elsewhere.
    logging.basicConfig(level=log_level_from(args),
                        format="%(levelname)s %(name)s: %(message)s")

    from .pipeline import Pipeline
    pipe = Pipeline(config=config)

    times = []
    t_start = time.perf_counter()
    n_ok, n_fail, failures = run_with_progress(
        pipe, targets, console, spinner=args.spinner, times=times)
    total = time.perf_counter() - t_start

    style = "bold green" if n_fail == 0 else "bold yellow"
    console.print(f"[{style}]Done:[/] {n_ok} ok, {n_fail} failed "
                  f"-> {config.get('output_dir', '')}")
    if times:
        # per-TARGET stats only (per-bin is not observable from here; see the
        # module docstring's SCOPE NOTE).
        avg = sum(times) / len(times)
        console.print(f"[dim]tiempo total[/] {_fmt_hms(total)} · "
                      f"[dim]promedio[/] {avg:.1f}s/galaxia · "
                      f"[dim]min/max[/] {min(times):.1f}s/{max(times):.1f}s")
    for tid, err in failures:
        console.print(f"  [red]{tid}[/]: {err}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
