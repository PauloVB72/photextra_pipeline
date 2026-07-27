#!/usr/bin/env python3
"""Full MKW8 production run: 450 cluster members x 3 sequential stages.

Stages (all over the SAME output_dir, relying on Pipeline's checkpointing):

  1. photometry    — full imaging pipeline per target (skips targets whose
                     {id}_photometry product already exists),
  2. spectroscopy  — DESI/SDSS spectrum resolution + XpectraFit fit (skips
                     targets with a cached {id}_spectral product),
  3. both          — combination only: reuses the cached photometry and
                     spectral products and just computes/writes
                     {id}_combined + plot (recomputing missing pieces).

Each stage runs its targets in parallel (ProcessPoolExecutor, BLAS pinned
to 1 thread per process in photextra_pipeline.__init__). Image downloads
are prefetched at low concurrency BEFORE the parallel photometry stage so
the Legacy Survey cutout API never sees N workers at once.

Bookkeeping (all inside the output dir):
  - run_manifest.jsonl  : append-only, one record per target x stage
                          attempt (reentrant across restarts),
  - run_summary.csv     : one row per target with per-stage status/errors,
  - MKW8_master_table.csv/.ecsv : wide merge of all successful _combined
                          rows,
  - MKW8_run_report.pdf : QC report (counts, failures, histograms).

Usage:
  python3 run_mkw8_full.py                        # full 450-target run
  python3 run_mkw8_full.py --limit 3 --out .../MKW8_full_run_smoketest
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

PIPE_DIR = "/home/polo/Escritorio/PHD/code/photextra_pipeline"
XDEB_DIR = "/home/polo/Escritorio/PHD/code/xdebpair"
for _p in (PIPE_DIR, XDEB_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_TARGETS_CSV = ("/home/polo/Escritorio/PHD/CAVITY/_from_PhDTHESIS/"
                       "GZDESI_PAN174.csv")
DEFAULT_OUT = "/home/polo/Escritorio/PHD/CAVITY/_from_PhDTHESIS/out_pan174"

SURVEYS = ["GALEX_FUV", "GALEX_NUV", "Legacy_g", "Legacy_r", "Legacy_z",
           "WISE_W1", "WISE_W2"]
DOWNLOAD_SIZE = 1
COMMON_GRID = {"pixscale": 1.375, "size": 45, "reference": "WISE_W2"}

STAGES = (
    "photometry",
    "spectroscopy",
    "both",
)

STAGE_WORKERS = {
    "photometry": 8,
    "spectroscopy": 6,
    "both": 10,
}
PREFETCH_WORKERS = 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_targets(path, limit=None):
    targets = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            tgt = {
                "id": str(row["ind"]).strip(),
                "ra": float(row["RAdeg"]),
                "dec": float(row["DEdeg"]),
                "z": float(row["cz"]),
            }
            # optional classification column (merger / pre_merger /
            # post_merger; case-insensitive header) consumed by the
            # photometry.separation policy — absent column = no override
            for key in ("type", "TYPE", "Type"):
                if key in row and str(row[key]).strip():
                    tgt["type"] = str(row[key]).strip()
                    break
            targets.append(tgt)
    if limit:
        targets = targets[:limit]
    return targets


def append_manifest(manifest_path, rec):
    """Append-only JSONL (O_APPEND single write: restart-safe)."""
    rec = dict(rec)
    rec["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(manifest_path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _run_one(stage, tgt, out_dir):
    """Worker: run one target in one stage. NEVER raises."""
    import logging
    import time as _time
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    row = {"stage": stage, "target_id": tgt["id"], "status": "FAILED",
           "seconds": None, "error": ""}
    t0 = _time.perf_counter()
    try:
        from photextra_pipeline.pipeline import Pipeline
        cfg = {
            "surveys": list(SURVEYS),
            "output_dir": out_dir,
            "common_grid": dict(COMMON_GRID),
            "download_size": DOWNLOAD_SIZE,
        }
        pipe = Pipeline(config=cfg, deblend=True, mode=stage)
        res = pipe.run(dict(tgt))
        row["status"] = ("SKIPPED_CACHED" if isinstance(res, dict)
                         and res.get("skipped") else "OK")
    except Exception as exc:  # per-target isolation: batch must survive
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    row["seconds"] = round(_time.perf_counter() - t0, 2)
    return row


def run_stage(stage, targets, out_dir, manifest_path):
    n_workers = STAGE_WORKERS[stage]
    print(f"\n=== stage {stage}: {len(targets)} targets, "
          f"{n_workers} workers ===", flush=True)
    t0 = time.perf_counter()
    counts = {"OK": 0, "FAILED": 0, "SKIPPED_CACHED": 0}
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_run_one, stage, t, out_dir): t for t in targets}
        for fut in as_completed(futs):
            tgt = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:  # worker process died
                row = {"stage": stage, "target_id": tgt["id"],
                       "status": "FAILED", "seconds": None,
                       "error": f"worker crash: {exc}"[:400]}
            append_manifest(manifest_path, row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            done += 1
            if row["status"] == "FAILED" or done % 10 == 0 or \
                    done == len(targets):
                print(f"[{stage} {done}/{len(targets)}] {row['target_id']} "
                      f"{row['status']} {row['seconds']}s {row['error']}",
                      flush=True)
    dt = time.perf_counter() - t0
    print(f"=== stage {stage} done in {dt/60:.1f} min: {counts} ===",
          flush=True)
    return counts


# ---------------------------------------------------------------------------
# final products: summary CSV, master table, PDF report
# ---------------------------------------------------------------------------

def load_manifest_last(manifest_path):
    """{(target_id, stage): record} keeping only the LAST attempt of each."""
    last = {}
    if not os.path.exists(manifest_path):
        return last
    with open(manifest_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            last[(str(rec.get("target_id")), rec.get("stage"))] = rec
    return last


def build_summary(targets, manifest_path, out_csv):
    import pandas as pd
    last = load_manifest_last(manifest_path)
    rows = []
    for t in targets:
        row = {"target_id": t["id"], "ra": t["ra"], "dec": t["dec"],
               "z": t["z"]}
        total_s = 0.0
        for stage in STAGES:
            rec = last.get((t["id"], stage), {})
            row[f"status_{stage}"] = rec.get("status", "NOT_RUN")
            row[f"error_{stage}"] = rec.get("error", "")
            if rec.get("seconds"):
                total_s += float(rec["seconds"])
        row["total_seconds"] = round(total_s, 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"summary -> {out_csv}", flush=True)
    return df


def build_master_table(summary_df, out_dir, out_base):
    """Merge all _combined.csv rows of targets whose 'both' stage succeeded."""
    import pandas as pd
    frames = []
    ok = summary_df[summary_df["status_both"].isin(["OK", "SKIPPED_CACHED"])]
    for tid in ok["target_id"].astype(str):
        p = os.path.join(out_dir, tid, "products", f"{tid}_combined.csv")
        if not os.path.exists(p):
            continue
        try:
            frames.append(pd.read_csv(p))
        except Exception as exc:
            print(f"master table: could not read {p}: {exc}", flush=True)
    if not frames:
        print("master table: no combined products found", flush=True)
        return None
    master = pd.concat(frames, ignore_index=True, sort=False)
    master["target_id"] = master["target_id"].astype(str)
    master.to_csv(out_base + ".csv", index=False)
    try:
        from astropy.table import Table
        Table.from_pandas(master).write(out_base + ".ecsv",
                                        format="ascii.ecsv", overwrite=True)
    except Exception as exc:
        print(f"master table ecsv failed: {exc}", flush=True)
    print(f"master table -> {out_base}.csv ({len(master)} rows)", flush=True)
    return master


def build_pdf_report(summary_df, master, out_pdf):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf) as pdf:
        # --- page 1: per-stage status counts -----------------------------
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        statuses = ["OK", "SKIPPED_CACHED", "FAILED", "NOT_RUN"]
        colors = {"OK": "#4caf50", "SKIPPED_CACHED": "#2196f3",
                  "FAILED": "#f44336", "NOT_RUN": "#9e9e9e"}
        for ax, stage in zip(axes, STAGES):
            counts = summary_df[f"status_{stage}"].value_counts()
            vals = [int(counts.get(s, 0)) for s in statuses]
            ax.bar(range(len(statuses)), vals,
                   color=[colors[s] for s in statuses])
            ax.set_xticks(range(len(statuses)))
            ax.set_xticklabels(statuses, rotation=30, fontsize=7)
            ax.set_title(f"{stage}", fontsize=10)
            for i, v in enumerate(vals):
                if v:
                    ax.text(i, v, str(v), ha="center", va="bottom",
                            fontsize=8)
        fig.suptitle(f"MKW8 full run — {len(summary_df)} targets — "
                     f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- failed-target table pages ------------------------------------
        fail_rows = []
        for _, r in summary_df.iterrows():
            for stage in STAGES:
                if r[f"status_{stage}"] == "FAILED":
                    fail_rows.append((str(r["target_id"]), stage,
                                      str(r[f"error_{stage}"])[:90]))
        per_page = 35
        for i0 in range(0, max(len(fail_rows), 1), per_page):
            chunk = fail_rows[i0:i0 + per_page]
            fig, ax = plt.subplots(figsize=(11, 8))
            ax.axis("off")
            if not fail_rows:
                ax.text(0.5, 0.5, "No failed targets", ha="center",
                        fontsize=14)
            else:
                tbl = ax.table(
                    cellText=chunk or [["", "", ""]],
                    colLabels=["target_id", "stage", "error"],
                    colWidths=[0.18, 0.12, 0.70],
                    loc="upper center", cellLoc="left")
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(6.5)
                tbl.scale(1, 1.25)
                ax.set_title(f"Failed targets "
                             f"({len(fail_rows)} total) — page "
                             f"{i0 // per_page + 1}", fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)
            if not fail_rows:
                break

        # --- master-table histograms --------------------------------------
        if master is not None and len(master):
            def _num(col):
                if col not in master.columns:
                    return np.array([])
                v = np.asarray(master[col], dtype=float)
                return v[np.isfinite(v)]

            specs = [("fAGN", _num("fAGN"), False),
                     ("Dn4000", _num("Dn4000"), False),
                     ("log10 stellar_mass_total",
                      np.log10(_num("stellar_mass_total")[
                          _num("stellar_mass_total") > 0]), False),
                     ("norm_factor", _num("norm_factor"), False),
                     ("norm_dispersion_pct", _num("norm_dispersion_pct"),
                      False),
                     ("norm_factor / norm_factor_single", (lambda a, b: (
                         lambda m: (a[m] / b[m]))(
                         np.isfinite(a) & np.isfinite(b) & (b != 0)))(
                         np.asarray(master.get(
                             "norm_factor", np.nan), dtype=float),
                         np.asarray(master.get(
                             "norm_factor_single", np.nan), dtype=float))
                      if {"norm_factor", "norm_factor_single"} <=
                      set(master.columns) else np.array([]), False)]
            fig, axes = plt.subplots(2, 3, figsize=(11, 7))
            for ax, (name, vals, _) in zip(axes.ravel(), specs):
                if len(vals):
                    ax.hist(vals, bins=30, color="#5b8dbe",
                            edgecolor="black", linewidth=0.4)
                ax.set_title(f"{name} (n={len(vals)})", fontsize=9)
                ax.tick_params(labelsize=7)
            fig.suptitle("Master table distributions")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # class bar charts
            fig, axes = plt.subplots(1, 3, figsize=(11, 4))
            for ax, col in zip(axes, ["spectral_class", "bpt_class",
                                      "whan_class"]):
                if col in master.columns:
                    vc = master[col].astype(str).value_counts()
                    ax.bar(range(len(vc)), vc.values, color="#8a6dbe")
                    ax.set_xticks(range(len(vc)))
                    ax.set_xticklabels(vc.index, rotation=40, fontsize=7,
                                       ha="right")
                    for i, v in enumerate(vc.values):
                        ax.text(i, v, str(v), ha="center", va="bottom",
                                fontsize=7)
                ax.set_title(col, fontsize=10)
            fig.suptitle("Spectral classifications")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"report -> {out_pdf}", flush=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default=DEFAULT_TARGETS_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N targets (smoke tests)")
    ap.add_argument("--stages", nargs="+", default=list(STAGES),
                    choices=list(STAGES))
    args = ap.parse_args()

    targets = load_targets(args.targets, args.limit)
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "run_manifest.jsonl")

    print(f"MKW8 full run: {len(targets)} targets -> {args.out}", flush=True)
    print(f"stages: {args.stages}", flush=True)
    grand_t0 = time.perf_counter()

    for stage in args.stages:
        # prefetch the imaging BEFORE any parallel stage that needs it, at
        # low concurrency (shared external rate limit). No-op for targets
        # whose FITS caches are already complete.
        if stage in ("photometry", "both"):
            from photextra_pipeline.downloader import prefetch_images
            print(f"\n=== prefetch (before {stage}), "
                  f"{PREFETCH_WORKERS} workers ===", flush=True)
            t0 = time.perf_counter()
            pf_failures = prefetch_images(targets, SURVEYS, DOWNLOAD_SIZE,
                                          args.out,
                                          max_workers=PREFETCH_WORKERS)
            print(f"=== prefetch done in "
                  f"{(time.perf_counter() - t0)/60:.1f} min, "
                  f"{len(pf_failures)} target(s) with failed bands ===",
                  flush=True)
        run_stage(stage, targets, args.out, manifest_path)

    # --- final products ----------------------------------------------------
    summary_df = build_summary(targets, manifest_path,
                               os.path.join(args.out, "run_summary.csv"))
    master = build_master_table(summary_df, args.out,
                                os.path.join(args.out, "PAN174_master_table"))
    try:
        build_pdf_report(summary_df, master,
                         os.path.join(args.out, "PAN174_run_report.pdf"))
    except Exception as exc:
        print(f"PDF report failed: {exc}", flush=True)

    grand = time.perf_counter() - grand_t0
    print(f"\n=== ALL DONE in {grand/3600:.2f} h ===", flush=True)


if __name__ == "__main__":
    main()
