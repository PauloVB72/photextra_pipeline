#!/usr/bin/env python
"""Standalone benchmark of 3 DESI DR1 spectrum-download methodologies.

Does NOT import or modify any pipeline/xpectrafit source. Self-contained.

Methods:
  A  SPARCL with connect_timeout=30 (batched retrieve + sequential retrieve)
  B  NOIRLab Data Lab SQL (desi_dr1.zpix) -> direct HTTPS coadd, fsspec range read
  C  Direct HTTPS via a *local* zpix catalog (resolution step); size tradeoff noted
"""
from __future__ import annotations
import time
import numpy as np

TARGETS = [
    dict(targetid=2842508660834304, ra=183.21095077874008, dec=-0.836998655550516,  z=0.0350147820346928),
    dict(targetid=2842514688049152, ra=182.46969186422692, dec=-0.5474801880787946, z=0.0506378620747331),
    dict(targetid=2842520698486787, ra=180.62298198086984, dec=-0.2124359381787506, z=0.0203219692822456),
]
CAT_Z = {t["targetid"]: t["z"] for t in TARGETS}


def quality(wave, flux, ivar, z_ret, targetid):
    """Return a dict of basic spectral quality checks."""
    wave = np.asarray(wave, float); flux = np.asarray(flux, float); ivar = np.asarray(ivar, float)
    finite = np.isfinite(flux) & np.isfinite(wave)
    n_fin = int(finite.sum())
    if n_fin == 0:
        return dict(wmin=None, wmax=None, n_finite=0, med_snr=None, z_ok=None)
    snr = flux * np.sqrt(np.clip(ivar, 0, None))
    m = finite & np.isfinite(snr)
    med_snr = float(np.median(snr[m])) if m.any() else None
    z_ok = (z_ret is not None) and (abs(float(z_ret) - CAT_Z[targetid]) < 1e-3)
    return dict(wmin=round(float(wave[finite].min()), 1), wmax=round(float(wave[finite].max()), 1),
                n_finite=n_fin, med_snr=None if med_snr is None else round(med_snr, 3), z_ok=z_ok)


# ---------------------------------------------------------------- METHOD A
def method_A():
    print("\n=== METHOD A: SPARCL (connect_timeout=30) ===")
    from sparcl.client import SparclClient
    include = ["wavelength", "flux", "ivar", "redshift", "targetid"]
    tids = [t["targetid"] for t in TARGETS]

    t0 = time.time(); sc = SparclClient(connect_timeout=30, verbose=False)
    init_s = time.time() - t0
    print(f"client init: {init_s:.2f}s")

    # --- A1: single batched find+retrieve for all 3 ---
    print("\n-- A1 batched --")
    tb = time.time()
    f = sc.find(outfields=["sparcl_id", "targetid", "redshift"],
                constraints={"data_release": ["DESI-DR1"], "targetid": tids}, limit=20)
    id_by_tid = {r["targetid"]: r["sparcl_id"] for r in f.records}
    ids = [id_by_tid[t] for t in tids if t in id_by_tid]
    res = sc.retrieve(uuid_list=ids, include=include)
    batch_total = time.time() - tb
    a1 = {}
    for rec in res.records:
        tid = rec.get("targetid")
        a1[tid] = quality(rec["wavelength"], rec["flux"], rec["ivar"], rec.get("redshift"), tid)
    print(f"A1 batched total (find+retrieve, 3 spectra): {batch_total:.2f}s "
          f"=> {batch_total/len(TARGETS):.2f}s/spectrum")
    for t in tids:
        print(f"  {t}: {a1.get(t)}")

    # --- A2: sequential single-target find+retrieve ---
    print("\n-- A2 sequential --")
    a2 = {}; times = {}
    for t in tids:
        ts = time.time()
        try:
            fi = sc.find(outfields=["sparcl_id", "targetid", "redshift"],
                         constraints={"data_release": ["DESI-DR1"], "targetid": [t]}, limit=2)
            uid = [r["sparcl_id"] for r in fi.records]
            r = sc.retrieve(uuid_list=uid, include=include).records[0]
            a2[t] = quality(r["wavelength"], r["flux"], r["ivar"], r.get("redshift"), t)
            ok = "OK"
        except Exception as e:
            a2[t] = dict(error=repr(e)); ok = "FAIL"
        times[t] = time.time() - ts
        print(f"  {t}: {times[t]:.2f}s [{ok}] {a2.get(t)}")
    print(f"A2 sequential mean: {np.mean(list(times.values())):.2f}s/spectrum")
    return dict(init_s=init_s, A1_total=batch_total, A1=a1, A2=a2, A2_times=times)


# ---------------------------------------------------------------- METHOD B
def method_B():
    print("\n=== METHOD B: Data Lab SQL resolve + direct HTTPS (fsspec range read) ===")
    from dl import queryClient as qc
    from astropy.io import fits

    # 1) resolve targetid -> survey/program/healpix via desi_dr1.zpix
    tr = time.time()
    ids = ",".join(str(t["targetid"]) for t in TARGETS)
    q = ("SELECT targetid, survey, program, healpix, z, zwarn, main_primary "
         f"FROM desi_dr1.zpix WHERE targetid IN ({ids}) AND main_primary")
    csv = qc.query(sql=q, fmt="csv")
    resolve_s = time.time() - tr
    rows = [ln.split(",") for ln in csv.strip().splitlines()[1:]]
    meta = {}
    for r in rows:
        tid = int(r[0])
        meta[tid] = dict(survey=r[1], program=r[2], healpix=int(r[3]), z=float(r[4]))
    print(f"SQL resolve (3 targets, one query): {resolve_s:.2f}s")
    for tid, m in meta.items():
        print(f"  {tid} -> {m}")

    # 2) group by (survey,program,healpix); one coadd file per healpix
    base = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"
    groups = {}
    for tid, m in meta.items():
        groups.setdefault((m["survey"], m["program"], m["healpix"]), []).append(tid)

    out = {}; per_spec_time = {}
    for (survey, program, hp), tids in groups.items():
        url = f"{base}/{survey}/{program}/{hp//100}/{hp}/coadd-{survey}-{program}-{hp}.fits"
        print(f"\n  coadd {survey}/{program}/{hp} ({len(tids)} target(s)): {url}")
        tdl = time.time()
        try:
            # Lazy open over HTTP: fsspec issues range requests; we only read the
            # FIBERMAP + the single matching row of each B/R/Z_FLUX/IVAR array.
            with fits.open(url, use_fsspec=True, fsspec_kwargs={"block_size": 4 * 1024 * 1024}) as h:
                fmap_tid = np.asarray(h["FIBERMAP"].data["TARGETID"])
                names = [hd.name for hd in h]
                wcache = {arm: np.asarray(h[f"{arm}_WAVELENGTH"].data, float)
                          for arm in ("B", "R", "Z") if f"{arm}_WAVELENGTH" in names}
                for tid in tids:
                    ts = time.time()
                    row = int(np.where(fmap_tid == tid)[0][0])
                    ws, fs, iv = [], [], []
                    for arm in ("B", "R", "Z"):
                        if arm not in wcache:
                            continue
                        fl = np.asarray(h[f"{arm}_FLUX"].section[row], float)
                        ii = (np.asarray(h[f"{arm}_IVAR"].section[row], float)
                              if f"{arm}_IVAR" in names else np.ones_like(fl))
                        ws.append(wcache[arm]); fs.append(fl); iv.append(ii)
                    wave = np.concatenate(ws); flux = np.concatenate(fs); ivar = np.concatenate(iv)
                    o = np.argsort(wave)
                    out[tid] = quality(wave[o], flux[o], ivar[o], meta[tid]["z"], tid)
                    per_spec_time[tid] = time.time() - ts
                    print(f"    {tid}: row {row} read {per_spec_time[tid]:.2f}s {out[tid]}")
        except Exception as e:
            print(f"    FAIL: {e!r}")
            for tid in tids:
                out[tid] = dict(error=repr(e))
        dl_s = time.time() - tdl
        print(f"    coadd group wall time (open+extract): {dl_s:.2f}s")
    return dict(resolve_s=resolve_s, B=out, per_spec_time=per_spec_time)


# ---------------------------------------------------------------- METHOD C
def method_C():
    print("\n=== METHOD C: direct HTTPS via LOCAL catalog (size tradeoff) ===")
    import requests
    # The DR1 zpix redshift catalog on disk:
    url = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1/zpix-main-dark.fits"
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
        size = int(r.headers.get("Content-Length", 0))
        print(f"  HEAD {url}\n  status={r.status_code} size={size/1e9:.2f} GB")
    except Exception as e:
        print(f"  HEAD failed: {e!r}")
    print("  -> Resolution identical to Method B, but from a locally downloaded "
          "catalog instead of a live SQL query. Download tradeoff reported; "
          "the coadd-download half is byte-for-byte the same as Method B.")


if __name__ == "__main__":
    T0 = time.time()
    ra = method_A()
    rb = method_B()
    method_C()
    print(f"\nTOTAL benchmark wall time: {time.time()-T0:.1f}s")
