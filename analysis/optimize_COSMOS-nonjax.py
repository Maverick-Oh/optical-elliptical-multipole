#!/usr/bin/env python3
"""
optimize_COSMOS-nonjax.py — Fit elliptical-multipole Sérsic profiles to COSMOS targets.

Pipeline order:
  1) load_COSMOS.py           → downloads FITS cutouts
  2) preprocess_COSMOS_w_source_extractor.py → SEP preprocessing → {seqid}-cropped.hdf5
  3) THIS SCRIPT              → optimisation → cosmos_optimization_result.csv

Features:
  • Sorted target selection with index slicing (--sorting-label, --select-ind-ini/fin)
  • Supersampled model evaluation (--supersample, default 3)
  • Multi-strategy optimisation (SLSQP → L-BFGS-B, early exit at --target-loss)
  • Incremental + cumulative CSV saving
  • Per-fit timing
  • Parallel fitting via ProcessPoolExecutor (--n-workers)
  Usage: python optimize_COSMOS-nonjax.py --data-dir ../data/HDUL_test7-big100 --n-workers 5 --skip-existing True --select-ind-fin 100
  python optimize_COSMOS-nonjax.py --data-dir ../data/HDUL_test7-big100 --n-workers 5 --skip-existing --select-ind-fin 100
"""

import os, sys, glob, warnings, time, argparse
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------
# project imports
# ---------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools_misc import elapsed_time_reporter, grab_matching_format
from tools_fitting import process_one_target_optimize, _warn_and_write_missing

PIX_SCALE = 0.03  # arcsec / pixel

# ============================================================
#  Single-target worker (top-level for pickle-ability)
# ============================================================

def fit_one_target(
    sid, data_dir, row_query_dict, row_sep_dict,
    m_multipole, supersample, target_loss, opt_method,
    plot_initial, plot_final, verbose, use_jax,
):
    """
    Fit a single target. Returns a dict (record).
    Designed to be called from ProcessPoolExecutor.
    """
    import pandas as pd  # reimport inside worker
    import time as _time

    # Reconstruct pandas Series from dicts
    row_query = pd.Series(row_query_dict)
    row_sep = pd.Series(row_sep_dict)

    t0 = _time.perf_counter()
    try:
        rec = process_one_target_optimize(
            row_query, data_dir, row_sep,
            opt_method=opt_method,
            m=list(m_multipole),
            plot_initial_contour=plot_initial,
            plot_final_contour=plot_final,
            verbose=verbose,
            target_loss=target_loss,
            supersample_factor=supersample,
            truth_row=None,
            use_jax=use_jax,
        )
        rec['fit_time'] = _time.perf_counter() - t0
    except Exception as e:
        import traceback
        traceback.print_exc()
        rec = dict(
            sequentialid=int(sid),
            loss_initial=np.nan, loss_final=np.nan,
            error=f"{type(e).__name__}: {e}",
            fit_time=_time.perf_counter() - t0,
        )
    return rec


# ============================================================
#  CSV merge helper
# ============================================================

def upsert_result_csv(out_csv, new_rec):
    """Append or overwrite a row in the cumulative results CSV by sequentialid."""
    df_new = pd.DataFrame([new_rec])
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        sid_col = 'sequentialid'
        if sid_col in df_old.columns and sid_col in df_new.columns:
            # Remove old row for this sid, append new
            mask = df_old[sid_col].astype(str) != str(new_rec.get(sid_col, ''))
            df_merged = pd.concat([df_old[mask], df_new], ignore_index=True)
        else:
            df_merged = pd.concat([df_old, df_new], ignore_index=True)
        df_merged.to_csv(out_csv, index=False)
    else:
        df_new.to_csv(out_csv, index=False)


# ============================================================
#  Main driver
# ============================================================

def run_cosmos_optimization(args):
    data_dir = args.data_dir

    # ── 1) Load catalogue CSV ──
    query_csv = grab_matching_format(data_dir, args.query_csv_format)
    df_query = pd.read_csv(query_csv)

    # ── 2) Load SEP summary CSV ──
    sep_csv = grab_matching_format(data_dir, "sep_summary*.csv")
    df_sep = pd.read_csv(sep_csv)

    # ── 3) Merge for sorting ──
    # Ensure column types match for merge
    df_query['sequentialid'] = df_query['sequentialid'].astype(int)
    df_sep['seqid'] = df_sep['seqid'].astype(int)

    df_merged = df_sep.merge(
        df_query, left_on='seqid', right_on='sequentialid', how='inner'
    )

    # ── 4) Sort ──
    if args.sorting_label:
        if args.sorting_label not in df_merged.columns:
            available = sorted(df_merged.columns.tolist())
            raise ValueError(
                f"sorting_label '{args.sorting_label}' not found. "
                f"Available columns: {available}"
            )
        ascending = (args.sorting_order == 'increase')
        df_merged.sort_values(args.sorting_label, ascending=ascending, inplace=True)
        df_merged.reset_index(drop=True, inplace=True)
        print(f"Sorted by '{args.sorting_label}' ({args.sorting_order})")

    # ── 5) Slice by index range ──
    n_total = len(df_merged)
    ini = args.select_ind_ini
    fin = args.select_ind_fin if args.select_ind_fin is not None else n_total
    fin = min(fin, n_total)
    df_to_fit = df_merged.iloc[ini:fin].copy()
    print(f"Selected indices [{ini}, {fin}) → {len(df_to_fit)} targets out of {n_total}")

    # ── 6) Check HDF5 availability ──
    hdf5_available = set()
    for p in glob.glob(os.path.join(data_dir, "*-cropped.hdf5")):
        base = os.path.splitext(os.path.basename(p))[0].replace("-cropped", "")
        hdf5_available.add(base)

    missing = []
    target_rows = []
    for _, row in df_to_fit.iterrows():
        sid = str(int(row['seqid']))
        if sid not in hdf5_available:
            missing.append(sid)
        else:
            target_rows.append(row)
    if missing:
        _warn_and_write_missing(set(missing), data_dir)
        print(f"  Skipping {len(missing)} targets with missing HDF5 files")

    print(f"  {len(target_rows)} targets ready to fit")
    if len(target_rows) == 0:
        print("Nothing to do.")
        return

    # ── 7) Prepare output CSV path ──
    out_csv = os.path.join(data_dir, args.results_csv_name)

    # ── 8) Skip logic (check for already-fitted targets) ──
    if args.skip_existing and os.path.exists(out_csv):
        df_existing = pd.read_csv(out_csv)
        existing_sids = set(df_existing['sequentialid'].astype(str).tolist())
    else:
        existing_sids = set()

    # ── 9) Build fitting tasks ──
    tasks = []
    for row in target_rows:
        sid = int(row['seqid'])
        if str(sid) in existing_sids:
            continue  # already fitted, skip unless overwrite
        # Build row_query and row_sep dicts for this target
        row_query = df_query[df_query['sequentialid'] == sid].iloc[0]
        row_sep_match = df_sep[df_sep['seqid'] == sid].iloc[0]
        tasks.append((sid, row_query.to_dict(), row_sep_match.to_dict()))

    n_skipped = len(target_rows) - len(tasks)
    if n_skipped > 0:
        print(f"  Skipped {n_skipped} already-fitted targets (use --overwrite-range to re-fit)")
    print(f"  {len(tasks)} targets to fit")

    if len(tasks) == 0:
        print("All targets already fitted. Nothing to do.")
        return

    # ── 10) Run fitting ──
    m_multipole = tuple(args.m_multipole)
    t_global = time.perf_counter()

    if args.n_workers <= 1:
        # Sequential
        for i, (sid, rq, rs) in enumerate(tasks):
            elapsed_time_reporter(t_global, i, len(tasks), seq_id=sid)
            rec = fit_one_target(
                sid, data_dir, rq, rs,
                m_multipole, args.supersample, args.target_loss,
                args.opt_method, args.plot_initial, args.plot_final,
                args.verbose, args.use_jax,
            )
            # Incremental save
            upsert_result_csv(out_csv, rec)
            # Per-sample save
            per_csv = os.path.join(data_dir, f"{sid}-fit.csv")
            pd.DataFrame([rec]).to_csv(per_csv, index=False)
            print(f"  [{i+1}/{len(tasks)}] sid={sid}  loss={rec.get('loss_final', np.nan):.4f}  "
                  f"time={rec.get('fit_time', 0):.1f}s")
    else:
        # Parallel
        print(f"  Running with {args.n_workers} parallel workers")
        from tqdm import tqdm
        records_done = []
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            future_to_sid = {}
            for sid, rq, rs in tasks:
                fut = pool.submit(
                    fit_one_target,
                    sid, data_dir, rq, rs,
                    m_multipole, args.supersample, args.target_loss,
                    args.opt_method, args.plot_initial, args.plot_final,
                    args.verbose, args.use_jax,
                )
                future_to_sid[fut] = sid

            for fut in tqdm(as_completed(future_to_sid), total=len(future_to_sid),
                            desc="Fitting", unit="target"):
                sid = future_to_sid[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = dict(sequentialid=int(sid), error=str(e),
                               loss_initial=np.nan, loss_final=np.nan, fit_time=np.nan)
                # Save incrementally
                upsert_result_csv(out_csv, rec)
                per_csv = os.path.join(data_dir, f"{sid}-fit.csv")
                pd.DataFrame([rec]).to_csv(per_csv, index=False)
                records_done.append(rec)

    # ── 11) Final report ──
    elapsed_total = time.perf_counter() - t_global
    print(f"\n{'='*50}")
    print(f"  Fitting complete in {elapsed_total:.1f}s")
    if os.path.exists(out_csv):
        df_out = pd.read_csv(out_csv)
        n_fitted = len(df_out)
        if 'loss_final' in df_out.columns:
            finite = df_out[np.isfinite(df_out['loss_final'])]
            if len(finite) > 0:
                print(f"  Total fitted: {n_fitted}")
                print(f"  Mean loss: {finite['loss_final'].mean():.4f}")
                print(f"  Median loss: {finite['loss_final'].median():.4f}")
                worst = finite.nlargest(min(5, len(finite)), 'loss_final')
                print(f"  Top-{len(worst)} worst:")
                for _, r in worst.iterrows():
                    t_str = f"  {r.get('fit_time', 0):.1f}s" if 'fit_time' in r else ""
                    print(f"    seq={r['sequentialid']}  loss={r['loss_final']:.4f}{t_str}")
        if 'fit_time' in df_out.columns:
            times = df_out['fit_time'].dropna()
            if len(times) > 0:
                print(f"  Avg fit time: {times.mean():.1f}s  |  Total wall: {elapsed_total:.1f}s")
    print(f"{'='*50}")
    print(f"  Results: {out_csv}")
    print("Done!")


# ============================================================
#  CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit elliptical-multipole Sérsic profiles to COSMOS targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ──
    parser.add_argument("--data-dir", type=str, default="../data/HDUL_test4-10",
                        help="Directory containing preprocessed HDF5 files and CSVs")
    parser.add_argument("--query-csv-format", type=str, default="cosmos_sample_N=*.csv",
                        help="Glob pattern for the query catalogue CSV")
    parser.add_argument("--results-csv-name", type=str,
                        default="cosmos_optimization_result.csv",
                        help="Name of the cumulative results CSV file")

    # ── Target selection ──
    parser.add_argument("--sorting-label", type=str, default="sequentialid",
                        help="Column to sort targets by (e.g. r50, sequentialid, sersic_n_gim2d)")
    parser.add_argument("--sorting-order", type=str, default="increase",
                        choices=["increase", "decrease"],
                        help="Sort order: 'increase' (ascending) or 'decrease' (descending)")
    parser.add_argument("--select-ind-ini", type=int, default=0,
                        help="Start index (inclusive) after sorting")
    parser.add_argument("--select-ind-fin", type=int, default=10,
                        help="End index (exclusive) after sorting; use a large number for all")
    parser.add_argument("--skip-existing", action="store_true", default=False,
                        help="Skip targets already in the results CSV (default: re-fit all in range)")

    # ── Fitting parameters ──
    parser.add_argument("--supersample", type=int, default=3,
                        help="Supersampling factor for model evaluation")
    parser.add_argument("--target-loss", type=float, default=1.2,
                        help="Reduced χ² threshold: stop optimisation when loss ≤ this value")
    parser.add_argument("--opt-method", type=str, default="SLSQP",
                        help="Primary optimisation method (SLSQP or L-BFGS-B)")
    parser.add_argument("--m-multipole", type=int, nargs="+", default=[3, 4],
                        help="Multipole orders to fit")

    # ── Plotting ──
    parser.add_argument("--plot-initial", action="store_true", default=True,
                        help="Generate initial contour plots")
    parser.add_argument("--plot-final", action="store_true", default=True,
                        help="Generate final comparison plots")

    # ── Parallelism ──
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel workers (1 = sequential)")

    # ── JAX acceleration ──
    parser.add_argument("--no-jax", dest="use_jax", action="store_false", default=True,
                        help="Disable JAX JIT loss (use numpy loss instead)")

    # ── Debug ──
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()
    run_cosmos_optimization(args)
