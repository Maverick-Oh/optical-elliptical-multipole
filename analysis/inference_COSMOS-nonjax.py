# run_cosmos_mcmc.py
import os
import glob
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd

from tools_misc import elapsed_time_reporter, grab_matching_format
from tools_fitting import process_one_target_mcmc, _warn_and_write_missing

PIX_SCALE = 0.03  # arcsec / pixel


def _infer_m_from_opt_columns(df_opt):
    """
    Infer which multipole orders are present in the optimization CSV
    by looking for columns like 'a_m3_best'.
    """
    m_set = set()
    for col in df_opt.columns:
        if col.startswith("a_m") and col.endswith("_best"):
            # extract integer between 'a_m' and '_best'
            middle = col[len("a_m"):-len("_best")]
            try:
                mm = int(middle)
                m_set.add(mm)
            except ValueError:
                continue
    return sorted(m_set)


def run_cosmos_mcmc(
    data_dir,
    query_csv_format,
    *,
    m_multipole=(3, 4),
    fit_params=None,
    fix_params=None,
    use_analytic_amplitude=True,
    results_csv_name="cosmos_mcmc_result.csv",
    start_index=None,
    start_seqid=None,
    target_seqid_list=None,
    skip_on_error=False,
    errors_csv_name="__errors_mcmc.csv",
    mcmc_config=None,
    continue_mcmc=False,
    restart_and_overwrite_mcmc=False,
    debug=False,
):
    """
    High-level driver for running MCMC on COSMOS cutouts.

    Parameters largely mirror run_cosmos_optimization in optimize_COSMOS-nonjax.py.
    """
    if mcmc_config is None:
        mcmc_config = {}

    # 1) Query CSV
    query_csv_file = grab_matching_format(data_dir, query_csv_format)
    df_query = pd.read_csv(query_csv_file)

    # 1b) SEP CSV (must be unique)
    sep_summary_file_list = glob.glob(os.path.join(data_dir, "sep_summary-N=*.csv"))
    if len(sep_summary_file_list) == 1:
        sep_summary_file = sep_summary_file_list[0]
    else:
        raise ValueError(
            f"len(sep_summary_file_list)={len(sep_summary_file_list)}! "
            "It is expected to be 1."
        )
    df_sep = pd.read_csv(sep_summary_file)

    # 1c) Optimization results CSV
    opt_csv_file = grab_matching_format(
        data_dir, "cosmos_optimization_result-*.csv"
    )
    df_opt = pd.read_csv(opt_csv_file)

    # ---- multipole consistency checks ----
    m_multipole = tuple(int(mm) for mm in m_multipole)
    m_user = sorted(m_multipole)
    m_opt = _infer_m_from_opt_columns(df_opt)

    extra_in_opt = [mm for mm in m_opt if mm not in m_user]
    missing_from_opt = [mm for mm in m_user if mm not in m_opt]

    if extra_in_opt:
        print(
            "WARNING: Optimization results include multipoles "
            f"{extra_in_opt} that you are NOT including in MCMC (m_multipole={m_user})."
        )
        ans = input("Proceed and ignore these extra modes? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborting MCMC run.")
            return

    if missing_from_opt:
        print(
            "WARNING: You requested multipoles "
            f"{missing_from_opt} in MCMC but they are NOT present in the optimization CSV."
        )
        print("They will be initialized with a_mX=0 and default uncertainties.")
        ans = input("Proceed with these additional modes? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborting MCMC run.")
            return

    # ---- figure out where to start based on SEP CSV ----
    if start_seqid is not None:
        seq_col = df_sep["seqid"].astype(str).tolist()
        if str(start_seqid) in seq_col:
            if start_index is not None:
                warnings.warn(
                    f"start_index is given as {start_index}, but start_seqid is "
                    f"given as {start_seqid}. start_index will be ignored."
                )
            start_index = seq_col.index(str(start_seqid))
        else:
            warnings.warn(
                f"start_seqid={start_seqid} not found; starting at index={start_index}"
            )

    if start_index is None:
        start_index = 0

    # If target_seqid_list not given, use all seqid in SEP CSV
    if target_seqid_list is None or len(target_seqid_list) == 0:
        print(
            "target_seqid_list is not given; all seqid in the SEP CSV file will "
            "be used (if you gave start_index or start_seqid, they still apply, "
            "but the reference is based on the Query CSV file)."
        )
        target_seqid_list = df_sep["seqid"].astype("int32").to_list()
    else:
        target_seqid_list = [int(s) for s in target_seqid_list]

    # 2) HDF5 set
    hdf5_paths = glob.glob(os.path.join(data_dir, "*.hdf5"))
    hdf5set = {os.path.splitext(os.path.basename(p))[0]: p for p in hdf5_paths}

    # 3) check missing HDF5
    csv_ids = [str(int(s)) for s in df_sep["seqid"].tolist()]
    missing = {sid for sid in csv_ids if sid + "-cropped" not in hdf5set}
    _warn_and_write_missing(missing, data_dir)

    # 4) iterate over query rows
    errors = []
    records = []
    total = len(df_query)
    t0 = time.perf_counter()

    for idx in range(start_index, total):
        row_query = df_query.iloc[idx]
        sid = int(row_query["sequentialid"])

        # target list filtering
        if target_seqid_list != []:
            if sid not in target_seqid_list:
                print(
                    f"sid={sid} not in target_seqid (total len {len(target_seqid_list)}), skipping!"
                )
                continue

        if sid in target_seqid_list:
            pass
        else:
            warnings.warn(f"sequential id {sid} not found, skipping!")
            continue

        elapsed_time_reporter(
            t0, idx - start_index, total - start_index, seq_id=sid
        )

        sid_str = str(sid)
        # skip if missing HDF5
        if sid_str in missing:
            rec = {"sequentialid": sid}
            records.append(rec)
            continue

        # find SEP row
        try:
            # assumes one row per seqid
            row_sep = df_sep.loc[df_sep["seqid"] == sid].iloc[0]
        except IndexError:
            msg = f"seqid={sid} not found in SEP CSV; skipping."
            warnings.warn(msg)
            errors.append({"sequentialid": sid, "error": msg})
            if skip_on_error:
                continue
            else:
                raise

        # find optimization row
        try:
            opt_row = df_opt.loc[df_opt["sequentialid"] == sid].iloc[0]
        except IndexError:
            msg = f"sequentialid={sid} not found in optimization CSV; skipping."
            warnings.warn(msg)
            errors.append({"sequentialid": sid, "error": msg})
            if skip_on_error:
                continue
            else:
                raise

        # run per-target MCMC
        try:
            rec = process_one_target_mcmc(
                row_query=row_query,
                data_dir=data_dir,
                row_sep=row_sep,
                opt_row=opt_row,
                m=m_multipole,
                PIX_SCALE=PIX_SCALE,
                fit_params=fit_params,
                fix_params=fix_params,
                use_analytic_amplitude=use_analytic_amplitude,
                mcmc_config=mcmc_config,
                continue_mcmc=continue_mcmc,
                restart_and_overwrite_mcmc=restart_and_overwrite_mcmc,
                debug=debug,
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            warnings.warn(
                f"Error for seqid={sid} at index {idx}: {msg}"
            )
            errors.append({"sequentialid": sid, "error": msg})
            if skip_on_error:
                continue
            else:
                raise

        # per-sample save
        per_path = os.path.join(data_dir, f"{sid}-mcmc.csv")
        pd.DataFrame([rec]).to_csv(per_path, index=False)

        records.append(rec)

    print()  # newline after progress

    # 5) save aggregate results CSV
    out_csv = os.path.join(data_dir, results_csv_name)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved MCMC summary: {out_csv}")

    # save accumulated errors, if any
    if errors:
        err_df = pd.DataFrame.from_records(errors)
        err_path = os.path.join(data_dir, errors_csv_name)
        err_df.to_csv(err_path, index=False)
        print(f"Errors: {len(errors)} (saved to {err_path})")


def combine_mcmc_results(
    data_dir,
    mcmc_per_target_pattern="*-mcmc.csv",
    out_csv_name=None,
    query_csv_format="cosmos_sample_N=*.csv",
    opt_csv_format="cosmos_optimization_result-*.csv",
):
    """
    Combine per-target MCMC summary CSVs into a single table, and merge in
    metadata from query / SEP / optimization CSVs.
    """
    # gather per-target MCMC CSVs
    mcmc_files = glob.glob(os.path.join(data_dir, mcmc_per_target_pattern))
    if not mcmc_files:
        print("No per-target MCMC CSV files found; nothing to combine.")
        return

    df_list = [pd.read_csv(f) for f in mcmc_files]
    df_mcmc = pd.concat(df_list, ignore_index=True)

    # query CSV
    query_csv_file = grab_matching_format(data_dir, query_csv_format)
    df_query = pd.read_csv(query_csv_file)

    # SEP CSV
    sep_summary_file_list = glob.glob(os.path.join(data_dir, "sep_summary-N=*.csv"))
    if len(sep_summary_file_list) != 1:
        raise ValueError(
            f"Expected exactly one SEP CSV, found {len(sep_summary_file_list)}."
        )
    sep_summary_file = sep_summary_file_list[0]
    df_sep = pd.read_csv(sep_summary_file)

    # optimization CSV
    opt_csv_file = grab_matching_format(data_dir, opt_csv_format)
    df_opt = pd.read_csv(opt_csv_file)

    # Merge metadata
    df_comb = df_mcmc.merge(
        df_query, on="sequentialid", how="left", suffixes=("", "_query")
    )
    df_comb = df_comb.merge(
        df_sep,
        left_on="sequentialid",
        right_on="seqid",
        how="left",
        suffixes=("", "_sep"),
    )
    df_comb = df_comb.merge(
        df_opt, on="sequentialid", how="left", suffixes=("", "_opt")
    )

    if out_csv_name is None:
        datetime_string = str(datetime.now()).replace(" ", "_").replace(":", "")
        datetime_string = datetime_string[: datetime_string.find(".")]
        out_csv_name = f"cosmos_mcmc_result-combined-at-{datetime_string}.csv"

    out_path = os.path.join(data_dir, out_csv_name)
    df_comb.to_csv(out_path, index=False)
    print(f"Combined MCMC results saved to: {out_path}")


if __name__ == "__main__":
    # -------- default settings to tweak here --------
    DATA_DIR = "../data/HDUL_test4-10"
    QUERY_CSV_FORMAT = "cosmos_sample_N=*.csv"

    target_seqid_list = []  # e.g., [131525, ...] or [] for "all"
    fit_params = None       # None = all (except amplitude if use_analytic_amplitude=True)
    fix_params = None       # e.g., {"background": 0.0}

    USE_ANALYTIC_AMPLITUDE = True

    MCMC_CONFIG = {
        "n_walkers": 32,
        "n_steps": 2000,
        "burnin_fraction": 0.3,
        "init_scale": 1.0,
        "random_seed": 1234,
    }

    START_INDEX = None
    START_SEQID = None

    CONTINUE_MCMC = False
    RESTART_AND_OVERWRITE_MCMC = False
    SKIP_ON_ERROR = False

    datetime_string_new = str(datetime.now()).replace(" ", "_").replace(":", "")
    datetime_string_new = datetime_string_new[: datetime_string_new.find(".")]
    results_csv_name = "cosmos_mcmc_result-at-" + datetime_string_new + ".csv"

    run_cosmos_mcmc(
        DATA_DIR,
        QUERY_CSV_FORMAT,
        m_multipole=(3, 4),
        fit_params=fit_params,
        fix_params=fix_params,
        use_analytic_amplitude=USE_ANALYTIC_AMPLITUDE,
        results_csv_name=results_csv_name,
        start_index=START_INDEX,
        start_seqid=START_SEQID,
        target_seqid_list=target_seqid_list,
        skip_on_error=SKIP_ON_ERROR,
        errors_csv_name="__errors_mcmc.csv",
        mcmc_config=MCMC_CONFIG,
        continue_mcmc=CONTINUE_MCMC,
        restart_and_overwrite_mcmc=RESTART_AND_OVERWRITE_MCMC,
        debug=True,
    )
