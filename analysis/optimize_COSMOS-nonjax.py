# run_cosmos_analysis.py
import os, glob, warnings

import numpy as np
import pandas as pd
import time
from datetime import datetime

from tools_misc import elapsed_time_reporter, grab_matching_format
from tools_fitting import process_one_target_optimize, _warn_and_write_missing

# ---------------------------
# small utilities
# ---------------------------
PIX_SCALE = 0.03  # arcsec / pixel

# IN cosmos_sample*.csv, I can find
# sequentialid, EXPTIME_SCI, EXPTIME_WHT

# IN sep_summary*.csv, I can find
# seqid, orientat, target_label, label (duplicate... lol), x, y, a, b, theta,

# ---------------------------
# driver
# ---------------------------

def run_cosmos_optimization(
    data_dir,
    query_csv_format,
    *,
    m_multipole=(3, 4),
    plot_initial_contour=True,
    plot_final_contour=True,
    optimization_method='SLSQP',
    fit_model=False,
    top_n=5,
    results_csv_name="cosmos_optimization_result.csv",
    start_index=0,
    start_seqid=None,
    target_seqid_list=None,
    skip_on_error=False,
    errors_csv_name="__errors.csv",
    debug=False,
    verbose=False,
    rerun_condition=None, # e.g. "loss_final > 1.0"
    target_loss=1.0,
):
    # 1) find query CSV file
    # ... (lines omitted) ...
    
    # Actually I should use MultiReplace to target signature and the call site separately if they are far apart. 
    # The signature is at line 42+.
    # The call is at line 125+.
    # I will replace signature first.

    # 1) find query CSV file
    query_csv_file = grab_matching_format(data_dir, query_csv_format) #grab the file that matches the given format
    df_query = pd.read_csv(query_csv_file) #

    # find SEP file
    sep_summary_file = grab_matching_format(data_dir, "sep_summary*.csv")
    df_sep = pd.read_csv(sep_summary_file)
    # keys: 'seqid', 'image_width', 'image_height', 'q', 'theta', 'orientat',
    #        'target_label', 'deblend_nthresh', 'deblend_cont',
    #        'detect_thresh_sigma', 'total_flux', 'thresh', 'npix', 'tnpix', 'xmin',
    #        'xmax', 'ymin', 'ymax', 'x', 'y', 'x2', 'y2', 'xy', 'errx2', 'erry2',
    #        'errxy', 'a', 'b', 'cxx', 'cyy', 'cxy', 'cflux', 'flux', 'cpeak',
    #        'peak', 'xcpeak', 'ycpeak', 'xpeak', 'ypeak', 'flag', 'R50', 'A50',
    #        'B50', 'R90', 'A90', 'B90', 'R99', 'A99', 'B99', 'BKG_sigma_clip',
    #        'RMS_sigma_clip', 'label'

    # figure out where to start based on SEP csv file (which might have less than query file, due to manual exceptions)
    if start_seqid is not None:
        seq_col = df_sep["seqid"].astype(str).tolist()
        if str(start_seqid) in seq_col:
            if start_index is not None:
                warnings.warn(f"start_index is given as {start_index}, but start_seqid is given as {start_seqid}. "
                              f"start_index will be ignored.")
            start_index = seq_col.index(str(start_seqid))
        else:
            warnings.warn(f"start_seqid={start_seqid} not found; starting at index={start_index}")
    if target_seqid_list is None or len(target_seqid_list) == 0:
        print("target_seqid_list is not given; all seqid in the SEP CSV file will be used (if you gave start_index of "
              "start_seqid, they will still be used, but the reference will be based on the Query CSV file)")
        target_seqid_list = df_sep['seqid'].astype('int32').to_list()
    # 2) FITS set
    hdf5_paths = glob.glob(os.path.join(data_dir, "*.hdf5")) # list of paths
    # build map by sequentialid filename (basename without .fits)
    hdf5set = {os.path.splitext(os.path.basename(p))[0]: p for p in hdf5_paths}

    # 3) check missing
    csv_ids = [str(int(s)) for s in (df_sep["seqid"]).tolist()]
    missing = {sid for sid in csv_ids if sid+'-cropped' not in hdf5set} # make a set of sid's that doesn't have SCI
    # images
    # missing.update({sid for sid in csv_ids if sid+'-WHT' not in fitset}) # add sid's that doesn't have WHT images
    # print("missing HDF5 files (from them being bad, probably... I haven't handled this in this python file yet.)")
    _warn_and_write_missing(missing, data_dir)

    t0 = time.perf_counter()
    # 4) iterate
    errors = []  # will store dicts: {'sequentialid': ..., 'error': 'type: message'}
    records = []
    total = len(df_query) # total query number
    for idx in range(start_index, total):
        row_query = df_query.iloc[idx]
        sid = int(row_query['sequentialid'])
        if (target_seqid_list != []):
            if int(sid) not in target_seqid_list:
                print(f"sid={sid} not in target_seqid (total len {len(target_seqid_list)}), skipping!")
                continue
            else:
                pass
        if sid in target_seqid_list:
            pass
        else:
            warnings.warn(f"sequential id {sid} not found, skipping!")
            continue
        # index in SEP
        index_sep_target = list(df_sep['seqid']).index(int(row_query['sequentialid']))
        row_sep = df_sep.loc[index_sep_target]
        #
        elapsed_time_reporter(t0, idx, total-start_index, seq_id=sid)
        # msg = f"\r[{idx + 1:>5}/{total:<5}] seq={sid} "
        # print(msg, end='', flush=True)

        # try:
        if (sid in missing) or (not fit_model):
            # missing FITS: record stub and keep going
            rec = dict(sequentialid=sid, l2_initial=np.nan, l2_final=np.nan)
        else:
            # Check for conditional rerun if condition provided
            should_run = True
            if rerun_condition:
                per_path = os.path.join(data_dir, f"{sid}-fit.csv")
                if os.path.exists(per_path):
                    try:
                        df_prev = pd.read_csv(per_path)
                        if not df_prev.empty:
                            # Evaluate condition. 
                            # If condition is TRUE, we rerun. 
                            # If condition is FALSE, we skip (should_run = False).
                            # Example: "loss_final > 1.0"
                            # If row has loss_final=0.5, "0.5 > 1.0" is False -> Skip.
                            # If row has loss_final=2.0, "2.0 > 1.0" is True -> Run.
                            
                            # We use safe evaluation or pandas eval
                            # For single row dataframe, eval returns a Series. We take item().
                            is_met = df_prev.eval(rerun_condition).iloc[0]
                            if not is_met:
                                should_run = False
                                # Load existing record to memory so we don't lose it?
                                # Actually, if we skip, we should probably output the OLD record to the new accumulate list
                                # s.t. the final CSV is complete!
                                rec = df_prev.to_dict(orient='records')[0]
                                print(f"  Skipping seq={sid} (condition '{rerun_condition}' not met)")
                    except Exception as e:
                        print(f"  Warning: failed to evaluate condition for {sid}: {e}")
                        should_run = True # Fail-safe: run it
            
            if should_run:
                rec = process_one_target_optimize(row_query, data_dir, row_sep,
                                                  opt_method=optimization_method, m=m_multipole,
                                                  debug=debug,
                                                  plot_initial_contour=plot_initial_contour,
                                                  plot_final_contour=plot_final_contour,
                                                  verbose=verbose,
                                                  target_loss=target_loss)

        # --- per-sample save so you can resume later ---
        # save "best" parameters + l2s, even if NaNs (skipped/missing)
        per_path = os.path.join(data_dir, f"{sid}-fit.csv")
        pd.DataFrame([rec]).to_csv(per_path, index=False)

        records.append(rec)

    print()  # newline after progress

    # 5) save results CSV
    out_csv = os.path.join(data_dir, results_csv_name)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Save accumulated errors, if any
    if errors:
        err_df = pd.DataFrame(errors)
        err_path = os.path.join(data_dir, errors_csv_name)
        err_df.to_csv(err_path, index=False)
        print(f"Errors: {len(errors)} (saved to {err_path})")

    # 6) Top-N report
    if fit_model:
        key = "loss_final"
    else:
        key = "loss_initial"

    if df.empty:
        # empty df while debugging
        warnings.warn("Emptry Dataframe!")
        pass
    else:
        vals = df[["sequentialid", key]].copy()
        vals = vals[np.isfinite(vals[key])]
        vals = vals.sort_values(key, ascending=False).head(top_n)
        if not vals.empty:
            print(f"Top {min(top_n, len(vals))} worst by {key}:")
            for _, r in vals.iterrows():
                print(f"  seq={r['sequentialid']}  {key}={r[key]:.3g}")
        else:
            print("No finite loss values to rank.")
    print("Done!")

# ---------------------------
# CLI-ish entry
# ---------------------------
# def select_nan_indices(df, key='a_m1_err(m=3)'):
#     ind_nan = np.where(np.isnan(df[key]))[0]
#     return ind_nan

if __name__ == "__main__":

    # -------- default settings you tweak here --------
    DATA_DIR = "../data/HDUL_test4-10"
    query_CSV_file_format = "cosmos_sample_N=*.csv"

    target_seqid_list = [] #select(filename, fieldname, fieldvalue_min, fieldvalule_max) # TODO: Read CSV file and select ones with final loss > 1.0
    optimization_method = 'SLSQP'

    debug=False
    verbose=False

    if verbose:
        print("optimization_method: ", optimization_method)

    fit_model = True
    MODES = [3, 4]
    PLOT_INITIAL_CONTOUR    = True
    PLOT_FINAL_CONTOUR      = True
    TOPN = 5 # top how many to report
    START_INDEX=0
    START_SEQID=None
    RERUN_CONDITION = "loss_final > 1.0" # None to run all; or string condition e.g. "loss_final > 1.0"
    TARGET_LOSS = 1.0 # Set lower (e.g. 0.0) to force exhaustive attempts

    datetime_string_new = str(datetime.now()).replace(' ', '_').replace(':', '')
    datetime_string_new = datetime_string_new[:datetime_string_new.find('.')]
    results_csv_name = "cosmos_optimization_result-at-"+datetime_string_new+".csv"

    # CSV file's path
    # query_CSV_file = grab_matching_format(DATA_DIR, query_CSV_file_format)
    # df = pd.read_csv(query_CSV_file)
    # print("CSV file headers:")
    # print(df.head())  # Prints the first 5 rows of the DataFrame
    # nan_indices = select_nan_indices(df)
    # seqid_list = df['sequentialid'].tolist()[nan_indices]
    # -------------------------------------------------

    run_cosmos_optimization(
        DATA_DIR,
        target_seqid_list=target_seqid_list,
        query_csv_format=query_CSV_file_format,
        optimization_method=optimization_method,
        m_multipole=tuple(MODES),
        fit_model=fit_model,
        plot_initial_contour=PLOT_INITIAL_CONTOUR,
        plot_final_contour=PLOT_FINAL_CONTOUR,
        top_n=TOPN, # how many top L2 to report; default 5
        start_index=START_INDEX,
        start_seqid=START_SEQID,
        skip_on_error=SKIP_ON_ERROR, # skip error if True
        errors_csv_name="__errors.csv",
        debug=debug,
        verbose=verbose,
        results_csv_name=results_csv_name,
        rerun_condition=RERUN_CONDITION,
        target_loss=TARGET_LOSS,
    )

