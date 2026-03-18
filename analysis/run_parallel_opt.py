import os
import glob
import pandas as pd
import numpy as np
from tools_fitting import process_one_target_optimize
import multiprocessing
import h5py
import time
from astropy.io import fits
import sys
from tqdm.contrib.concurrent import process_map  # from tqdm

def make_dummy_rows_fast(sci_header, seq_id):
    truth_n = float(sci_header.get('n_sersic', 4.0))
    truth_R = float(sci_header.get('R_sersic', 0.4))
    truth_amp = float(sci_header.get('amplitude', 0.05))
    truth_q = float(sci_header.get('q', 0.8))
    truth_theta = float(sci_header.get('theta_ell', 0.0))
    row_query = pd.Series({'sequentialid': seq_id, 'EXPTIME_SCI': float(sci_header.get('EXPTIME', 2000)), 'EXPTIME_WHT': 0, 'sersic_n_gim2d': truth_n, 'r50': truth_R})
    ny, nx = sci_header['NAXIS2'], sci_header['NAXIS1']
    row_sep = pd.Series({'seqid': seq_id, 'image_width': nx, 'image_height': ny, 'q': truth_q, 'theta': truth_theta, 'x': 0, 'y': 0, 'xcpeak': nx/2 - 0.5, 'ycpeak': ny/2 - 0.5, 'R50': truth_R, 'flux': truth_amp * 100})
    return row_query, row_sep

def worker(task):
    target_d, base, truth_row = task
    f_sci = os.path.join(target_d, f"{base}-SCI.fits")
    try:
        with fits.open(f_sci) as hdu:
            hdr = hdu[0].header
        row_query, row_sep = make_dummy_rows_fast(hdr, base)
        
        hdf5_fn = os.path.join(target_d, f"{base}-cropped.hdf5")
        if not os.path.exists(hdf5_fn):
             hdf5_fn_alt = os.path.join(target_d, f"{base}.hdf5")
             if os.path.exists(hdf5_fn_alt): hdf5_fn = hdf5_fn_alt
             else:
                 return None
        
        with h5py.File(hdf5_fn, "r") as f:
            sci_crop = f["sci_bgsub_crop"][()]
            wht_crop = f["wht_crop"][()]
            seg_crop = f["segmap_crop"][()]
            msk_crop = f["mask_crop"][()]
            psf_crop = np.ones((5,5))
            
        t0 = time.time()
        rec_fit = process_one_target_optimize(
            row_query=row_query, data_dir=target_d, row_sep=row_sep,
            sci=sci_crop, wht=wht_crop, psf=psf_crop, mask=msk_crop, segmap=seg_crop,
            initial_guess=None, plot_name=os.path.join(target_d, f"{base}_fit"),
            plot_final_contour=False, supersample_factor=3, truth_row=truth_row, target_loss=1.2
        )
        rec_fit['fit_time'] = time.time() - t0
        rec_fit['id'] = int(base)
        rec_fit['filename'] = base
        return rec_fit
    except Exception as e:
        print(f"Failed {base}: {e}")
        return None

if __name__ == "__main__":
    target_d = "../data/mock_grid_validation/mock_varying_all"
    out_path = os.path.join(target_d, "fitting_results.csv")
    
    # Pre-process all remaining if missing
    # But wait, did mock_run_fitting run preprocess on all 110?
    # Yes, preprocess_directory preprocesses ALL files first. I saw the logs generating up to 109.
    # We just need to load truth_dict
    truth_dict = {}
    df_truth = pd.read_csv(os.path.join(target_d, "simulation_truth.csv"))
    for idx, row in df_truth.iterrows():
        truth_dict[int(row['seqid'])] = row.to_dict()
        
    # Read existing done
    done_ids = []
    if os.path.exists(out_path):
        df_done = pd.read_csv(out_path)
        done_ids = df_done['id'].astype(int).tolist()
    
    fits_files = glob.glob(os.path.join(target_d, "*-SCI.fits"))
    tasks = []
    for f in fits_files:
        base = os.path.basename(f).replace("-SCI.fits", "")
        seqpixel = int(base)
        if seqpixel not in done_ids:
            tasks.append((target_d, base, truth_dict.get(seqpixel, None)))
            
    print(f"Found {len(tasks)} tasks remaining. Processing with {multiprocessing.cpu_count()} cores.")
    
    # Silence matplotlib and user prints inside worker
    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True
    
    # Process
    results = process_map(worker, tasks, max_workers=multiprocessing.cpu_count(), chunksize=1)
    
    # Filter valid
    results = [r for r in results if r is not None]
    
    if len(results) > 0:
        df_new = pd.DataFrame(results)
        if os.path.exists(out_path):
            df_new.to_csv(out_path, mode='a', header=False, index=False)
        else:
            df_new.to_csv(out_path, mode='w', header=True, index=False)
        print(f"Saved {len(results)} new results.")
    else:
        print("No new results to save.")
