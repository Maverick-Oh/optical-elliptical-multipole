import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import warnings
import time

# Import shared tools
from tools_fitting import process_one_target_optimize, unpack_params

# Configuration
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR_BASE = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILENAME = "fitting_results.csv"

# Mock Rows needs to provide enough info for process_one_target_optimize to start
def make_dummy_rows(sci_header, seq_id):
    # We need to mimic the columns expected by process_one_target_optimize
    # The code uses:
    # row_query: ['EXPTIME_SCI', 'EXPTIME_WHT'] and optionally 'r50', 'sersic_n_gim2d' for initial guess if not in row_sep?
    # row_sep: ['seqid', 'image_width', 'image_height', 'x', 'y' (cutout coords?), 'R50', etc]
    
    # Actually, let's look at how process_one_target_optimize determines p0.
    # If it uses row_sep['R50'] or row_query['r50'].
    # We'll populate reasonable guesses.
    
    # We will assume "Perfect Guess" for now? Or Representative?
    # Using Truth values as Initial Guess is the most "stable" test of the OPTIMIZER (gradient descent).
    # If we want to test "Robustness to bad initialization", we should offset them.
    # Given the user wants to test "optimization", usually that implies finding the minimum.
    # Starting close is better to verify the *minimum* exists and is recovered.
    # Starting far tests global search capability.
    # I'll default to using the Truth parameters from the header as the "Guess" (or close to it),
    # but maybe perturb them slightly to be realistic?
    # Actually, in generate_mocks.py we stored Truth in header.
    
    # Let's extract Truth
    truth_n = float(sci_header.get('n_sersic', 4.0))
    truth_R = float(sci_header.get('R_sersic', 0.4))
    truth_amp = float(sci_header.get('amplitude', 0.05))
    truth_q = float(sci_header.get('q', 0.8))
    truth_theta = float(sci_header.get('theta_ell', 0.0))
    
    # Create Dummy Series
    row_query = pd.Series({
        'sequentialid': seq_id,
        'EXPTIME_SCI': float(sci_header.get('EXPTIME', 2000)),
        'EXPTIME_WHT': 0, # Not used for scaling if WHT map provided?
        'sersic_n_gim2d': truth_n, # Guess
        'r50': truth_R # Guess radius
    })
    
    # image size
    ny, nx = sci_header['NAXIS2'], sci_header['NAXIS1']
    
    row_sep = pd.Series({
        'seqid': seq_id,
        'image_width': nx,
        'image_height': ny,
        'q': truth_q,
        'theta': truth_theta,
        'x': 0, 'y': 0, # Cutout offsets?
        'xcpeak': nx/2 - 0.5, # Center pixel (0-indexed)
        'ycpeak': ny/2 - 0.5,
        'R50': truth_R, # usage depends on code
        'flux': truth_amp * 100, # dummy
    })
    
    return row_query, row_sep

import argparse
import shutil
from tools_source_extractor import load_fits, choose_sep_bg_config, extract_with_sep, pick_target_label, crop_target
import h5py # Added for HDF5 operations in preprocess_directory

# Configuration
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR_BASE = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILENAME = "fitting_results.csv"

def preprocess_directory(d):
    """
    Run Source Extractor Preprocessing on all FITS in directory d.
    Updates/Overwrites the *-cropped.hdf5 files with cropped versions.
    """
    print(f"  Preprocessing mocks in {d}...")
    fits_files = glob.glob(os.path.join(d, "*-SCI.fits"))
    fits_files.sort()
    
    for f_sci in fits_files:
        base = os.path.basename(f_sci).replace("-SCI.fits", "")
        f_wht = os.path.join(d, f"{base}-WHT.fits")
        hdf5_fn = os.path.join(d, f"{base}-cropped.hdf5")
        
        if not os.path.exists(f_wht): continue

        try:
            # Load Data
            sci, wht, orientat, center_xy = load_fits(f_sci, f_wht, return_orientat=True, return_center=True)
            
            # Read header for R_sersic (to set auto_r50)
            with fits.open(f_sci) as hdu:
                hdr = hdu[0].header
                r_sersic = hdr.get('R_sersic', 0.4) # Default 0.4 if missing
                pixscale = hdr.get('PIXSCALE', 0.03)
                
            # SEP Configuration
            # Detect
            objs, segmap = extract_with_sep(sci, wht, 
                                          deblend_nthresh=32, 
                                          deblend_cont=0.005, 
                                          detect_thresh_sigma=3.0, 
                                          minarea=5, 
                                          return_segmap=True)
            
            # Identify Target (Assume center)
            # Center of image
            h, w = sci.shape
            target_xy = (w/2.0, h/2.0)
            
            label, rec, dist = pick_target_label(objs, segmap, target_xy, verbose=False)
            
            # Crop
            # Use crop_target which returns many maps
            # We need: sci_bgsub_crop, wht_crop, segmap_crop, mask_crop
            # Wait, crop_target expects a list of maps.
            # And it returns row (dict) and cropped_data (dict)
            
            # We need to prepare inputs for crop_target
            # It expects [sci_bgsub, wht, rms, segmap, mask] usually?
            # Let's see how process_cutout does it:
            # mask = (segmap != label) * (segmap != 0) + mask_invalid_values
            # params: crop_mode='minmax', crop_factor=1.5
            
            # Simple background subtraction for cropping? 
            # We can use simple sigma clip for now as in process_cutout
            from astropy.stats import sigma_clipped_stats
            mean, median, std = sigma_clipped_stats(sci, sigma=3.0)
            sci_bgsub = sci - median
            
            # Mask
            mask_invalid = (sci==0) | (wht==0)
            mask_others = (segmap != label) & (segmap != 0)
            mask_comb = mask_invalid | mask_others
            
            # We pass [sci_bgsub, wht, segmap, mask_comb]
            # crop_target returns cropped versions
            
            title_list = ['SCI', 'WHT', 'SEGMAP', 'MASK']
            map_list = [sci_bgsub, wht, segmap, mask_comb]
            
            # Temporarily suppress print/plot
            # We define a helper to just do the crop logic or call crop_target interactively?
            # user `preprocess_COSMOS...` calls crop_target.
            # Let's call it.
            
            # We need summary stats dict
            sc_vals = {'mean': mean, 'median': median, 'stdev': std}
            
            _, cropped_data = crop_target(
                map_list, label, rec, verbose=False, plot=False,
                fig_savename=None, title_list=title_list, 
                sigma_clipped_values=sc_vals,
                crop_mode='minmax', crop_factor=2.5, # Generous crop to ensure wings are included (mock profiles can be large)
                pixscale_arcsec=pixscale
            )
            
            # Save to HDF5
            # keys returned: sci_bgsub_crop, wht_crop, ... (based on title list?)
            # Actually crop_target returns list of cropped maps matching input list.
            # Wait, looking at tools_source_extractor.py:
            # It returns (row, cropped_data) dictionary IF called from process_cutout logic?
            # No, crop_target in tools_source_extractor returns (map_list_cropped, obj_rec_for_cropped)
            
            map_list_cropped, rec_cropped = crop_target(
                 map_list, label, rec, verbose=False, plot=False,
                 title_list=title_list,
                 crop_mode='minmax', crop_factor=2.5 # Factor 2.5 of minmax extent
            )
            
            sci_crop = map_list_cropped[0]
            wht_crop = map_list_cropped[1]
            seg_crop = map_list_cropped[2]
            msk_crop = map_list_cropped[3]
            
            with h5py.File(hdf5_fn, "w") as f:
                f.create_dataset("sci_bgsub_crop", data=sci_crop.filled(0) if hasattr(sci_crop, 'filled') else sci_crop)
                f.create_dataset("wht_crop", data=wht_crop.filled(0) if hasattr(wht_crop, 'filled') else wht_crop)
                f.create_dataset("mask_crop", data=msk_crop.filled(0) if hasattr(msk_crop, 'filled') else msk_crop)
                f.create_dataset("segmap_crop", data=seg_crop.filled(0) if hasattr(seg_crop, 'filled') else seg_crop)
                
        except Exception as e:
            print(f"  Preprocessing failed for {base}: {e}")
            # If fail, we leave the original HDF5 (uncropped) or create empty?
            # Original HDF5 exists from generation. We should probably accept it as fallback or fail.
            pass

def run_fitting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default=None, help="Process only dirs matching pattern")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory root (e.g. data/mock_fitting-0201-1)")
    parser.add_argument("--preprocess", action="store_true", help="Run SEP preprocessing/cropping")
    parser.add_argument("--supersample", type=int, default=1, help="Supersampling factor for fitting (default 1)")
    args = parser.parse_args()

    # Create Output Directory Root
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    # Source Directories (Raw Mocks)
    # Assumes generate_mocks.py outputs to PROJECT_ROOT/data/mock_varying_*
    # We look for them there.
    source_pattern = os.path.join(DATA_DIR_BASE, "mock_varying_*")
    all_source_dirs = glob.glob(source_pattern)
    all_source_dirs.sort()

    # Filter
    if args.pattern:
        source_dirs = [d for d in all_source_dirs if args.pattern in os.path.basename(d)]
        print(f"Filtering directories with pattern '{args.pattern}': found {len(source_dirs)}")
    else:
        source_dirs = all_source_dirs
        
    print(f"Found {len(source_dirs)} mock directories to process.")
    
    for src_d in source_dirs:
        dir_name = os.path.basename(src_d)
        target_d = os.path.join(args.out_dir, dir_name)
        
        print(f"Processing {dir_name} -> {target_d}")
        
        # 1. Copy Data (if not exists or update?)
        # Use rsync-like behavior: copy if missing. To ensure clean slate, maybe remove first if exists?
        # For safety/speed, we copy if missing.
        if os.path.exists(target_d):
            # print(f"  Target dir exists. Skipping copy (assuming data present).")
            # If we want to ensure fresh preprocessing, we might need to overwrite HDF5.
            pass
        else:
            print(f"  Copying clean data...")
            shutil.copytree(src_d, target_d)
            
        # 2. Preprocess (Optional)
        if args.preprocess:
             preprocess_directory(target_d)
        
        # Load Truth Data
        truth_csv = os.path.join(target_d, "simulation_truth.csv")
        truth_dict = {}
        if os.path.exists(truth_csv):
            try:
                df_truth = pd.read_csv(truth_csv)
                # create dict keyed by seqid (int)
                for idx, row in df_truth.iterrows():
                     truth_dict[int(row['seqid'])] = row.to_dict()
                print(f"  Loaded truth data for {len(truth_dict)} samples.")
            except Exception as e:
                print(f"  Failed to load truth CSV: {e}")

        # 3. Fit
        # Check if results exist
        out_path = os.path.join(target_d, OUTPUT_FILENAME)
        if os.path.exists(out_path):
             print(f"  Result file exists. Skipping fitting.")
             continue

        # Load Truth to match seqid? Or just loop over FITS
        fits_files = glob.glob(os.path.join(target_d, "*-SCI.fits"))
        fits_files.sort()
        
        results = []
        
        for f_sci in fits_files:
            base = os.path.basename(f_sci).replace("-SCI.fits", "")
            f_wht = os.path.join(target_d, f"{base}-WHT.fits")
            
            if not os.path.exists(f_wht): continue
            
            with fits.open(f_sci) as hdu:
                hdr = hdu[0].header
            
            row_query, row_sep = make_dummy_rows(hdr, base)
            
            # Retrieve truth row
            try:
                seqpixel = int(base) # Assuming base is integer
                row_truth = truth_dict.get(seqpixel, None)
            except:
                row_truth = None
             
            m_list = [3, 4]
            t0 = time.time()
            try:
                rec = process_one_target_optimize(
                    row_query, 
                    target_d, # Data dir
                    row_sep, 
                    m=m_list,
                    opt_method='SLSQP',
                    PIX_SCALE=0.03,
                    plot_initial_contour=False,
                    plot_final_contour=True, # Ensure final plot is made
                    verbose=False,
                    target_loss=1.5,
                    supersample_factor=args.supersample,
                    truth_row=row_truth # Pass truth
                )
                
                rec['fit_time'] = time.time() - t0
                rec['seqid'] = base
                results.append(rec)
                
                # Incremental Save
                df_new = pd.DataFrame([rec])
                if os.path.exists(out_path):
                    df_new.to_csv(out_path, mode='a', header=False, index=False)
                else:
                    df_new.to_csv(out_path, mode='w', header=True, index=False)
                
            except Exception as e:
                print(f"  Failed to fit {base}: {repr(e)}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    run_fitting()
