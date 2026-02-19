import os
import numpy as np
import pandas as pd
from astropy.io import fits
import h5py
from tools_fitting import simulate_model_elliptical_multipole, build_arcsec_grid, pack_params
import warnings

# Configuration
# -----------------------------------------------------------
PIX_SCALE = 0.03 # arcsec/pixel
SUPERSAMPLE_FACTOR = 10
EXPTIME = 4056.0 # seconds (Updated per user request)
RMS_NOISE = 0.005 # Gaussian RMS assumption (gives SNR ~25 for Amp=0.05)
WHT_VALUE = 1.0 / (RMS_NOISE**2)
# IMG_SIZE = 81 # 81x81 pixels -> ~2.4 arcsec box (enough for R_sersic up to ~2, but user wants up to 51.2??)

# User wanted R_sersic up to 51.2 arcsec.
# If R_sersic = 51.2, we need a HUGE box.
# "Fitting Window: Set the image dimensions to 10 times the R_sersic value."
# So image size varies per object! 
# We must adjust grid size dynamically.

DEFAULT_PARAMS = {
    'n_sersic': 4.0,
    'R_sersic': 0.2,
    'amplitude': 0.03,
    'q': 0.8,
    'theta_ell': 0.0,
    'x0': 0.0,
    'y0': 0.0,
    'background': 0.001,
    'a_m': np.array([0.0004, 0.0002]), # for m=3, 4
    'phi_m': np.array([0.0, 0.0]),     # for m=3, 4
}
# m_indices corresponding to a_m/phi_m
M_INDICES = [3, 4]

# Ensure we write to 'data' in the project root
# Script is in analysis/
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "data")

def get_grid_size(R_sersic, factor=10):
    # Rule: factor * R_sersic
    # Convert to pixels
    box_arcsec = factor * R_sersic
    box_pix = int(np.ceil(box_arcsec / PIX_SCALE))
    # Ensure odd
    if box_pix % 2 == 0:
        box_pix += 1
    return box_pix

def generate_noise(image_shape, exptime, wht_value):
    # Sigma Map
    # sigma_tot = sqrt(1/WHT + SCI/EXP_TIME)
    # We add noise: N ~ Gaussian(0, sigma_tot)?
    # Actually, strictly:
    # Counts = Poisson(TrueFlux * Exptime)
    # MeasuredFlux = Counts / Exptime + Gaussian(BackgroundNoise)
    return np.random.normal(0, 1/np.sqrt(wht_value), size=image_shape)

def run_simulation():
    # Define Variations
    variations = {
        'n_sersic': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        'R_sersic': [0.1, 0.2, 0.4, 0.8],
        'amplitude': [1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
        'q': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
        'theta_ell': [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi],
        # 'x0': [-0.5, -0.25, 0.0, 0.25, 0.5],
        # 'y0': [-0.5, -0.25, 0.0, 0.25, 0.5],
        'background': [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05],
        'a_m3': [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
        'a_m4': [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03],
        'phi_m3': np.linspace(-np.pi/6, np.pi/6, 10),
        'phi_m4': np.linspace(-np.pi/8, np.pi/8, 10)
    }

    # Iterate over each parameter to vary
    for param_name, values in variations.items():
        dir_name = f"mock_varying_{param_name}"
        out_dir = os.path.join(OUTPUT_BASE, dir_name)
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Generating mocks for {param_name} in {out_dir}...")
        
        records = []
        
        for i, val in enumerate(values):
            # Start with default params
            p = DEFAULT_PARAMS.copy()
            p['a_m'] = p['a_m'].copy()
            p['phi_m'] = p['phi_m'].copy()
            
            # Update the specific parameter
            if param_name == 'a_m3':
                p['a_m'][0] = val
            elif param_name == 'a_m4':
                p['a_m'][1] = val
            elif param_name == 'phi_m3':
                p['phi_m'][0] = val
            elif param_name == 'phi_m4':
                p['phi_m'][1] = val
            else:
                p[param_name] = val
                
            # Determine Image Size
            nx = get_grid_size(p['R_sersic'])
            ny = nx
            
            # 1. Supersampling
            # Generate grid at finer resolution
            factor = SUPERSAMPLE_FACTOR
            
            # Safety check for memory: Limit fine grid side to 10,000 pixels
            # 10k x 10k x 8 bytes ~ 800 MB. 
            MAX_FINE_SIDE = 10000
            if nx * factor > MAX_FINE_SIDE:
                new_factor = max(1, MAX_FINE_SIDE // nx)
                if new_factor < factor:
                    print('R_sersic: ', p['R_sersic'])
                    print(f"  WARNING: Image too large ({nx}x{nx}) for 10x supersampling. Reducing factor {factor} -> {new_factor}")
                    factor = new_factor
            
            if factor > 1:
                X_fine, Y_fine, ext = build_arcsec_grid((ny*factor, nx*factor), pixscale=PIX_SCALE/factor)
                
                # Evaluate Model
                I_fine = simulate_model_elliptical_multipole(
                    X_fine, Y_fine,
                    n_sersic=p['n_sersic'],
                    R_sersic=p['R_sersic'],
                    amplitude=p['amplitude'],
                    q=p['q'],
                    theta_ell=p['theta_ell'],
                    m=M_INDICES,
                    a_m=p['a_m'],
                    phi_m=p['phi_m'],
                    x0=p['x0'],
                    y0=p['y0'],
                    background=p['background']
                )
                
                # Downsample (Binning) - Mean preserves Surface Brightness
                # Reshape to (ny, factor, nx, factor) and mean over axis 1 and 3
                I_coarse = I_fine.reshape(ny, factor, nx, factor).mean(axis=(1, 3))
            else:
                 # No supersampling case
                X_fine, Y_fine, ext = build_arcsec_grid((ny, nx), pixscale=PIX_SCALE)
                I_coarse = simulate_model_elliptical_multipole(
                    X_fine, Y_fine,
                    n_sersic=p['n_sersic'],
                    R_sersic=p['R_sersic'],
                    amplitude=p['amplitude'],
                    q=p['q'],
                    theta_ell=p['theta_ell'],
                    m=M_INDICES,
                    a_m=p['a_m'],
                    phi_m=p['phi_m'],
                    x0=p['x0'],
                    y0=p['y0'],
                    background=p['background']
                )
            
            # 2. Add Noise
            # Shot noise (Poisson)
            # Flux counts = I * EXPTIME
            # We assume I is in counts/sec (flux)
            counts_true = I_coarse * EXPTIME
            # Poisson realization
            counts_noisy = np.random.poisson(np.abs(counts_true)).astype(float)
            # Recover flux
            I_noisy_poisson = counts_noisy / EXPTIME
            # Add Background Gaussian Noise
            bkg_noise = np.random.normal(0, 1/np.sqrt(WHT_VALUE), size=I_coarse.shape)
            
            SCI = I_noisy_poisson + bkg_noise
            
            # WHT map - constant
            WHT = np.full_like(SCI, WHT_VALUE)
            
            # Save Fits
            seq_id = str(i) # f"{i:04d}" - tools_fitting strips leading zeros
            base_fn = f"{seq_id}"
            sci_fn = os.path.join(out_dir, f"{base_fn}-SCI.fits")
            wht_fn = os.path.join(out_dir, f"{base_fn}-WHT.fits")
            
            hdr = fits.Header()
            hdr['EXPTIME'] = EXPTIME
            hdr['PIXSCALE'] = PIX_SCALE
            # Store Truth in Header too for convenience
            for k, v in p.items():
                if k in ['a_m', 'phi_m']:
                    for j, mv in enumerate(v):
                        hdr[f"{k}{M_INDICES[j]}"] = mv
                else:
                    hdr[k] = v
            
            fits.writeto(sci_fn, SCI, header=hdr, overwrite=True)
            fits.writeto(wht_fn, WHT, header=hdr, overwrite=True)
            
            # Record Truth
            rec = p.copy()
            del rec['a_m']
            del rec['phi_m']
            rec['a_m3'] = p['a_m'][0]
            rec['a_m4'] = p['a_m'][1]
            rec['phi_m3'] = p['phi_m'][0]
            rec['phi_m4'] = p['phi_m'][1]
            rec['seqid'] = seq_id
            rec['filename_sci'] = sci_fn
            rec['param_varied'] = param_name
            rec['value_varied'] = val
            rec['EXPTIME_SCI'] = EXPTIME
            records.append(rec)
            
            # # Save HDF5 for tools_fitting compatibility
            # hdf5_fn = os.path.join(out_dir, f"{seq_id}-cropped.hdf5")
            # with h5py.File(hdf5_fn, "w") as f:
            #     # sci_bgsub_crop, wht_crop, mask_crop, segmap_crop
            #     f.create_dataset("sci_bgsub_crop", data=SCI)
            #     f.create_dataset("wht_crop", data=WHT)
            #     # mask: False = good pixel
            #     f.create_dataset("mask_crop", data=np.zeros_like(SCI, dtype=bool))
            #     f.create_dataset("segmap_crop", data=np.zeros_like(SCI, dtype=int))

            
        # Save Truth CSV
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(out_dir, "simulation_truth.csv"), index=False)
        print(f"Saved {len(df)} mocks to {out_dir}")

if __name__ == "__main__":
    run_simulation()
