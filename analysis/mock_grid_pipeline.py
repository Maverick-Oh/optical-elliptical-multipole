import os
import numpy as np
import pandas as pd
from astropy.io import fits
from tools_fitting import simulate_model_elliptical_multipole, build_arcsec_grid
from tqdm import tqdm

PIX_SCALE = 0.03
SUPERSAMPLE_FACTOR = 3
EXPTIME = 4056.0
RMS_NOISE = 0.005
WHT_VALUE = 1.0 / (RMS_NOISE**2)

DEFAULT_PARAMS = {
    'amplitude': 0.03,
    'q': 0.8,
    'theta_ell': 0.0,
    'x0': 0.0,
    'y0': 0.0,
    'background': 0.001,
}
M_INDICES = [3, 4]

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(OUTPUT_BASE, "mock_grid_validation/mock_varying_all")

def get_grid_size(R_sersic, factor=10):
    box_arcsec = factor * R_sersic
    box_pix = int(np.ceil(box_arcsec / PIX_SCALE))
    if box_pix % 2 == 0:
        box_pix += 1
    return box_pix

def run_grid_generation():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    a_ms = [0.001, 0.003, 0.01, 0.03, 0.1]
    n_sersics = [2.0, 2.5, 3.0, 3.5, 4.0]
    R_sersics = [0.4, 0.8, 1.2, 1.6, 2.0]
    
    records = []
    seq_id_counter = 0

    print(f"Generating mocks in {OUT_DIR}...")
    
    total_iters = len(a_ms) * len(n_sersics) * len(R_sersics)
    
    with tqdm(total=total_iters) as pbar:
        for a_m in a_ms:
            for n_sersic in n_sersics:
                # Check for existing sets to skip
                if np.isclose(a_m, 0.01) and (np.isclose(n_sersic, 2.0) or np.isclose(n_sersic, 3.0) or np.isclose(n_sersic, 4.0)):
                    for R_sersic in R_sersics:
                        pbar.update(1)
                    continue
                
                for R_sersic in R_sersics:
                    p = DEFAULT_PARAMS.copy()
                    p['n_sersic'] = n_sersic
                    p['R_sersic'] = R_sersic
                    p['a_m'] = np.array([a_m, a_m])
                    p['phi_m'] = np.array([0.0, 0.0])
                    
                    nx = get_grid_size(p['R_sersic'])
                    ny = nx
                    factor = SUPERSAMPLE_FACTOR
                    
                    MAX_FINE_SIDE = 10000
                    if nx * factor > MAX_FINE_SIDE:
                        factor = max(1, MAX_FINE_SIDE // nx)
                    
                    if factor > 1:
                        X_fine, Y_fine, ext = build_arcsec_grid((ny*factor, nx*factor), pixscale=PIX_SCALE/factor)
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
                        I_coarse = I_fine.reshape(ny, factor, nx, factor).mean(axis=(1, 3))
                    else:
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
                    
                    counts_true = I_coarse * EXPTIME
                    counts_noisy = np.random.poisson(np.abs(counts_true)).astype(float)
                    I_noisy_poisson = counts_noisy / EXPTIME
                    bkg_noise = np.random.normal(0, 1/np.sqrt(WHT_VALUE), size=I_coarse.shape)
                    
                    SCI = I_noisy_poisson + bkg_noise
                    WHT = np.full_like(SCI, WHT_VALUE)
                    
                    seq_id = str(seq_id_counter)
                    seq_id_counter += 1
                    
                    base_fn = f"{seq_id}"
                    sci_fn = os.path.join(OUT_DIR, f"{base_fn}-SCI.fits")
                    wht_fn = os.path.join(OUT_DIR, f"{base_fn}-WHT.fits")
                    
                    hdr = fits.Header()
                    hdr['EXPTIME'] = EXPTIME
                    hdr['PIXSCALE'] = PIX_SCALE
                    for k, v in p.items():
                        if k in ['a_m', 'phi_m']:
                            for j, mv in enumerate(v):
                                hdr[f"{k}{M_INDICES[j]}"] = mv
                        else:
                            hdr[k] = v
                    
                    fits.writeto(sci_fn, SCI, header=hdr, overwrite=True)
                    fits.writeto(wht_fn, WHT, header=hdr, overwrite=True)
                    
                    rec = p.copy()
                    del rec['a_m']
                    del rec['phi_m']
                    rec['a_m3'] = p['a_m'][0]
                    rec['a_m4'] = p['a_m'][1]
                    rec['phi_m3'] = p['phi_m'][0]
                    rec['phi_m4'] = p['phi_m'][1]
                    rec['seqid'] = seq_id
                    rec['filename_sci'] = sci_fn
                    rec['target_a_m'] = a_m
                    rec['EXPTIME_SCI'] = EXPTIME
                    records.append(rec)
                    
                    pbar.update(1)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUT_DIR, "simulation_truth.csv"), index=False)
    print(f"\nSaved {len(df)} mocks to {OUT_DIR}")

if __name__ == "__main__":
    run_grid_generation()
