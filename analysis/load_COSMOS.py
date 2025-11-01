from datetime import datetime
import os
from tools_cutout_service import fetch_cutout, param_generator
from astropy.stats import sigma_clipped_stats, sigma_clip
# datetime_string = str(datetime.now()).replace(' ', '_').replace(':', '')
# datetime_string = datetime_string[:datetime_string.find('.')]

datetime_string = 'test'

#%%
# General reference for COSMOS:
# https://cosmos.astro.caltech.edu/page/astronomers
import pyvo
import numpy as np
import matplotlib.pyplot as plt

#%% Code to check all column names
"""
svc = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
adql = "SELECT TOP 1 * FROM cosmos_morph_zurich_1"
tab = svc.run_sync(adql).to_table()  # Astropy Table
colnames = tab.colnames
print("Column names:")
print(colnames)
# Check column information here:
# https://irsa.ipac.caltech.edu/data/COSMOS/gator_docs/cosmos_morph_zurich_colDescriptions.html
# For Cutouts Program and information, Check:
# https://irsa.ipac.caltech.edu/applications/Cutouts/docs/CutoutsProgramInterface.html
print("DONE!")
"""

#%%
# TYPE=1 are early-type (E/S0) in ZEST; STELLARITY=0 excludes stars
# R50 is in pixels (ACS scale = 0.03 arcsec/pix per catalog docs)
# ELL_GIM2D = 1 - (b/a); we also return b/a explicitly.
adql = """SELECT 
TOP 10 
sequentialid, CAPAK_ID, ra, dec, type, 
ACS_MU_CLASS, R50, ACS_X_IMAGE, ACS_Y_IMAGE,
ACS_A_IMAGE, ACS_B_IMAGE, ACS_THETA_IMAGE, 
R_GIM2D, ell_gim2d, PA_GIM2D, SERSIC_N_GIM2D
FROM cosmos_morph_zurich_1
WHERE stellarity=0 AND type=1 AND ACS_MU_CLASS=1 ORDER BY R50 DESC
"""
## type: ZEST Type CLASS, 1 = Early type, 2 = Disk, 3 = Irregular Galaxy, 9 = no classification
# ACS_MU_CLASS: Type of object. 1 = galaxy, 2 = star, 3 = spurious
# STELLARITY: 0 if ACS_CLASS_STAR<0.6 (object is ASSUMED to be a galaxy; no visual inspection); 0 if ACS_CLASS_STAR>=0.6 AND object visually identified as a galaxy.
# ELL_GIM2D: GIM2D ellipticity = 1-b/a of object
# ACS_MU_CLASS: Type of object. 1 = galaxy, 2 = star, 3 = spurious
# ACS_A_IMAGE	float	 	SExtractor semi-major axis
# ACS_B_IMAGE	float	 	SExtractor semi-minor axis
# ACS_X_IMAGE	float	pixel	X-pixel position on ACS-tile
# ACS_Y_IMAGE	float	pixel	Y-pixel position on ACS-tile

data_dir = '../data'
hdul_dir = os.path.join(data_dir, f'HDUL_{datetime_string}')
os.makedirs(hdul_dir, exist_ok=True)

svc = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
tab = svc.run_sync(adql).to_table()  # Astropy Table
tab.write(os.path.join(hdul_dir, f"cosmos_sample_N={len(tab)}_{datetime_string}.csv"), format="csv", overwrite=True)
# SAVE ADQL (archiving purpose)
with open(os.path.join(hdul_dir, f"ADQL_Query_{datetime_string}.sql"), "w") as file:
    file.write(adql)

import time

def elapsed_time_reporter(t0, i, total, seq_id=None):
    done = i + 1
    elapsed = time.perf_counter() - t0
    # items/sec (avoid div by zero)
    rate = done / elapsed if elapsed > 0 else float('inf')
    rem = len(tab) - done
    eta_sec = rem / rate if np.isfinite(rate) and rate > 0 else float('nan')
    if np.isfinite(eta_sec):
        m, s = divmod(int(round(eta_sec)), 60)
        h, m = divmod(m, 60)
        eta_str = f"{h:02d}:{m:02d}:{s:02d}"
    else:
        eta_str = "--:--:--"
    msg = f"\rProcessing: [{done:>5}/{len(tab):<5}]  ETA: {eta_str}, sequentialid: {seq_id}"
    print(msg, end='', flush=True)
    return None

plot = True
bg_mean = np.zeros(len(tab))
bg_median = np.zeros(len(tab))
bg_std = np.zeros(len(tab))
for i in range(len(tab)):
    seq_id = int(tab[i]['sequentialid'])
    t0 = time.perf_counter()
    elapsed_time_reporter(t0, i, total=len(tab), seq_id = seq_id)
    param = param_generator(tab[i], cutout_arcsec=80.)
    hdul = fetch_cutout(param)
    if len(hdul) > 1:
        raise ValueError("len(hdul)>1")
    # Saving
    mean, median, stdev = sigma_clipped_stats(hdul[0].data, sigma=3.0)
    bg_mean[i] = mean; bg_median[i] = median; bg_std[i] = stdev
    if plot:
        im = hdul[0].data
        im_clipped = sigma_clip(im, sigma=3.0)
        #
        fig, axes = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)
        with np.errstate(invalid='ignore', divide='ignore'):
            im_ = axes[0].imshow(np.log10(im), origin="lower", vmin=-6, vmax=np.nanmax(np.log10(im)))
        plt.colorbar(mappable=im_, ax=axes[0])
        #
        axes[1].set_facecolor('k')
        from matplotlib.colors import ListedColormap
        cmap_for_negative = ListedColormap(["#000000", "#ff0000"])
        im__negative = axes[1].imshow((im_clipped<0), origin="lower", cmap=cmap_for_negative)
        with np.errstate(invalid='ignore', divide='ignore'):
            im__ = axes[1].imshow(np.log10(im_clipped), origin="lower", vmin=-6, vmax=np.nanmax(np.log10(im)))
        plt.colorbar(mappable=im__, ax=axes[1])
        #
        axes[0].set_aspect("equal"); axes[1].set_aspect("equal")
        #
        axes[2].hist(im_clipped.flatten(), bins=100)
        axes[2].axvline(mean, color='r', label='mean')
        axes[2].axvline(median, color='b', label='median')
        #
        axes[0].set_title("log10(image)\n")
        axes[1].set_title("log10(clipped image)\n(red: negative)")
        axes[2].set_title("histogram of clipped image")
        # x and y labels
        axes[0].set_xlabel("x (pixels)")
        axes[1].set_xlabel("x (pixels)")
        axes[2].set_xlabel("flux")
        #
        axes[2].set_yticks([])
        #
        plt.legend()
        plt.tight_layout()
        #
        fig.savefig(os.path.join(hdul_dir, f"{seq_id}.pdf"))
        plt.show()
        #
        print("")
    plt.show()
    hdul.writeto(os.path.join(hdul_dir, f"{seq_id}.fits"), overwrite=True)
print("Done!")