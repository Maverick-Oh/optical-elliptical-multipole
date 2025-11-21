from datetime import datetime
import os
from tools_cutout_service import fetch_cutout, param_generator
from astropy.stats import sigma_clipped_stats, sigma_clip
from optical_elliptical_multipole.plotting.plot_tools import AsinhStretchPlot
from scipy.stats import moment

#%%
# General reference for COSMOS:
# https://cosmos.astro.caltech.edu/page/astronomers
import pyvo
import numpy as np
import matplotlib.pyplot as plt
from astropy.io.fits.hdu.hdulist import HDUList
import time
from tool_time_report import elapsed_time_reporter

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
# For DIY manual cutout service:
# https://irsa.ipac.caltech.edu/data/COSMOS/index_cutouts.html
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
WHERE stellarity=0 AND type=1 AND ACS_MU_CLASS=1 ORDER BY sequentialid DESC
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

datetime_string = str(datetime.now()).replace(' ', '_').replace(':', '')
datetime_string = datetime_string[:datetime_string.find('.')]
# datetime_string = 'test'

data_dir = '../data'
hdul_dir = os.path.join(data_dir, f'HDUL_{datetime_string}')
os.makedirs(hdul_dir, exist_ok=True)

# target_sid = None # ( or [] if there is not specific target to debug ); or list of numbers like [131286, 131486]
target_sid = [] #
verbose = True
debug = True

svc = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
tab = svc.run_sync(adql).to_table()  # Astropy Table
tab.write(os.path.join(hdul_dir, f"cosmos_sample_N={len(tab)}_{datetime_string}.csv"), format="csv", overwrite=True)
# SAVE ADQL (archiving purpose)
with open(os.path.join(hdul_dir, f"ADQL_Query_{datetime_string}.sql"), "w") as file:
    file.write(adql)


pixel_width = 0.03 #arcsec/px

plot = True
bg_mean = np.zeros(len(tab))
bg_median = np.zeros(len(tab))
bg_std = np.zeros(len(tab))
exposure_time_sci_list = []
exposure_time_wht_list = []
exposure_flag_sci_list = []
exposure_flag_wht_list = []
for i in range(len(tab)):
    seq_id = int(tab[i]['sequentialid'])
    if target_sid is None or target_sid == []:
        pass
    else:
        if seq_id in target_sid:
            pass
        else:
            continue
    r50_arcsec = tab[i]['r50'] * pixel_width # pixels to arcsec
    cutout_r50_factor = 20 # make cutout that is this times bigger
    cutout_arcsec = min(180, r50_arcsec*cutout_r50_factor)
    t0 = time.perf_counter()
    elapsed_time_reporter(t0, i, total=len(tab), seq_id = seq_id)
    enlarged_count = 0
    while True:
        param = param_generator(tab[i],
                                cutouttbl1='acs_2.0_cutouts', # 'acs_mosaic_2.0' for mosaic (no WHT) or 'acs_2.0_cutouts' for tiles
                                cutout_arcsec=cutout_arcsec)
        hdul_dict, sci_url_list, wht_url_list = fetch_cutout(param, debug=True)
        # fig, ax = plt.subplots(); AsinhStretchPlot(ax, hdul['SCI'][0].data, origin='lower'); plt.show()
        # Saving
        sci_list = []
        wht_list = []
        for i, (sci_hdul, wht_hdul) in enumerate(zip(hdul_dict['SCI'], hdul_dict['WHT'])):
            assert len(hdul_dict['SCI'][i]) == 1 # Each HDUL usually has one file only as a list item
            assert len(hdul_dict['WHT'][i]) == 1 #
            sci = hdul_dict['SCI'][i][0]
            wht = hdul_dict['WHT'][i][0]
            sci_list.append(sci)
            wht_list.append(wht)
        assert len(sci_list) == len(wht_list)
        if len(sci_list) > 1:
            if verbose:
                print("#### Multiple SCI and WHT images found!")
                print("SCI image lists: \n", sci_url_list)
                print("WHT image lists: \n", wht_url_list)
            # check how many zero pixels there are in each science image
            num_zeros = [np.sum(sci_ == 0.) for sci_ in sci_list]
            inds = np.where(np.array(num_zeros) == 0)[0] # indices that has 0 pixel of zeros; all good picture
            if len(inds) == 1: # if there is only one such index, proceed with it
                sci = sci_list[inds[0]]
                wht = wht_list[inds[0]]
                if verbose:
                    print(f"{inds[0]}-th files are chosen because they dont have masked/zero regions:")
                    print("Number of pixels that had zeros in different files: ", num_zeros)
            elif len(inds) > 1: # if there are multiple such images, just choose one and report
                sci = sci_list[inds[0]]
                wht = wht_list[inds[0]]
                if verbose:
                    print(f"{inds[0]}-th files are chosen randomly; multiple files has 0 zero pixels:")
                    print("Number of pixels that had zeros in different files: ", num_zeros)
            elif len(inds) == 0:
                ind = np.argmin(num_zeros)
                sci = sci_list[ind]
                wht = wht_list[ind]
                if verbose:
                    print("No file has 0 zero pixels. Choosing a better one.")
                    print(f"{ind}-th files are chosen because they have less zero pixels:")
                    print("Number of pixels that had zeros in different files: ", num_zeros)
        # overwriting existing list of sci and wht to the chosen one
        hdul_dict['SCI'] = sci; hdul_dict['WHT'] = wht
        print(f"weight mean: {np.mean(wht.data)}, stdev: {np.std(wht.data)}")
        if np.isnan(sci.data).any():
            if verbose:
                print(f"{np.sum(np.isnan(sci))} pixels are NaN: masking them")
            sci = np.ma.masked_array(sci, mask = np.isnan(sci.data))
        if (sci.data == 0.).any():
            if verbose:
                print(f"{np.sum(sci == 0.)} pixels are zeros: masking them")
            sci_data = np.ma.masked_array(sci.data, mask = sci.data == 0.)
        else:
            sci_data = sci.data
        if debug:
            fig, ax = plt.subplots(); my_plot = AsinhStretchPlot(ax, sci_data, origin='lower'); plt.colorbar(
                mappable=my_plot)
            fig.savefig(os.path.join(data_dir, f"{seq_id}-enlarged-{enlarged_count}.pdf")); plt.show()
        im_clipped = sigma_clip(sci_data, sigma=3.0)
        clipped_im_flat = im_clipped.flatten()
        moment_val = moment(clipped_im_flat[~clipped_im_flat.mask], order=3)
        if moment_val<1e-8:
            break # good end
        elif cutout_arcsec==180:
            print(f"cutout_arcsec={cutout_arcsec} but moment value still: {moment_val} for seq_id: {seq_id}")
            break
        else:
            enlarged_count += 1
            cutout_arcsec = min(2*cutout_arcsec, 180)
    # The correct sutout size is sorted out (zooming out)
    # sci and wht are now not data array but PrimaryHDU with headers!
    exposure_time_sci = sci.header['EXPTIME']
    exposure_time_wht = wht.header['EXPTIME']
    exposure_flag_sci = sci.header['EXPFLAG']
    exposure_flag_wht = wht.header['EXPFLAG']
    print(f"Exposure time (SCI): {exposure_time_sci} ({exposure_flag_sci})")
    print(f"Exposure time (WHT): {exposure_time_wht} ({exposure_flag_wht})")
    exposure_time_sci_list.append(exposure_time_sci)
    exposure_time_wht_list.append(exposure_time_wht)
    exposure_flag_sci_list.append(exposure_flag_sci)
    exposure_flag_wht_list.append(exposure_flag_wht)
    #
    if type(hdul_dict) is dict:
        for key in hdul_dict.keys():
            hdul_dict[key].writeto(os.path.join(hdul_dir, f"{seq_id}-{key}.fits"), overwrite=True)
    elif type(hdul_dict) is HDUList:
        hdul_dict.writeto(os.path.join(hdul_dir, f"{seq_id}.fits"), overwrite=True)

    mean, median, stdev = sigma_clipped_stats(sci_data, sigma=3.0)
    bg_mean[i] = mean; bg_median[i] = median; bg_std[i] = stdev
    if plot:
        #
        extent = [-cutout_arcsec/2, cutout_arcsec/2, -cutout_arcsec/2, cutout_arcsec/2]
        fig, axes = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)
        im_, norm = AsinhStretchPlot(axes[0], sci_data,
                                     return_norm=True,
                                     origin="lower",
                                     extent=extent)
        plt.colorbar(mappable=im_, ax=axes[0])
        #
        axes[1].set_facecolor('k')
        im__ = AsinhStretchPlot(axes[1], im_clipped,
                                norm=norm, return_norm=False,
                                origin="lower", extent=extent)
        plt.colorbar(mappable=im__, ax=axes[1])
        #
        axes[0].set_aspect("equal"); axes[1].set_aspect("equal")
        #
        axes[2].hist(clipped_im_flat[~clipped_im_flat.mask], bins=100)
        axes[2].axvline(mean, color='r', label='mean')
        axes[2].axvline(median, color='b', label='median')
        #
        axes[0].set_title(f"image\n({enlarged_count} times enlarged)")
        axes[1].set_title("clipped image\n")
        axes[2].set_title(f"histogram of clipped image\nmoment-3: {moment_val:.1e}")
        # x and y labels
        axes[0].set_xlabel("x (arcsec)")
        axes[1].set_xlabel("x (arcsec)")
        axes[2].set_xlabel("flux")
        #
        axes[2].set_yticks([])
        #
        plt.legend()
        plt.tight_layout()
        #
        fig.savefig(os.path.join(hdul_dir, f"{seq_id}-00-clip.pdf"))
        plt.show()
        #
        print("")
    plt.show()

tab['EXPTIME_SCI'] = exposure_time_sci_list
tab['EXPTIME_WHT'] = exposure_time_wht_list
tab['EXPFLAG_SCI'] = exposure_flag_sci_list
tab['EXPFLAG_WHT'] = exposure_flag_wht_list
# Rewrite the tab because the exposure time information is added!
tab.write(os.path.join(hdul_dir, f"cosmos_sample_N={len(tab)}_{datetime_string}.csv"), format="csv", overwrite=True)

print("Done!")