from datetime import datetime
import os
from tools_cutout_service import fetch_cutout, param_generator, cutout_selection_metric
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
from tools_misc import elapsed_time_reporter, play_sound_list, radec_to_pixel
import warnings
from astropy.wcs import FITSFixedWarning

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

show_plot = False
adql = """SELECT 
TOP 10000
sequentialid, CAPAK_ID, acs_ident, ra, dec, type, 
ACS_MU_CLASS, R50, ACS_X_IMAGE, ACS_Y_IMAGE,
ACS_A_IMAGE, ACS_B_IMAGE, ACS_THETA_IMAGE, 
R_GIM2D, ell_gim2d, PA_GIM2D, SERSIC_N_GIM2D
FROM cosmos_morph_zurich_1
WHERE stellarity=0 AND type=1 AND ACS_MU_CLASS=1 ORDER BY R50 DESC
"""
# column names and descriptions can be found in: https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd

# ORDER BY r50 DESC
# ORDER BY sequentialid ASC
# type ZEST Type CLASS, 1 = Early type, 2 = Disk, 3 = Irregular Galaxy, 9 = no classification
# ACS_MU_CLASS: Type of object. 1 = galaxy, 2 = star, 3 = spurious
# STELLARITY: 0 if ACS_CLASS_STAR<0.6 (object is ASSUMED to be a galaxy; no visual inspection); 0 if ACS_CLASS_STAR>=0.6 AND object visually identified as a galaxy.
# ELL_GIM2D: GIM2D ellipticity = 1-b/a of object
# ACS_MU_CLASS: Type of object. 1 = galaxy, 2 = star, 3 = spurious
# ACS_A_IMAGE	float	 	SExtractor semi-major axis
# ACS_B_IMAGE	float	 	SExtractor semi-minor axis
# ACS_X_IMAGE	float	pixel	X-pixel position on ACS-tile
# ACS_Y_IMAGE	float	pixel	Y-pixel position on ACS-tile

# datetime_string = str(datetime.now()).replace(' ', '_').replace(':', '')
# datetime_string = datetime_string[:datetime_string.find('.')]
datetime_string = 'ALL'
ignore_FITSFixedWarning = True

data_dir = '../data'
hdul_dir = os.path.join(data_dir, f'HDUL_{datetime_string}')
os.makedirs(hdul_dir, exist_ok=True)

# target_sid = None # ( or [] if there is not specific target to debug ); or list of numbers like [131286, 131486]
target_sid = [] #
ignore_sid = []
ignore_index = []
verbose = True
debug = False

svc = pyvo.dal.TAPService("https://irsa.ipac.caltech.edu/TAP")
tab = svc.run_sync(adql).to_table()  # Astropy Table
tab.write(os.path.join(hdul_dir, f"cosmos_sample_N={len(tab)}_{datetime_string}.csv"), format="csv", overwrite=True)
# SAVE ADQL (archiving purpose)
with open(os.path.join(hdul_dir, f"ADQL_Query_{datetime_string}.sql"), "w") as file:
    file.write(adql)

pixel_width = 0.03 #arcsec/px

if ignore_FITSFixedWarning:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)

plot = True
bg_mean = np.zeros(len(tab))
bg_median = np.zeros(len(tab))
bg_std = np.zeros(len(tab))
# typically 4056 s for exposure time
exposure_time_sci_list = [None] * len(tab)
exposure_time_wht_list = [None] * len(tab)
exposure_flag_sci_list = [None] * len(tab)
exposure_flag_wht_list = [None] * len(tab)
target_x_pixel_list = [None] * len(tab)
target_y_pixel_list = [None] * len(tab)
cutout_metric_list = [None] * len(tab)
t0 = time.perf_counter()
for i in range(len(tab)):
    seq_id = int(tab[i]['sequentialid'])
    if i in ignore_index:
        continue
    if seq_id in ignore_sid:
        continue
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
    elapsed_time_reporter(t0, i, total=len(tab), seq_id = seq_id)
    enlarged_count = 0
    while True:
        bad_flag = False
        param = param_generator(tab[i],
                                cutouttbl1='acs_2.0_cutouts', # 'acs_mosaic_2.0' for mosaic (no WHT) or 'acs_2.0_cutouts' for tiles
                                cutout_arcsec=cutout_arcsec)
        if debug:
            # print location string
            print("location string (can be pasted to: https://irsa.ipac.caltech.edu/data/COSMOS/index_cutouts.html)")
            print(param["locstr"])
            print("cutout_arcsec:", cutout_arcsec)
        if debug:
            print("fetch_cutout starting...")
        hdul_dict, sci_url_list, wht_url_list = fetch_cutout(param, debug=True)
        if debug:
            print("fetch_cutout done!!!")
        # fig, ax = plt.subplots(); AsinhStretchPlot(ax, hdul['SCI'][0].data, origin='lower'); plt.show()
        # Saving
        sci_list = []
        wht_list = []
        for ii, (sci_hdul, wht_hdul) in enumerate(zip(hdul_dict['SCI'], hdul_dict['WHT'])):
            assert len(hdul_dict['SCI'][ii]) == 1 # Each HDUL usually has one file only as a list item
            assert len(hdul_dict['WHT'][ii]) == 1 #
            sci = hdul_dict['SCI'][ii][0]
            wht = hdul_dict['WHT'][ii][0]
            sci_list.append(sci)
            wht_list.append(wht)
        assert len(sci_list) == len(wht_list)
        if len(sci_list) == 1: # single Tile image
            sci = sci_list[0]
            wht = wht_list[0]
            target_x, target_y = radec_to_pixel(sci, ra=tab[i]['ra'], dec=tab[i]['dec'])
            # still evaluate cutout metric just in case something is weird
            target_xy_coords = [radec_to_pixel(sci_, ra=tab[i]['ra'], dec=tab[i]['dec']) for sci_ in sci_list]
            full_metric, metric_dict = cutout_selection_metric(target_xy_coords, sci_list, pixel_width, r50_arcsec)
            cutout_metric_list[i] = full_metric
        else: # multiple imagse
            if verbose:
                print("## Multiple SCI and WHT images found! ##")
                print("SCI image lists: \n", sci_url_list)
                print("WHT image lists: \n", wht_url_list)
            target_xy_coords = [radec_to_pixel(sci_, ra=tab[i]['ra'], dec=tab[i]['dec']) for sci_ in sci_list]
            full_metric, metric_dict = cutout_selection_metric(target_xy_coords,
                                                               sci_list, pixel_width, r50_arcsec)
            # indices that has zeros
            ind_selected = np.argmin(full_metric) # indices that has 0 pixel of zeros; all good picture
            #
            full_metric_thresh = 5.0
            # Check if there are multiple best ones; if so, flag bad.
            if np.sum(full_metric==np.min(full_metric)) > 1:
                bad_flag = True
            if verbose:
                print("inside: ", metric_dict['inside'], "inside metric: ", metric_dict['inside_metric'])
                print("dist_to_bdry_px: ", metric_dict['dist_to_bdry_px'], "distance_metric: ", metric_dict['distance_metric'])
                print(f"row vs. column: {metric_dict['ratio_array']}", "ratio_metric: ", metric_dict['ratio_metric'])
                print(f"zero ratio: {metric_dict['zero_ratio']}", "zero_ratio_metric: ", metric_dict['zero_ratio_metric'])
                print(f"Selected index: {ind_selected}, with metric value: {full_metric[ind_selected]} "
                      f"(good if < {full_metric_thresh:.1f})")
            if full_metric[ind_selected] >= full_metric_thresh:
                bad_flag = True # Let human to check
            fig, axes = plt.subplots(1, len(sci_list), figsize=(len(sci_list)*3, 3))
            for ii, sci_ii in enumerate(sci_list):
                title_ii = f'{seq_id}-{ii} (selected)' if ii==ind_selected else f'{seq_id}-{ii}'
                data2plot = np.ma.masked_array(sci_ii.data, mask = sci_ii.data==0.)
                im_bg = plt.imshow(sci_ii.data==0., origin='lower', cmap='gray') # such that the axis face (black) is
                # seperated from the tile's no-data region
                im_ii = AsinhStretchPlot(axes[ii], data2plot, title=title_ii, origin='lower')
                axes[ii].axvline(target_xy_coords[ii][0], ymin=0., ymax=cutout_arcsec/pixel_width, color='w',
                                 linewidth=0.5, linestyle='--')
                axes[ii].axhline(target_xy_coords[ii][1], xmin=0., xmax=cutout_arcsec / pixel_width, color='w',
                                 linewidth=0.5, linestyle='--')
                axes[ii].set_facecolor('k')
                plt.colorbar(mappable=im_ii)
                axes[ii].set_aspect('equal')
            fig.tight_layout()
            fig.savefig(os.path.join(hdul_dir, f"{seq_id}-00-comparison_cutout.pdf"))
            plt.show() if (debug or bad_flag) else plt.close(fig)
            # in case things look weird...
            if bad_flag:
                print(f"#### It seems SCI image might not satisfy criteria. full_metric = {full_metric}; ind_selected = {ind_selected}")
                while True:
                    play_sound_list(frequency_list=[220,440,880, 1760], duration_list=[0.5,0.5,0.5, 0.5])
                    ind_selected = input("#### Select which one SCI image to go with, by putting an integer (e.g. 0, 1, 2): ")
                    try:
                        # Criteria to match
                        cri1 = float(ind_selected) == int(ind_selected)
                        cri2 = int(ind_selected) <= (len(sci_list)-1)
                        cri3 = int(ind_selected) >= 0
                        if cri1 and cri2 and cri3:
                            ind_selected = int(ind_selected)
                            break
                    except ValueError:
                        print("Please enter a valid integer.")
                    except Exception as e:
                        raise e

            sci = sci_list[ind_selected]
            wht = wht_list[ind_selected]
            target_x, target_y = radec_to_pixel(sci, ra=tab[i]['ra'], dec=tab[i]['dec'])

        # overwriting existing list of sci and wht to the chosen one
        # play_sound_list()
        hdul_dict['SCI'] = sci; hdul_dict['WHT'] = wht
        # print(f"weight mean: {np.mean(wht.data)}, stdev: {np.std(wht.data)}")
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

        im_clipped = sigma_clip(sci_data, sigma=3.0)
        clipped_im_flat = im_clipped.flatten()
        moment_val = moment(clipped_im_flat[~clipped_im_flat.mask], order=3)
        moment_thresh = 1e-8
        if moment_val<moment_thresh:
            break # good end
        elif cutout_arcsec==180:
            print(f"cutout_arcsec={cutout_arcsec} but moment value still: {moment_val} for seq_id: {seq_id}")
            break
        else:
            enlarged_count += 1
            cutout_arcsec = min(2*cutout_arcsec, 180)
            if verbose:
                print(f"enlarging further for better noise evaluation... Current moment-3 value: {moment_val:1e} ("
                      f"{moment_thresh} or lower required)")
    target_x_pixel_list[i] = target_x
    target_y_pixel_list[i] = target_y
    # The correct sutout size is sorted out (zooming out)
    # sci and wht are now not data array but PrimaryHDU with headers!
    exposure_time_sci = sci.header['EXPTIME']
    exposure_time_wht = wht.header['EXPTIME']
    exposure_flag_sci = sci.header['EXPFLAG']
    exposure_flag_wht = wht.header['EXPFLAG']
    if debug:
        print(f"Exposure time (SCI): {exposure_time_sci} ({exposure_flag_sci})")
        print(f"Exposure time (WHT): {exposure_time_wht} ({exposure_flag_wht})")
    exposure_time_sci_list[i] = exposure_time_sci
    exposure_time_wht_list[i] = exposure_time_wht
    exposure_flag_sci_list[i] = exposure_flag_sci
    exposure_flag_wht_list[i] = exposure_flag_wht
    #
    if type(hdul_dict) is dict:
        for key in hdul_dict.keys():
            hdul_dict[key].writeto(os.path.join(hdul_dir, f"{seq_id}-{key}.fits"), overwrite=True)
    elif type(hdul_dict) is HDUList:
        hdul_dict.writeto(os.path.join(hdul_dir, f"{seq_id}.fits"), overwrite=True)
    if debug:
        print("sigma clipping...")
    mean, median, stdev = sigma_clipped_stats(sci_data, sigma=3.0)
    bg_mean[i] = mean; bg_median[i] = median; bg_std[i] = stdev
    if plot:
        row, col = sci.data.shape
        extent = pixel_width * np.array([-col/2, +col/2, -row/2, +row/2])
        # extent = [-cutout_arcsec/2, cutout_arcsec/2, -cutout_arcsec/2, cutout_arcsec/2] # [left, right, bottom, top]
        fig, axes = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)
        im_, norm = AsinhStretchPlot(axes[0], sci_data,
                                     return_norm=True,
                                     origin="lower",
                                     extent=extent)
        plt.colorbar(mappable=im_, ax=axes[0])
        #
        axes[0].set_facecolor('k')
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
        plt.show() if (debug or bad_flag) else plt.close(fig)
        #

tab['EXPTIME_SCI'] = exposure_time_sci_list
tab['EXPTIME_WHT'] = exposure_time_wht_list
tab['EXPFLAG_SCI'] = exposure_flag_sci_list
tab['EXPFLAG_WHT'] = exposure_flag_wht_list
tab['target_x_pixel'] = target_x_pixel_list
tab['target_y_pixel'] = target_y_pixel_list
# Rewrite the tab because the exposure time information is added!
datetime_string_new = str(datetime.now()).replace(' ', '_').replace(':', '')
datetime_string_new = datetime_string_new[:datetime_string_new.find('.')]
tab.write(os.path.join(hdul_dir, f"cosmos_sample_N={len(tab)}_{datetime_string}-at-{datetime_string_new}.csv"), format="csv",
          overwrite=True)
print(f"Written: cosmos_sample_N={len(tab)}_{datetime_string}-at-{datetime_string_new}.csv")
print(f"At the following directory: {hdul_dir}")

print("Done!")