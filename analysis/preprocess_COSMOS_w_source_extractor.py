# TODO:
# 0. Glob glob list
# 1. Make args for each case for
# 2. Crop Ratio = 1.0; if crop ratio == 1.0, pass
# 3. auto_r50_arcsec <- based on my object of interest
# 4. size_arcsec <- Remove dependency
# 5. filename_format, seqid_list <- Deal withit before putting it into process cutout
# 6.  <-
#
#
# min area 20 <- enough? Think about it.

#!/usr/bin/env python3
from matplotlib import pyplot as plt
from tools_source_extractor import load_fits, choose_sep_bg_config, background_analysis, extract_with_sep, pick_target_label, crop_target, cumulative_ellipse_curve, find_percent_radii, crop_array_list_w_ratio, AsinhStretchPlot
from optical_elliptical_multipole.plotting.plot_tools import draw_segmentation, AsinhStretchPlot
from pathlib import Path
import os
import csv
import numpy as np
import h5py, glob
from types import SimpleNamespace
import pandas
import time
from tools_misc import elapsed_time_reporter, play_sound_list, upsert_sep_summary_row
from astropy.stats import sigma_clipped_stats
from matplotlib.ticker import MaxNLocator
from astropy.wcs import FITSFixedWarning
import warnings

ignore_FITSFixedWarning = True
if ignore_FITSFixedWarning:
    warnings.filterwarnings("ignore", category=FITSFixedWarning)

def process_cutout(seqid: str, args, verbose=True, debug=True):
    fits_path_sci = Path(os.path.join(args.path, args.filename_format.format(seqid=seqid, TYPE='SCI')))
    fits_path_wht = Path(os.path.join(args.path, args.filename_format.format(seqid=seqid, TYPE='WHT')))
    if not fits_path_sci.exists():
        raise FileNotFoundError(f"FITS not found: {fits_path_sci}")
    if not fits_path_wht.exists():
        raise FileNotFoundError(f"FITS not found: {fits_path_wht}")

    sci, wht, orientat, target_xy_px = load_fits(fits_path_sci, fits_path_wht,
                                                   center_radec = (args.ra, args.dec),
                                                   return_orientat=True, return_center=True,
                                                 return_HDUL_only=False) # data:
    mask_invalid_values = (sci==0.) + (wht==0.)
    # SEP background config: No longer needed for BKG and RMS maps
    # bg_cfg = choose_sep_bg_config(args.sep_background, args.pixscale_arcsec, args.auto_r50_arcsec, verbose=verbose)
    # bkg_map, rms_map = background_analysis(sci, sci.mask, bg_cfg, method='sep') # method should be 'sep' or 'flat'
    mean_SC, BKG_sigma_clip, RMS_sigma_clip = sigma_clipped_stats(sci[sci!=0.], sigma=3.0) # SC stands for Sigma Clipping
    sci = np.ma.masked_array(sci, mask=sci == 0.)
    wht = np.ma.masked_array(wht, mask=wht == 0.)
    sci_bgsub = sci - BKG_sigma_clip # background subtracted image
    # Sigma clip BKG and RMS for reference & cross-checking
    # First crop with given ratio
    sci_bgsub, wht = crop_array_list_w_ratio([sci_bgsub, wht],
                                                   title_list = ['sci-bkg', 'wht'],
                                                   save_list=[True, True],
                                                   save_path=args.path, seqid=seqid,
                                                   ratio=args.first_crop_ratio, verbose=verbose,)

    # SEP detection + segmentation
    if verbose:
        print("Processing SEP... can take a while for big images...")
    objs, segmap = extract_with_sep(sci_bgsub, wht,
                                    deblend_nthresh=args.deblend_nthresh,
                                    deblend_cont=args.deblend_cont,
                                    detect_thresh_sigma=args.detect_thresh_sigma,
                                    minarea=args.minarea,
                                    return_segmap=True)
    # obj contains information of all the detedcted objects (x and y coordiantes, major and minor axis, npix, etc)
    if verbose:
        print(f"SEP detected the following number of objects: {len(objs)}")
        # print(f"SEP fields: {list(objs.dtype.names)}")
    # h, w = sci_bgsub.shape
    # target_xy_px = (w/2.0, h/2.0) # this is assuming that all targets are at the center, which is not the case for
    # tiles (mosaic is all centered), so target_xy_px needs to be given.
    if verbose:
        print("Checking the target from SEP... can take a while for big images...")
    label, rec, dist = pick_target_label(objs, segmap, target_xy_px, verbose=verbose)
    if verbose:
        print("target check done, now plotting...")
    fig_hw_unit_inch = 5
    fig, axes = plt.subplots(1, 5, figsize=(fig_hw_unit_inch*5, fig_hw_unit_inch))
    axes[0].set_title("SCI")
    axes[1].set_title("SCI - BKG")
    axes[2].set_title("WHT")
    axes[3].set_title("1/sqrt(WHT)")
    # axes[4].set_title("BKG from SEP (unused)")
    # axes[5].set_title("RMS from SEP (unused)")
    extent = None
    # extent = np.array([-sci.shape[1] / 2., sci.shape[1] / 2., -sci.shape[0] / 2.,
    #                    sci.shape[0] / 2.]) * args.pixscale_arcsec
    im_img, norm = AsinhStretchPlot(axes[0], sci, origin="lower", return_norm=True, extent=extent)
    plt.colorbar(mappable=im_img, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axvline(target_xy_px[0], ymin=0., ymax=sci.shape[0], color='w',
                     linewidth=0.5, linestyle='--')
    axes[0].axhline(target_xy_px[1], xmin=0., xmax=sci.shape[1], color='w',
                     linewidth=0.5, linestyle='--')
    im_img_bkgsub = axes[1].imshow(sci_bgsub, origin="lower", norm=norm, extent=extent)
    plt.colorbar(mappable=im_img_bkgsub, ax=axes[1], fraction=0.046, pad=0.04)
    im_wht = AsinhStretchPlot(axes[2], wht, origin='lower', extent=extent)
    plt.colorbar(mappable=im_wht, ax=axes[2], fraction=0.046, pad=0.04)
    im_wht_sqrt_inverse = AsinhStretchPlot(axes[3], 1/np.sqrt(wht), origin='lower', extent=extent)
    plt.colorbar(mappable=im_wht_sqrt_inverse, ax=axes[3], fraction=0.046, pad=0.04)
    # im_bkg_map = axes[4].imshow(bkg_map, origin="lower", cmap='gray', extent=extent)
    # plt.colorbar(mappable=im_bkg_map, ax=axes[4], fraction=0.046, pad=0.04)
    # im_rms_map = axes[5].imshow(rms_map, origin="lower", cmap='gray', extent=extent)
    # plt.colorbar(mappable=im_rms_map, ax=axes[5], fraction=0.046, pad=0.04)

    # fig.savefig(os.path.join(args.path, f"{seqid}-01-bg_and_rms.pdf"))
    # plt.show()
    # fig, ax = plt.subplots()
    # extent = np.array([-segmap.shape[1] / 2., segmap.shape[1] / 2., -segmap.shape[0] / 2.,
    #                    segmap.shape[0] / 2.]) * args.pixscale_arcsec
    im, cmap = draw_segmentation(axes[4], segmap, title='Segmentation', target_label=label,
                        outline = False, origin='lower', extent=extent)
    cbar = plt.colorbar(mappable=im, ax=axes[4], fraction=0.046, pad=0.04)
    cbar.locator = MaxNLocator(integer=True, nbins=5)
    cbar.update_ticks()
    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x (px)')
    fig.tight_layout()
    fig.suptitle(f"{seqid} large cutout")
    fig.savefig(os.path.join(args.path, f"{seqid}-02-bg_and_segmap.pdf"))
    plt.show() if debug else plt.close(fig)
    # list objects overlapping the exact center or whose (2.5×Kron) ellipse contains the center
    # overlapping_labels = objects_covering_center(segmap, center_xy, objs)

    # img_cropped_masked has all pixels including background within the region, except nan (out-of-region) and other objects
    # img_cropped_masked2 has only the pixels that are sorted as the target object by SEP & indicated by segmap
    img_cropped_masked2 = np.ma.masked_array(sci_bgsub, mask=segmap != label, fill_value=np.nan)
    # total definition (A/B/C)
    # total_flux, a_mom, b_mom, q = compute_total_flux(img_bgsub, rec, segmap, label, center_xy, args.total_def)
    total_flux = np.sum(img_cropped_masked2)

    q = rec['b'] / rec['a']
    flux_cumulative, R_1d_sorted = cumulative_ellipse_curve(img_cropped_masked2, rec, verbose=verbose,
                                                            plot=False)
    # note that flux_cumulative max value might not match exactly with total_flux, but it should be similar.
    assert np.isclose(total_flux, flux_cumulative.max(), rtol=1e-3)
    prs = find_percent_radii(flux_cumulative, R_1d_sorted, q=q,
                             percents=tuple(p / 100.0 for p in args.radii))
    if verbose:
        print(f"Percent radii for {args.radii} (in the unit of pixels; elliptical radius R: R^2=q*x^2 + y^2/q) ")
        print(prs)

    # cropped and masked for region of interest (where the object is; crop with xmin, xmax, ymin, ymax while updating the rec.)
    mask = (segmap != label) * (segmap != 0) + mask_invalid_values
    title_list = ['SCI-BKG', 'WHT', '1/sqrt(WHT)', 'Segmentation', 'MASK']
    sigma_clipped_values = {'mean': mean_SC, 'median': BKG_sigma_clip, 'stdev': RMS_sigma_clip}
    rms_from_wht = 1./np.sqrt(wht)
    [sci_bgsub_crop, wht_crop, rms_from_wht_crop, segmap_crop, mask_crop], rec_cropped = crop_target(
        [sci_bgsub, wht, rms_from_wht, segmap, mask], label=label, obj_rec=rec,verbose=verbose, plot=True,
        fig_savename=os.path.join(args.path, f"{seqid}-03-masked_and_cropped.pdf"),
        title_list=title_list, suptitle=str(seqid), sigma_clipped_values=sigma_clipped_values,
        plot_masked=True, plot_masked_sci_and_mask_inds=[0,-1], masked_title='Masked', debug=debug,
        crop_mode='minmax', radius=None,#prs['A99'],
        crop_factor=1.5)
    h, w = sci_bgsub_crop.shape
    sci_bgsub_crop_masked = np.ma.masked_array(sci_bgsub_crop, mask=mask_crop)

    # Text box with measurements
    a50 = prs.get(f"A{args.radii[0]}")
    a90 = prs.get(f"A{args.radii[1]}") if len(args.radii) > 1 else None
    a99 = prs.get(f"A{args.radii[2]}") if len(args.radii) > 2 else None
    req50 = prs.get(f"R{args.radii[0]}")
    req90 = prs.get(f"R{args.radii[1]}") if len(args.radii) > 1 else None
    req99 = prs.get(f"R{args.radii[2]}") if len(args.radii) > 2 else None

    text = [
        # f"SEP BG: mode={bg_cfg.mode}, BACK_SIZE={bg_cfg.back_size}, FILTERSIZE={bg_cfg.back_filtersize}",
        f"Deblend: nthresh={args.deblend_nthresh}, cont={args.deblend_cont}",
        f"Target label: {label}",
        f"Total flux: flux={total_flux:.3g}",
        f"Moments: a={rec_cropped['a']:.2f} px, b={rec_cropped['b']:.2f} px, q={q:.2f}",
        f"R{args.radii[0]}: a={a50:.2f} px, r_eq={req50:.2f} px",
    ]
    if a90 is not None:
        text.append(f"R{args.radii[1]}: a={a90:.2f} px, r_eq={req90:.2f} px")
    if a99 is not None:
        text.append(f"R{args.radii[2]}: a={a99:.2f} px, r_eq={req99:.2f} px")
    if verbose:
        print(text)
    # out_pdf = Path(os.path.join(args.path, f"{seqid}-cutout_{size_arcsec}_arcsec_sep.pdf"))
    # return summary row for CSV
    # med_bg, std_bg = sigma_clip_stats(bkg_map, None)
    # med_rms, std_rms = sigma_clip_stats(rms_map, None)

    # row: to be saved as a csv file
    row = {
        'seqid': seqid,
        # 'percent_radii': prs,
        'image_width': w,
        'image_height': h,
        'q': q,
        'theta': rec_cropped['theta'],
        'orientat': orientat,
        'target_label': label,
        #
        # 'BACK_SIZE': bg_cfg.back_size,
        # 'BACK_FILTERSIZE': bg_cfg.back_filtersize,
        'deblend_nthresh': args.deblend_nthresh,
        'deblend_cont': args.deblend_cont,
        'detect_thresh_sigma': args.detect_thresh_sigma,
        # 'labels_overlapping_center': ";".join(map(str, overlapping_labels)),
        # 'bg_map_median': med_bg,
        # 'bg_map_std_of_map': std_bg,
        # 'rms_map_median': med_rms,
        # 'rms_map_std_of_map': std_rms,
        'total_flux': total_flux,
        **rec_cropped, **prs, # R50, etc.
        # 'bkg_map_median': np.median(bkg_map),
        # 'rms_map_median': np.median(rms_map),
        'BKG_sigma_clip': BKG_sigma_clip,
        'RMS_sigma_clip': RMS_sigma_clip ,
        'label': label,
        'distance_to_target (px)': dist[0],
        'distance_to_next_closest (px)': dist[1],
        'distance_ratio (healthy if << 1. or nan)': dist[0]/dist[1],
    }


    cropped_data = {'sci_bgsub_crop':sci_bgsub_crop.filled(),
                    'wht_crop': wht_crop,
                    # 'bkg_map_crop_from_SEP':bkg_map_crop,
                    # 'rms_map_crop_from_SEP':rms_map_crop,
                    'segmap_crop':segmap_crop,
                    'mask_crop': mask_crop,
                    }
    return row, cropped_data

# ----------------------------
# Main routine & Configuration
# ----------------------------

def main(config, bad_seq_id_file_csv, verbose=False, debug=False):
    args = SimpleNamespace(**config)
    # grab CSV file too and use their R50 value
    csv_file_format_catalog = 'cosmos_sample*.csv'
    csv_file_list = glob.glob(os.path.join(args.path, csv_file_format_catalog))
    if len(csv_file_list) > 1:
        while True:
            for i, csv_file in enumerate(csv_file_list):
                print(f"{i}: {csv_file}")
            play_sound_list(frequency_list=[220, 440, 880, 1760], duration_list=[0.5, 0.5, 0.5, 0.5])
            ind_selected = input("#### which CSV to go with, by putting an integer (e.g. 0, 1, 2): ")
            try:
                # Criteria to match
                cri1 = float(ind_selected) == int(ind_selected)
                cri2 = int(ind_selected) <= (len(csv_file_list) - 1)
                cri3 = int(ind_selected) >= 0
                if cri1 and cri2 and cri3:
                    ind_selected = int(ind_selected)
                    csv_file_catalog = csv_file_list[ind_selected]
                    break
            except ValueError:
                print("Please enter a valid integer.")
            except Exception as e:
                raise e
    elif len(csv_file_list) == 1:
        csv_file_catalog = csv_file_list[0]
    else:
        raise ValueError("len(csv_file_list) is not an ")
    df_catalog = pandas.read_csv(csv_file_catalog)
    print("Catalog CSV file loaded!")
    #
    df_bad_data = pandas.read_csv(bad_seq_id_file_csv)
    bad = df_bad_data['bad'] # 1 if marked as bad, NaN if not
    num_bad = np.sum(bad==1.); num_notbad = np.sum(np.isnan(bad) + (bad==0.))
    if (num_bad + num_notbad) != len(df_bad_data):
        play_sound_list()
        print("Following seqid's have neither 1 nor NaN nor 0! Check!!!")
        seq_to_check = list(df_bad_data['seqid'][(bad!=1.) * (~np.isnan(bad)) * (bad!=0.)])
        print(seq_to_check)
        play_sound_list()
    df_bad = df_bad_data[bad==1.]
    if verbose:
        print("Following samples will be omitted in analysis because they are marked as bad")
        print(df_bad)
    sids_bad = list(df_bad['seqid'])
    #
    if args.seqid_list == 'all' or args.seqid_list == []:
        args.seqid_list = list(df_catalog['sequentialid'])
    # Determine starting index for processing
    start_index = args.start_index if hasattr(args, "start_index") else None
    start_seqid = args.start_seqid if hasattr(args, "start_seqid") else None

    if start_seqid is not None:
        seq_col = [str(int(s)) for s in args.seqid_list]
        if str(start_seqid) in seq_col:
            if start_index is not None:
                warnings.warn(
                    f"start_index is given as {start_index}, but start_seqid is given as {start_seqid}. "
                    f"start_index will be ignored."
                )
            start_index = seq_col.index(str(start_seqid))
        else:
            warnings.warn(
                f"start_seqid={start_seqid} not found in args.seqid_list; starting at index={start_index if start_index is not None else 0}"
            )
    if start_index is None:
        start_index = 0
    # Prepare error log for CSV upserts
    upsert_errors = []

    t0 = time.perf_counter()
    total_n = len(args.seqid_list)

    for count, sid in enumerate(args.seqid_list[start_index:], start=start_index):
        sid = int(sid) if type(sid) is not int else sid
        elapsed_time_reporter(t0, count, total_n, seq_id=sid)

        index = df_catalog[
            df_catalog['sequentialid'] == sid].index.tolist()  # find the index that corresponds with the target sid
        assert len(index) == 1  # only one index that has the sequential id we are looking for

        if sid in sids_bad:
            if verbose:
                print("Omitting seqid:", sid, " because it is marked as bad!")
            continue

        config_each = {
            'first_crop_ratio': 1.,
            'auto_r50_arcsec': df_catalog['r50'][index] * args.pixscale_arcsec,
            # typical object's R50 size: its 8 times will be used for background estimation.
            'ra': df_catalog['ra'][index],
            'dec': df_catalog['dec'][index],
        }
        for key in config_each.keys():
            setattr(args, key, config_each[key])

        try:
            # print("Processing", sid)
            row, cropped_data = process_cutout(sid, args, debug=debug, verbose=verbose)

            # NEW: upsert this row into sep_summary.csv immediately
            if args.csv:
                csv_path = os.path.join(args.path, "sep_summary.csv")
                try:
                    upsert_sep_summary_row(row, csv_path)
                except Exception as e:
                    msg = f"[csv upsert error] seqid={sid}: {e}"
                    print(msg)
                    upsert_errors.append(msg)

            # Save cropped data to HDF5
            filename = os.path.join(args.path, f"{sid}-cropped.hdf5")
            with h5py.File(filename, "w") as data_file:
                for key in cropped_data.keys():
                    data_file.create_dataset(key, data=cropped_data[key])

        except FileNotFoundError as e:
            print(f"[skip] {e}")

    # Report any CSV upsert errors at the end
    if upsert_errors:
        print("\nThe following CSV upsert errors occurred during the run:")
        for msg in upsert_errors:
            print("  ", msg)


if __name__ == '__main__':
    # >>> Edit these defaults instead of passing CLI args <<<

    config = {
        'path': '../data/HDUL_test7-big100',
        'filename_format': "{seqid}-{TYPE}.fits",
        # File naming: expects {seqid}-cutout_{size}_arcsec.fits in cwd
        'start_index': None,
        'start_seqid': None,
        'seqid_list': [],  # 'all' or [] for everything; or, list of numbers like [131286, 131486]
        # Pixel scale and typical half-light radius (arcsec)
        'pixscale_arcsec': 0.03,
        # Percent-light radii settings
        'radii': [50, 90, 99],  # the list of values to calculate R[n]; R50, R90, R99.
        # Output: write CSV summary?
        'csv': True,
    }

    sep_config = {# SEP detection/deblend parameters
        'deblend_nthresh': 32, # DEBLEND_NTHRESH : the number of thresholds the intensity range is devided up in. 32 is the most common number.
        'deblend_cont': 1e-4, # Minimum contrast ratio used for object deblending. Default is 0.005. To entirely disable deblending, set to 1.0.
        'detect_thresh_sigma': 3.0, # 3.0  # Check: https://sep.readthedocs.io/en/stable/api/sep.extract.html; when err
        # map is given, the interpretation changes so it needs to be updated.
        'minarea': 10,  #20 # minimum area; default 5 pixels
        }

    config.update(sep_config)

    bad_seq_id_file_csv = '../data/bad_seq_ids.csv'

    main(config, bad_seq_id_file_csv, verbose=False, debug=False)
