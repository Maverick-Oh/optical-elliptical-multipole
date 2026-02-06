"""
Add SEP (Source-Extractor-in-Python) background/segmentation options to the HST cutout explorer.

Key features
------------
- Uses SEP to estimate a tiled background map (BACK_SIZE/BACK_FILTERSIZE) and subtract it.
- Middle-panel switch ("--middle"): sigma-clipped (existing), SEP bg-sub view with map inset,
  or SEP segmentation map with R50/R90/R99 overlays.
- Elliptical R50/R90/R99 on the target-only pixels (others masked via segmentation).
  * Percent-light radii are computed with respect to a configurable "total" definition:
    A: within 2.5×Kron (default), B: within segmentation footprint, C: within 4×Kron.
  * Reports semi-major radii (a50/a90/a99) and equivalent-circular radii r_eq = a*sqrt(q).
- Deblending enabled (deblend_nthresh, deblend_cont configurable).
- Pure zeros (mosaic boundaries) are ignored in stats and painted blue in plots.
- Two background configs: "default" (BACK_SIZE=64, BACK_FILTERSIZE=3) or "auto"
  (BACK_SIZE set from pixel scale and typical R50; see --auto-r50-arcsec / --pixscale-arcsec).
- CSV log of background medians/RMS, mesh params, target label, deblend params, and R50/R90/R99 per cutout size.

Dependencies: numpy, astropy, matplotlib, sep, scipy

Example
-------
python cutout_size_and_noise_test_sep.py \
  --seqid 32836 \
  --sizes 20 40 80 160 180 \
  --middle sep-seg \
  --sep-background auto \
  --radii 50 90 99 \
  --total-def A \
  --pixscale-arcsec 0.03 \
  --auto-r50-arcsec 0.30

This script assumes your FITS cutouts are named like:
  {seqid}-cutout_{size}_arcsec.fits
and will save figures as:
  {seqid}-cutout_{size}_arcsec_sep.pdf

Notes on SEP params
-------------------
- deblend_nthresh: number of multi-threshold levels used to split touching objects (typ. 16–64).
- deblend_cont: minimum contrast ratio between components to keep them separate (lower → more splitting; 0.001–0.01 common).
- BACK_SIZE: linear size (px) of the background mesh grid boxes (power of two often convenient).
- BACK_FILTERSIZE: median-filter window (in mesh cells) applied to the coarse background map (typ. 3) to suppress small-scale residuals.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import sep
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats, sigma_clip
import warnings

from matplotlib.patches import Rectangle

# from analysis.load_COSMOS import seq_id
import os
from optical_elliptical_multipole.plotting.plot_tools import AsinhStretchPlot
from optical_elliptical_multipole.plotting.plot_tools import draw_segmentation
from matplotlib.ticker import MaxNLocator

sep.set_extract_pixstack(1e6) # max number of pixels
sep.set_sub_object_limit(5e3)
# ----------------------------
# Utilities
# ----------------------------

from tools_misc import radec_to_pixel

def load_fits(path_sci: Path,
              path_wht: Path|None = None,
              return_orientat=True,
              return_center=True, center_radec=None,
              return_HDUL_only=False) -> Tuple[np.ndarray, np.ndarray]:
    """Load FITS, return masked array image (float32).
    Converts NaNs to 0 before zero-mask detection, then restores NaNs to 0 afterwards.
    """
    if return_HDUL_only:
        if path_wht is None:
            return fits.open(path_sci)
        else:
            return fits.open(path_sci), fits.open(path_wht)
    else:
        with fits.open(path_sci) as hdul:
            sci = hdul[0].data.astype(np.float32)
            if return_center:
                ra, dec = center_radec
                center_xy = radec_to_pixel(hdul[0], ra=ra, dec=dec)
        # assert np.sum(data==0.) == 0 # check if there is any true zeros before converting nan to zeros
        # data = np.nan_to_num(data, copy=False, nan=0.0)
        assert np.sum(np.isnan(sci)) == 0
        # nan_mask = np.isnan(sci)
        # sci = np.ma.masked_array(sci, mask=nan_mask, fill_value=np.nan) # make data masked array
        if return_orientat:
            orientat = hdul[0].header['ORIENTAT']
        else:
            orientat = 0.0 # Default/Ignored
        #
        if path_wht is not None:
            with fits.open(path_wht) as hdul:
                wht = hdul[0].data.astype(np.float32)
            if return_orientat:
                assert np.isclose(orientat, hdul[0].header['ORIENTAT'], 1e-4) # check if orient at matches between sci and wht
            #
            my_return = (sci, wht)
            if return_orientat:
                my_return += (orientat, ) # comma is important!
            if return_center:
                my_return += (center_xy, )
            return my_return
        else:
           return sci
        # if load_wht:
        #     # for COSMOS Tiles (both SCI and WHT images, and orientation )
        #     with fits.open(path_sci) as hdul:
        #         assert 'SCI' in hdul.keys()
        #         assert 'WHT' in hdul.keys()
        #         assert len(hdul['SCI']) == 1
        #         assert len(hdul['WHT']) == 1
        #         sci = hdul['SCI'][0].data
        #         wht = hdul['WHT'][0].data
        #         assert hdul['SCI'][0].header['ORIENTAT'] == hdul['WHT'][0].header['ORIENTAT']
        #         orientat = hdul['SCI'][0].header['ORIENTAT']
        #         nan_mask = np.isnan(sci) + np.isnan(wht)
        #         sci = np.ma.masked_array(sci, mask=nan_mask, fill_value=np.nan)
        #         wht = np.ma.masked_array(wht, mask=nan_mask, fill_value=np.nan)
        #     if return_orientat:
        #         return sci, wht, orientat
        #     else:
        #         return sci, wht

# def sigma_clip_stats(arr: np.ndarray, mask: np.ndarray | None = None, nsig=3.0, iters=None) -> Tuple[float, float]:
#     """Simple sigma-clipped median/std for reference/plotting.
#     Ignores masked pixels.
#     """
#     x = arr.copy()
#     if mask is not None:
#         x = x[~mask]
#     x = x[np.isfinite(x)]
#     if x.size == 0:
#         return 0.0, 0.0
#     for _ in range(iters):
#         med = np.median(x)
#         std = np.std(x)
#         sel = np.abs(x - med) < nsig * (std if std > 0 else 1.0)
#         x = x[sel]
#         if x.size == 0:
#             break
#     return (float(np.median(x)) if x.size else 0.0, float(np.std(x)) if x.size else 0.0)

@dataclass
class SepBgConfig:
    mode: str  # 'default' or 'auto'
    back_size: int
    back_filtersize: int

def choose_sep_bg_config(mode: str, pixscale_arcsec: float, auto_r50_arcsec: float, verbose: bool = True) -> SepBgConfig:
    # Background configuration
    if mode == 'default':
        return SepBgConfig(mode='default', back_size=64, back_filtersize=3)
    elif mode=='auto':
        pass
    else:
        raise ValueError(f'Unknown mode {mode}; mode must be "default" or "auto".')
    # auto mode
    r50_pix = auto_r50_arcsec / pixscale_arcsec  # ~10 px for 0.3" at 0.03"/px
    # choose BACK_SIZE ≈ 8×R50 rounded to nearest power of two, minimum 64
    back_size_raw = int(max(8 * r50_pix, 64))
    # if nearest power of two is needed
    # pow2 = 1 << (int(round(np.log2(back_size_raw))))
    # back_size = int(max(64, pow2))
    if verbose:
        print(f"r50 for bkg removal: {r50_pix:.1f} pixels; back_size_raw: {back_size_raw:.1f} pixels; used power-of-2 "
              f"back_size: {back_size_raw:.1f} pixels.")
        # reference for BACK_SIZE: https://sextractor.readthedocs.io/en/latest/Background.html
    return SepBgConfig(mode='auto', back_size=back_size_raw, back_filtersize=3)

# ----------------------------
# SEP pipeline pieces
# ----------------------------

def background_analysis(image: np.ndarray, nan_mask: np.ndarray, cfg: SepBgConfig, method:str = 'sigma_clip') \
        -> Tuple[np.ndarray, np.ndarray]|Tuple[float, float]:
    """Compute SEP background map and RMS, respecting zeros_mask as invalid.
    Returns (bkg_map, rms_map, bkg_object).
    """
    if method == 'sep':
        mask = nan_mask.astype(np.bool_)
        # SEP requires C-contiguous float32
        img = np.ascontiguousarray(image.astype(
            np.float32))  # ensuring that the array elements are stored in a single, unbroken block of memory in C-style (row-major) order
        bkg = sep.Background(img, mask=mask, bw=cfg.back_size, bh=cfg.back_size,
                             fw=cfg.back_filtersize, fh=cfg.back_filtersize)
        # .back(): Create an array of the background. .rms(): Create an array of the background rms. Reference: https://sep.readthedocs.io/en/v1.0.x/api/sep.Background.html
        back = bkg.back(); back = np.ma.masked_array(back, mask=mask, fill_value=np.nan)
        rms = bkg.rms(); rms = np.ma.masked_array(rms, mask=mask, fill_value=np.nan)
        return back, rms
    elif method == 'sigma_clip':
        mean, median, stdev = sigma_clipped_stats(image, mask=image.mask, sigma=3.0)
        return median, stdev
    else:
        raise ValueError(f"unknown method '{method}'--it should be one of 'sep' or 'sigma_clip'")

def extract_with_sep(sci_bgsub: np.ma.core.MaskedArray,
                     wht: np.ma.core.MaskedArray,
                     deblend_nthresh=32,
                     deblend_cont=0.005,
                     detect_thresh_sigma=1.5,
                     minarea=5,
                     return_segmap=True, ):
    """Run SEP detection+deblend, optionally returning a segmentation map."""
    if type(sci_bgsub) is np.ndarray:
        sci_bgsub = np.ma.masked_array(sci_bgsub, mask=None)
    if type(wht) is np.ndarray:
        wht = np.ma.masked_array(wht, mask=None)
    mask = sci_bgsub.mask + wht.mask
    img = np.ascontiguousarray(sci_bgsub.astype(np.float32))
    # threshold in sigma: compute from robust std
    # _, med, std = sigma_clipped_stats(img, mask)
    # Threshold pixel value for detection. If an err or var array is not given, this is interpreted as an absolute threshold. If err or var is given, this is interpreted as a relative threshold: the absolute threshold at pixel (j, i) will be thresh * err[j, i] or thresh * sqrt(var[j, i]).
    while True:
        try:  # most oens get sorted out this way, but some extreme cases (e.g. 5519) needs special handling to
            # increase the active object pixels
            objs, segmap = sep.extract(img, thresh=detect_thresh_sigma, var=1./wht.filled(), mask=mask,
                                       minarea=minarea,
                                       deblend_nthresh=deblend_nthresh,
                                       deblend_cont=deblend_cont,
                                       clean=True,
                                       segmentation_map=return_segmap)
            break
        except Exception as e:
            print(e)  # Exception: internal pixel buffer full: The limit of 1000000 active object pixels over the detection threshold was reached. Check that the image is background subtracted and the detection threshold is not too low. If you need to increase the limit, use set_extract_pixstack.
            if 'active object pixels over the detection threshold was reached' in e.args[0]:
                print(f"!!! Increashing the limit of active object pixels by factor of 10: from {sep.get_extract_pixstack():.0e} to {sep.get_extract_pixstack()*10:.0e} !!!")
                sep.set_extract_pixstack(sep.get_extract_pixstack()*10)
            else:
                raise NotImplementedError()
    sep.set_extract_pixstack(int(1e6))
    return objs, segmap

def pick_target_label(objs, segmap: np.ndarray, target_xy: Tuple[float, float], verbose: bool = True) -> Tuple[int,
Dict, list]:
    """Choose target label as the object whose centroid is closest to cutout center.
    Returns (label, object_record_dict).
    """
    if len(objs) == 0:
        return 0, {}
    cx, cy = target_xy
    # objs is a structured array; convert minimal fields
    dx = objs['x'] - cx
    dy = objs['y'] - cy
    dr_sq = dx*dx + dy*dy
    if len(dr_sq) >1: # more than one particles detected
        idx1, idx2 = np.argsort(dr_sq)[0:2] # two indices; idx1 is the closest target and idx2 is the next closest target
        dist1, dist2 = np.sqrt(np.array([ dr_sq[idx1], dr_sq[idx2] ]))
    elif len(dr_sq) == 1:
        idx1 = np.argmin(dr_sq)
        dist1 = np.sqrt(np.array([dr_sq[idx1]])); idx2=np.nan; dist2 = np.nan
    else:
        raise ValueError(f"len(dr_sq): {len(dr_sq)}")
    dist1 = float(dist1); dist2 = float(dist2)
    if verbose:
        print(f"Target index: {idx1} with distance from the center: {dist1:.1f} pixels.")
        print(f"Next closest index: {idx2} with distance from the center: {dist2:.1f} pixels.")
    if dist1 > 10:
        print(f"!!!!! distance={dist1:.1f} pixels too large! something went wrong! !!!!!")
    # segmap labels are 1..N in SEP order
    label = idx1 + 1
    rec = {k: objs[k][idx1].item() if np.ndim(objs[k]) == 1 else objs[k][idx1] for k in objs.dtype.names}
    dist = [dist1, dist2]
    return label, rec, dist

# def objects_covering_center(segmap: np.ndarray, center_xy: Tuple[float, float], objs) -> List[int]:
#     """Return labels that include the exact center pixel OR whose 2.5×(a,b) ellipse contains the center.
#     Uses simple 2.5×(a,b) (no Kron) to avoid calling kron_radius here.
#     """
#     h, w = segmap.shape
#     cx, cy = center_xy
#     xi = int(np.clip(round(cx), 0, w-1))
#     yi = int(np.clip(round(cy), 0, h-1))
#     labels = set()
#     lab_center = int(segmap[yi, xi])
#     if lab_center > 0:
#         labels.add(lab_center)
#     # Simple ellipse inclusion test with 2.5×(a,b)
#     for i in range(len(objs)):
#         a = float(objs['a'][i])
#         b = float(objs['b'][i])
#         theta = float(objs['theta'][i])
#         x0 = float(objs['x'][i])
#         y0 = float(objs['y'][i])
#         if not (a > 0 and b > 0):
#             continue
#         dx = cx - x0
#         dy = cy - y0
#         ct = np.cos(theta)
#         st = np.sin(theta)
#         xr = dx*ct + dy*st
#         yr = -dx*st + dy*ct
#         ak = 2.5 * a
#         bk = 2.5 * b
#         val = (xr/ak)**2 + (yr/bk)**2
#         if val <= 1.0:
#             labels.add(i+1)
#     return sorted(labels)

# ----------------------------
# Percent-light radii (elliptical)
# ----------------------------
def crop_array_list_w_ratio(img_list: list,
                            title_list: list[str]|None = None,
                            save_list: list[bool]|None = None,
                            save_path: str = '.',
                            seqid: int|str=None,
                            verbose: bool = True,
                            **kwargs):
    if kwargs['ratio'] >= 1.:
        if verbose:
            print("ratio greater than 1.; no cropping.")
        return img_list
    list_return = []
    if title_list is None:
        title_list = len(img_list)*[None]
    if save_list is None:
        save_list = len(img_list)*[None]
    for img, title, save in zip(img_list, title_list, save_list):
        img_cropped, lims = crop_array_w_ratio(img, title=title, save=save, save_path=save_path, seqid=seqid, **kwargs)
        list_return.append(img_cropped)
    return list_return

def crop_array_w_ratio(img: np.ma.core.MaskedArray,
                       save=True,
                       save_path='.',
                       seqid=None,
                       ratio=0.5,
                       center=None,  # (x, y) in pixels; defaults to image center
                       verbose=True,
                       plot=True, title=None):
    """
    Crop a masked image around `center` by a designated ratio.

    Parameters
    ----------
    img : np.ma.MaskedArray
        Input masked image (2D).
    ratio : float or (float, float)
        Fraction of the original size to keep.
        - float r: keeps r*H by r*W (centered)
        - (ry, rx): keeps ry*H by rx*W
        Must be (0, 1] for each dimension.
    center : (float, float) or None
        (x, y) center in pixel coordinates of the crop. If None, use image center.
    verbose : bool
        Print crop info.
    plot : bool
        If True, show a figure with log10(img) and the crop box, plus the cropped view.

    Returns
    -------
    cropped : np.ma.MaskedArray
        Cropped masked image.
    slices  : tuple
        (ymin, ymax, xmin, xmax) integer bounds used for the crop.
    """
    if title is None:
        title='img'
    if img.ndim != 2:
        raise ValueError("crop_array expects a 2D masked array")
    H, W = img.shape

    # Normalize ratio input
    if isinstance(ratio, (tuple, list, np.ndarray)):
        if len(ratio) != 2:
            raise ValueError("ratio tuple must be (ry, rx)")
        ry, rx = float(ratio[0]), float(ratio[1])
    else:
        ry = rx = float(ratio)

    if not (0 < ry <= 1.0 and 0 < rx <= 1.0):
        raise ValueError("ratio must be in (0, 1] for each dimension")

    # Center
    if center is None:
        cx, cy = (W / 2.0), (H / 2.0)
    else:
        cx, cy = float(center[0]), float(center[1])

    # Desired crop sizes (at least 1 pixel)
    new_h = max(1, int(round(H * ry)))
    new_w = max(1, int(round(W * rx)))

    # Compute integer bounds (clip to image)
    ymin = int(round(cy - new_h / 2.0))
    ymax = ymin + new_h
    xmin = int(round(cx - new_w / 2.0))
    xmax = xmin + new_w

    # Clip and re-adjust to maintain size
    if ymin < 0:
        ymax -= ymin
        ymin = 0
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    if ymax > H:
        shift = ymax - H
        ymin = max(0, ymin - shift)
        ymax = H
    if xmax > W:
        shift = xmax - W
        xmin = max(0, xmin - shift)
        xmax = W

    # Final safety: ensure at least 1 pixel
    if ymax <= ymin or xmax <= xmin:
        raise RuntimeError("Invalid crop bounds after clipping")

    cropped = img[ymin:ymax, xmin:xmax]

    if verbose:
        print(f"crop_array: input shape={img.shape}, ratio=({ry:.3f}, {rx:.3f}), "
              f"center=({cx:.1f}, {cy:.1f}) → crop=({ymin}:{ymax}, {xmin}:{xmax}) "
              f"shape={cropped.shape}")

    if plot:
        # AsinhStretch makes things brighter: https://docs.astropy.org/en/stable/api/astropy.visualization.AsinhStretch.html
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Prepare full image array (masked → NaN)
        # Left: full image with crop rectangle
        ax0 = axes[0]
        im0, norm = AsinhStretchPlot(ax0, img, title='img with crop box', origin='lower', cmap='viridis',
                                     return_norm=True)
        ax0.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                edgecolor='r', facecolor='none', linewidth=1.0))
        ax0.set_title(f"{title} with crop box")
        ax0.set_xticks([])
        ax0.set_yticks([])
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Right: cropped image with same normalization
        ax1 = axes[1]
        crop_arr = np.where(cropped.mask, np.nan, cropped.filled(np.nan))
        im1 = AsinhStretchPlot(ax1, crop_arr, title='cropped img', origin='lower', cmap='viridis', norm=norm)
        ax1.set_title("cropped")
        ax1.set_xticks([])
        ax1.set_yticks([])
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        fig.tight_layout()
        if save:
            filename = os.path.join(save_path, f"{seqid}-02-{title}-first_crop.pdf")
            fig.savefig(filename)
        plt.show()

    return cropped, (ymin, ymax, xmin, xmax)

def crop_array_list_w_index(im_list, ymin, ymax, xmin, xmax):
    return_list = []
    for im in im_list:
        return_list.append(crop_array_w_index(im, ymin, ymax, xmin, xmax))
    return return_list

def crop_array_w_index(im, ymin, ymax, xmin, xmax):
    # includes all edges (ymin, ymax, xmin, xmax).
    return im[ymin:ymax + 1, xmin:xmax + 1]

def crop_minmax_calc_symmetric(rec, factor=1.0):
    # This calculates symmetric crop range
    x_center = np.round(rec['x'])
    x_width1 = x_center - rec['xmin']; x_width2 = rec['xmax'] - x_center; assert x_width1 > 0; assert x_width2>0
    x_width = max(x_width1, x_width2)
    x_min = np.round(x_center - factor * x_width).astype(int); x_max = np.round(x_center + factor * x_width).astype(int)
    #
    y_center = np.round(rec['y'])
    y_width1 = y_center - rec['ymin']; y_width2 = rec['ymax'] - y_center; assert y_width1 > 0; assert y_width2>0
    y_width = max(y_width1, y_width2)
    y_min = np.round(y_center - factor * y_width).astype(int); y_max = np.round(y_center + factor * y_width).astype(int)
    return x_min, x_max, y_min, y_max

def crop_radius_calc_symmetric(rec, radius, factor):
    # rec: object record
    # radius: radius in pixels
    # factor: factor to multiply on radius
    # rec['x'] and rec['y'] are weighted centroid computed using intensity; xcpeak and ycpeak can be used if
    x_cut_lo = np.round(rec['x'] - radius*factor); x_cut_hi = np.round(rec['x'] + radius*factor)
    y_cut_lo = np.round(rec['y'] - radius*factor); y_cut_hi = np.round(rec['y'] + radius*factor)
    return int(x_cut_lo), int(x_cut_hi), int(y_cut_lo), int(y_cut_hi)

def crop_target(map_list: List[np.ndarray],
                label: int, obj_rec: Dict, verbose:bool=True, plot:bool=True,
                fig_savename:str='crop_target.pdf', title_list:list|None=None,
                pixscale_arcsec=0.03, suptitle=None, sigma_clipped_values=None,
                plot_masked=True, plot_masked_sci_and_mask_inds=[0,-1], masked_title='Masked', debug=False,
                crop_mode='minmax', crop_factor=1.5, radius=None):
    if title_list is None:
        title_list = [None]*len(map_list)
    # boundaries in pixels
    if crop_mode=='minmax': # uses object x_min and x_max to crop
        x_cut_lo, x_cut_hi, y_cut_lo, y_cut_hi = crop_minmax_calc_symmetric(obj_rec, factor=crop_factor)
        if radius is not None:
            warnings.warn(f"crop_mode=='minmax', so radius (given as {radius}) is ignored.")
    elif crop_mode=='given_radius': # crop with given radius * crop_factor around ; given radius can be a99.
        x_cut_lo, x_cut_hi, y_cut_lo, y_cut_hi = crop_radius_calc_symmetric(obj_rec, radius, factor=crop_factor)
    else:
        raise ValueError(f"crop_mode={crop_mode} not recognized")
    x_cut_lo, x_cut_hi = np.clip([x_cut_lo, x_cut_hi], 0, map_list[0].shape[1])
    y_cut_lo, y_cut_hi = np.clip([y_cut_lo, y_cut_hi], 0, map_list[0].shape[0])
    #
    # mask out other objects; but do not mask out background (segmap zeros)
    map_list_cropped = crop_array_list_w_index(map_list, y_cut_lo, y_cut_hi, x_cut_lo, x_cut_hi)
    #
    obj_rec_for_cropped = obj_rec.copy()
    obj_rec_for_cropped['x'] -= x_cut_lo
    obj_rec_for_cropped['xmin'] -= x_cut_lo
    obj_rec_for_cropped['xmax'] -= x_cut_lo
    obj_rec_for_cropped['xpeak'] -= x_cut_lo
    obj_rec_for_cropped['y'] -= x_cut_lo
    obj_rec_for_cropped['ymin'] -= y_cut_lo
    obj_rec_for_cropped['ymax'] -= y_cut_lo
    obj_rec_for_cropped['ypeak'] -= x_cut_lo
    if verbose:
        print(f"image cropped from: {map_list[0].shape} to {map_list_cropped[0].shape}")
        n_masked_pixels = np.sum(map_list_cropped[-1])
        print(f"masked pixels in the cropped image: {n_masked_pixels} ("
              f"{n_masked_pixels/np.prod(map_list_cropped[0].shape)*100} %)")
    if plot:
        n_col = len(map_list_cropped)+1 if plot_masked else len(map_list_cropped)
        fig, axes = plt.subplots(1, n_col, figsize=(3*len(map_list_cropped), 3))
        # axes[0].set_title("image"); axes[1].set_title("masked"); axes[2].set_title("masked & cropped")
        for i, map in enumerate(map_list_cropped):
            extent = np.array([-map.shape[1] / 2., map.shape[1] / 2., -map.shape[0] / 2.,
                               map.shape[0] / 2.]) * pixscale_arcsec
            if map.dtype == bool or title_list[i]=='MASK':
                # It is supposed to be mask
                im = axes[i].imshow(map, origin='lower', extent=extent, vmin=0, vmax=1)
                plt.colorbar(mappable=im, ax=axes[i], fraction=0.046, pad=0.04)
            elif np.issubdtype(map.dtype, np.integer) or title_list[i]=='Segmentation':
                # It is supposed to be segmentation, so use segmentation plot
                im, cmap = draw_segmentation(axes[i], map, target_label=label,
                                             origin='lower', extent=extent, outline=False,)
                cbar = plt.colorbar(mappable=im, ax=axes[i], fraction=0.046, pad=0.04)
                cbar.locator = MaxNLocator(integer=True, nbins=5)
                cbar.update_ticks()
            else:
                im = AsinhStretchPlot(axes[i], map, origin='lower', extent=extent)
                plt.colorbar(mappable=im, ax=axes[i], fraction=0.046, pad=0.04)
            axes[i].set_title(title_list[i])
        if plot_masked:
            sci = map_list_cropped[plot_masked_sci_and_mask_inds[0]]
            mask = map_list_cropped[plot_masked_sci_and_mask_inds[-1]]
            masked = np.ma.masked_array(sci, mask=mask)
            im = AsinhStretchPlot(axes[-1], masked, origin='lower', extent=extent)
            plt.colorbar(mappable=im, ax=axes[-1], fraction=0.046, pad=0.04)
            axes[-1].set_title(masked_title)
        for ax in axes:
            # ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect('equal')
            ax.set_xlabel('x (arcsec)')
        fig.suptitle(f'{suptitle} small cutout')
        if sigma_clipped_values is not None:
            median_sc = sigma_clipped_values['median'] # values from sigma clipping
            stdev_sc = sigma_clipped_values['stdev'] # values from sigma clipping
        else:
            median_sc = np.nan
            stdev_sc = np.nan
        # values from source extractor; masking manually because np.median() ignores masks
        if 'BKG' in title_list:
            my_bkg = map_list_cropped[title_list.index('BKG')]; my_bkg = my_bkg.filled()[~my_bkg.mask]
            median_se = np.median(my_bkg)
        else:
            median_se = None
        if 'RMS' in title_list:
            my_rms = map_list_cropped[title_list.index('RMS')]; my_rms = my_rms.filled()[~my_rms.mask]
            stdev_se = np.median(my_rms)
        else:
            stdev_se = None
        if 'WHT' in title_list:
            my_stdev_wht = 1/np.sqrt(map_list_cropped[title_list.index('WHT')]); my_stdev_wht = my_stdev_wht.filled()[~my_stdev_wht.mask]
            stdev_wht = np.median(my_stdev_wht)
        else:
            stdev_wht = np.nan

        figtext_list = [f'σ-clipping BKG median: {median_sc:.1e} ',
                        f'σ-clipping RMS: {stdev_sc:.1e}']
        figtext_list[0] += f' ({median_sc/median_se:.1f} times of BKG median)' if median_se is not None else ''
        figtext_list[1] += f' ({stdev_sc/stdev_se:.1f} times of RMS median)' if stdev_se is not None else ''
        figtext_list[1] += f' ({stdev_sc/stdev_wht:.1f} times of 1/sqrt(WHT) median)' if stdev_wht is not None else ''
        figtext = "\n".join(figtext_list)
        plt.figtext(1.0, 1.0, figtext, ha='right', va='top')
        fig.tight_layout()
        fig.savefig(fig_savename)
        plt.show() if debug else plt.close(fig)
    return map_list_cropped, obj_rec_for_cropped

def sort_and_match(metric: np.ndarray, target: np.ndarray):
    # sort 2d numpy array metric and target such that A in the increasing order and B is sorted the same way as A.
    metric_1d = metric.flatten()
    target_1d = target.flatten()
    inds = metric_1d.argsort()
    metric_1d = metric_1d[inds]
    target_1d = target_1d[inds]
    return metric_1d, target_1d

def cumulative_ellipse_curve(img: np.ma.core.MaskedArray, obj_rec: Dict, verbose:bool = True, plot: bool = False) -> (
        np.ndarray):
    """Compute cumulative flux inside elliptical apertures with semi-major radii from a_grid.
    Ellipse: (a, b=q*a, theta) centered at center_xy.
    Returns array of cumulative sums for each a in a_grid.
    """
    # Check: https://sextractor.readthedocs.io/en/latest/Position.html
    # cxx,cyy,cxy = obj_rec['cxx'], obj_rec['cyy'], obj_rec['cxy'] #
    x_center, y_center = obj_rec['x'], obj_rec['y']
    # Elliptical Radius R squared (to sort out flux and add from center)
    ny, nx = img.shape
    x = np.arange(nx); y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    q = obj_rec['b']/obj_rec['a']
    theta = obj_rec['theta']
    R_sq = q*(np.cos(theta) * (X-x_center) + np.sin(theta) * (Y -y_center))**2 + \
            1/q*(-np.sin(theta) * (X-x_center) + np.cos(theta) * (Y-y_center) )**2
    # R_sq = cxx * (X-x_center)**2 + cyy * (Y-y_center)**2 + cxy * (X-x_center)*(Y-y_center)
    if plot:
        fig, ax = plt.subplots(figsize=(6,4))
        im_R = ax.imshow(np.sqrt(R_sq), origin='lower', cmap='gray')
        ax.set_title(f"Distance R (px) from ({x_center:.2f}, {y_center:.2f})")
        plt.contour(R_sq, levels=10)
        plt.colorbar(mappable=im_R)
        plt.show()
    # organize signal in the order of R squared
    R_sq_1d_sorted, img_1d_sorted = sort_and_match(R_sq, img)
    R_1d_sorted = np.sqrt(R_sq_1d_sorted)
    #
    flux_cumulative = np.cumsum(img_1d_sorted)
    return flux_cumulative, R_1d_sorted

def find_percent_radii(cumflux: np.ma.core.MaskedArray, R_1d_sorted: np.ndarray,
                       q: float,
                       percents=(0.5, 0.9, 0.99)) -> Dict[str, float]:
    out = {}
    total_flux = cumflux.max()
    if total_flux <= 0:
        for p in percents:
            out[f"R{int(p*100)}"] = np.nan
        return out
    target = total_flux * np.array(percents)
    # ensure monotonic
    # cf = np.maximum.accumulate(cumflux)
    for p, t in zip(percents, target):
        ind = int(np.where(cumflux >= t)[0][0]) # First index that
        R_p = R_1d_sorted[ind]
        out[f"R{int(p*100)}"] = R_p
        out[f"A{int(p*100)}"] = R_p / np.sqrt(q)
        out[f"B{int(p * 100)}"] = R_p * np.sqrt(q)
    return out


def compute_total_flux(img_bgsub: np.ma.core.MaskedArray, obj_rec: Dict,
                       segmap: np.ndarray, label: int, center_xy: Tuple[float, float],
                       definition: str = 'A') -> Tuple[float, float, float, float]:
    """Return (total_flux, kron_a, kron_b, q) given definition A/B/C.
    - A: flux within 2.5×Kron.
    - B: flux within segmentation footprint of the target.
    - C: flux within 4×Kron.
    Also returns kron_a/kron_b and q=b/a for later use.
    """
    cx, cy = center_xy
    a = float(obj_rec.get('a', None))
    b = float(obj_rec.get('b', None))
    theta = float(obj_rec.get('theta', None))
    q = (b / a) if (a > 0) else 1.0

    work = img_bgsub.copy() # Masked array; TODO: need to check if it works well with SEP
    # work[zeros_mask] = 0.0

    # Kron radius (in units of semi-major axis a)
    kronr, krflag = sep.kron_radius(work.astype(np.float32), np.atleast_1d(cx), np.atleast_1d(cy), a, b, theta, 6.0)
    if (not np.isfinite(kronr)) or (kronr <= 0) or (krflag != 0):
        kronr = 2.5  # fallback

    if definition.upper() == 'B':
        # sum over segmentation footprint
        total = float(work[segmap == label].sum())
        return total, a, b, q

    # A or C: elliptical sum with scaling factor
    scale = 2.5 if definition.upper() == 'A' else 4.0
    ak = scale * kronr * a
    bk = scale * kronr * b
    total, _, _ = sep.sum_ellipse(work.astype(np.float32), np.atleast_1d(cx), np.atleast_1d(cy), ak, bk, theta, subpix=5)
    return float(total[0]), a, b, q


# ----------------------------
# Plotting
# ----------------------------

def inset_background_map(fig, ax_parent, bkg_map):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax_parent, width="32%", height="32%", loc='upper right', borderpad=0.6)
    im = axins.imshow(bkg_map, origin='lower', cmap='magma')
    axins.set_xticks([]); axins.set_yticks([])
    axins.set_title('SEP bkg map', fontsize=8)
    fig.colorbar(im, ax=axins, fraction=0.046, pad=0.04)