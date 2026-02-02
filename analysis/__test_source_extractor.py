#!/usr/bin/env python3
from matplotlib import pyplot as plt
from tools_source_extractor import extract_with_sep, pick_target_label, crop_target, cumulative_ellipse_curve, find_percent_radii, crop_array_list_w_ratio, AsinhStretchPlot
from optical_elliptical_multipole.plotting.plot_tools import draw_segmentation, AsinhStretchPlot
import os
import numpy as np
from types import SimpleNamespace
from matplotlib.ticker import MaxNLocator
import pickle
from datetime import datetime

seqid = 51116
# with open('./test_source_extractor/51116_bigger_cutout.pkl', 'rb') as f:
#     sci_bgsub, wht = pickle.load(f)
with open('./test_source_extractor/51116_smaller_cutout.pkl', 'rb') as f:
    sci_bgsub, wht = pickle.load(f)

# target_xy_px = 3000.82808107628, 3000.7875026885663
target_xy_px = np.array(sci_bgsub.shape)/2

sep_args = {'deblend_nthresh': 32, # 32
            # DEBLEND_NTHRESH : the number of thresholds the intensity range is devided up in. 32 is the most common number.
            'deblend_cont': 1e-4, # 1e-4
            # Minimum contrast ratio used for object deblending. Default is 0.005. To entirely disable deblending, set to 1.0.
            'detect_thresh_sigma': 2.0, #3.0
            # Check: https://sep.readthedocs.io/en/stable/api/sep.extract.html; when err map is given,
            # the interpretation changes so it needs to be updated.
            'minarea': 20, # 50  # minimum area; default 5 pixels
            'path': './test_source_extractor'
            }

args = SimpleNamespace(**sep_args)

extra_text = (f"deblend_nthresh: {args.deblend_nthresh}\n"
              f"deblend_cont: {args.deblend_cont}\n"
              f"detect_thresh_sigma: {args.detect_thresh_sigma}\n"
              f"minarea: {args.minarea}\n")

objs, segmap = extract_with_sep(sci_bgsub, wht,
                                deblend_nthresh=args.deblend_nthresh,
                                deblend_cont=args.deblend_cont,
                                detect_thresh_sigma=args.detect_thresh_sigma,
                                minarea=args.minarea,
                                return_segmap=True)
verbose = True
# obj contains information of all the detedcted objects (x and y coordiantes, major and minor axis, npix, etc)
if verbose:
    print(f"SEP detected the following number of objects: {len(objs)}")
    # print(f"SEP fields: {list(objs.dtype.names)}")

if verbose:
    print("Checking the target from SEP... can take a while for big images...")
label, rec, dist = pick_target_label(objs, segmap, target_xy_px, verbose=verbose)
if verbose:
    print("target check done, now plotting...")
fig_hw_unit_inch = 5
fig, axes = plt.subplots(1, 6, figsize=(fig_hw_unit_inch*6, fig_hw_unit_inch))
axes[5].set_title("Masked")
axes[0].set_title("Args")
axes[1].set_title("SCI - BKG")
axes[2].set_title("WHT")
axes[3].set_title("1/sqrt(WHT)")
extent = None

axes[0].text(0.,1.0, extra_text, horizontalalignment='left', verticalalignment='top')
axes[0].set_xticks([])
axes[0].set_yticks([])
im_img_bkgsub, norm = AsinhStretchPlot(axes[1], sci_bgsub, origin="lower", return_norm=True, extent=extent)
# im_img_bkgsub = axes[1].imshow(sci_bgsub, origin="lower", norm=None, extent=extent)
plt.colorbar(mappable=im_img_bkgsub, ax=axes[1], fraction=0.046, pad=0.04)
im_wht = AsinhStretchPlot(axes[2], wht, origin='lower', extent=extent)
plt.colorbar(mappable=im_wht, ax=axes[2], fraction=0.046, pad=0.04)
im_wht_sqrt_inverse = AsinhStretchPlot(axes[3], 1/np.sqrt(wht), origin='lower', extent=extent)
plt.colorbar(mappable=im_wht_sqrt_inverse, ax=axes[3], fraction=0.046, pad=0.04)

im, cmap = draw_segmentation(axes[4], segmap, title='Segmentation', target_label=label,
                    outline = False, origin='lower', extent=extent)
cbar = plt.colorbar(mappable=im, ax=axes[4], fraction=0.046, pad=0.04)
cbar.locator = MaxNLocator(integer=True, nbins=5)
cbar.update_ticks()
for ax in axes:
    ax.set_aspect('equal')
    ax.set_xlabel('x (px)')

mask = (segmap != label) * (segmap != 0)
sci_bgsub_masked = np.ma.masked_array(sci_bgsub, mask=mask)
im_img_bkgsub_masked = AsinhStretchPlot(axes[5], sci_bgsub_masked, origin="lower", norm=norm, extent=extent)
plt.colorbar(mappable=im_img_bkgsub_masked, ax=axes[5], fraction=0.046, pad=0.04)

fig.tight_layout()
fig.suptitle(f"{seqid} large cutout")
datetime_string_new = str(datetime.now()).replace(' ', '_').replace(':', '')
datetime_string_new = datetime_string_new[:datetime_string_new.find('.')]
fig.savefig(os.path.join(args.path, f"{seqid}-02-bg_and_segmap-{datetime_string_new}.pdf"))
plt.show()

print("done!")


