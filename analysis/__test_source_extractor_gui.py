#!/usr/bin/env python3
"""
Interactive SEP parameter tuner using Tkinter + Matplotlib.

Requirements:
- tools_source_extractor in your PYTHONPATH
- optical_elliptical_multipole.plotting.plot_tools in your PYTHONPATH
- ./test_source_extractor/51116_smaller_cutout.pkl existing as in __test_source_extractor.py
"""

import tkinter as tk
from tkinter import ttk #, filedialog
import sys
import traceback

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np
import pickle
import h5py

from tools_source_extractor import extract_with_sep, pick_target_label
from optical_elliptical_multipole.plotting.plot_tools import (
    AsinhStretchPlot,
    draw_segmentation,
)
import os
from datetime import datetime
import copy
# ---------------------------------------------------------------------
# Data loading (mirrors your __test_source_extractor.py)
# ---------------------------------------------------------------------

seqid = 51116

# with open("./test_source_extractor/51116_smaller_cutout.pkl", "rb") as f:
#     sci_bgsub, wht = pickle.load(f)

with h5py.File(f"./test_source_extractor/{seqid}-cropped.hdf5",'r') as f:
    # File operations go here
    mask, sci_bgsub, _, wht = np.array(f['mask_crop']), np.array(f['sci_bgsub_crop']), f['segmap_crop'], np.array(f[
                                                                                                               'wht_crop'])
print('HDF5 file successfully loaded')
# sci_bgsub = np.ma.masked_array(sci_bgsub, mask=mask)
# wht = np.ma.masked_array(wht, mask=mask)
# sci_bgsub *= np.sqrt(wht)
# wht = np.ones_like(wht)

# Center of the cutout as target for pick_target_label
target_xy_px = np.array(sci_bgsub.shape) / 2.0
extent = None  # consistent with your test script
DRAW_MODE = 'Asinh' # 'log10' or 'Asinh'

# ---------------------------------------------------------------------
# GUI App
# ---------------------------------------------------------------------


class SepGuiApp:
    def __init__(self, root):
        self.root = root
        self.save_dir = './test_source_extractor'
        self.root.title("SEP Parameter Tuner")

        # Default parameter values (starting point)
        self.default_deblend_nthresh = 32
        self.default_deblend_cont = 1e-4
        self.default_detect_thresh_sigma = 3.0
        self.default_minarea = 50

        # Tkinter variables (slider underlying values)
        # We store exponent for log scales and exponent for powers of 2
        self.var_log2_nthresh = tk.IntVar(
            value=int(np.log2(self.default_deblend_nthresh))
        )
        self.var_log10_deblend_cont = tk.DoubleVar(
            value=float(np.log10(self.default_deblend_cont))
        )
        self.var_detect_thresh = tk.DoubleVar(value=self.default_detect_thresh_sigma)
        self.var_minarea = tk.IntVar(value=self.default_minarea)

        # Build UI
        self._build_controls()
        self._build_figure()

        # Internal references to images so we can update them
        self.seg_im = None
        self.mask_im = None
        self.sci_norm = None

        # Pre-compute static background image and its stretch norm
        self._draw_static_background()

        # Initial info
        self.info_label.config(text="Adjust sliders, then click 'Run'.")
        self.cbar = None

    # ---------------- GUI layout ----------------

    def _build_controls(self):
        """Create sliders, labels, and buttons on the left side."""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Title
        ttk.Label(
            control_frame,
            text="SEP Parameters",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(pady=(0, 10))

        # deblend_nthresh (log2 scale: 2, 4, 8, ..., 1024)
        frame_nthresh = ttk.LabelFrame(control_frame, text="deblend_nthresh (2^k)")
        frame_nthresh.pack(fill=tk.X, pady=5)

        self.label_nthresh_val = ttk.Label(frame_nthresh, text="")
        self.label_nthresh_val.pack(anchor="w", padx=5, pady=(2, 0))

        scale_nthresh = tk.Scale(
            frame_nthresh,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.var_log2_nthresh,
            command=self.on_slider_change,
        )
        scale_nthresh.pack(fill=tk.X, padx=5, pady=2)

        # deblend_cont (log10 scale: 1e-10 ... 1e+2)
        frame_deblend_cont = ttk.LabelFrame(control_frame, text="deblend_cont (10^k)")
        frame_deblend_cont.pack(fill=tk.X, pady=5)

        self.label_deblend_cont_val = ttk.Label(frame_deblend_cont, text="")
        self.label_deblend_cont_val.pack(anchor="w", padx=5, pady=(2, 0))

        scale_deblend_cont = tk.Scale(
            frame_deblend_cont,
            from_=-10,
            to=2,
            resolution=1,  # integer exponents -10..2
            orient=tk.HORIZONTAL,
            variable=self.var_log10_deblend_cont,
            command=self.on_slider_change,
        )
        scale_deblend_cont.pack(fill=tk.X, padx=5, pady=2)

        # detect_thresh_sigma (linear: 0.0 .. 10.0, step 0.1)
        frame_detect = ttk.LabelFrame(control_frame, text="detect_thresh_sigma")
        frame_detect.pack(fill=tk.X, pady=5)

        self.label_detect_val = ttk.Label(frame_detect, text="")
        self.label_detect_val.pack(anchor="w", padx=5, pady=(2, 0))

        # Extended range so default 3.0 is within slider range
        scale_detect = tk.Scale(
            frame_detect,
            from_=0.0,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.var_detect_thresh,
            command=self.on_slider_change,
        )
        scale_detect.pack(fill=tk.X, padx=5, pady=2)

        # minarea (linear int: 1 .. 100)
        frame_minarea = ttk.LabelFrame(control_frame, text="minarea")
        frame_minarea.pack(fill=tk.X, pady=5)

        self.label_minarea_val = ttk.Label(frame_minarea, text="")
        self.label_minarea_val.pack(anchor="w", padx=5, pady=(2, 0))

        scale_minarea = tk.Scale(
            frame_minarea,
            from_=1,
            to=100,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.var_minarea,
            command=self.on_slider_change,
        )
        scale_minarea.pack(fill=tk.X, padx=5, pady=2)

        # Run & Save buttons
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 5))

        ttk.Button(
            btn_frame,
            text="Run",
            command=self.update_plot,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))

        ttk.Button(
            btn_frame,
            text="Save Figure",
            command=self.save_figure,
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        # Info label for SEP output (e.g., number of objects)
        self.info_label = ttk.Label(control_frame, text="", justify="left")
        self.info_label.pack(fill=tk.X, pady=(10, 5), padx=5)

        # Initialize label text
        self._update_param_labels()

    def _build_figure(self):
        """Create Matplotlib figure embedded in Tkinter."""
        figure_frame = ttk.Frame(self.root)
        figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.ax_sci = self.fig.add_subplot(1, 3, 1)
        self.ax_seg = self.fig.add_subplot(1, 3, 2)
        self.ax_mask = self.fig.add_subplot(1, 3, 3)

        self.ax_sci.set_title("SCI - BKG")
        self.ax_seg.set_title("Segmentation")
        self.ax_mask.set_title("Masked Target")

        for ax in (self.ax_sci, self.ax_seg, self.ax_mask):
            ax.set_aspect("equal")
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")

        # Leave some space at the top for suptitle
        self.fig.suptitle(f"{seqid} - SEP parameter tuner", y=0.98)
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        self.canvas = FigureCanvasTkAgg(self.fig, master=figure_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.sci_norm = None

    # ---------------- Plotting ----------------

    def _draw_static_background(self):
        """Draw the static SCI-BKG image once, keep its norm for masked image."""
        self.ax_sci.clear()
        self.ax_sci.set_title("SCI - BKG")

        if DRAW_MODE == 'log10':
            im_sci = self.ax_sci.imshow(np.log10(sci_bgsub), vmin=-6, origin='lower')
            plt.colorbar(im_sci, ax=self.ax_sci, fraction=0.046, pad=0.04)
        elif DRAW_MODE == 'Asinh':
            im_sci, norm = AsinhStretchPlot(
                self.ax_sci, sci_bgsub, origin="lower", return_norm=True,
                a = 0.01
            )
            self.sci_norm = norm

        self.ax_sci.set_aspect("equal")
        self.ax_sci.set_xlabel("x (px)")
        self.ax_sci.set_ylabel("y (px)")
        self.ax_sci.set_facecolor('k')

    def _current_params(self):
        """Return current SEP parameters computed from slider values."""
        # deblend_nthresh = 2^k
        k_log2 = int(self.var_log2_nthresh.get())
        deblend_nthresh = 2 ** k_log2

        # deblend_cont = 10^k
        k_log10 = float(self.var_log10_deblend_cont.get())
        deblend_cont = 10.0 ** k_log10

        detect_thresh_sigma = float(self.var_detect_thresh.get())
        minarea = int(self.var_minarea.get())

        return deblend_nthresh, deblend_cont, detect_thresh_sigma, minarea

    def update_plot(self):
        """Run SEP with current parameters and update the plots."""
        deblend_nthresh, deblend_cont, detect_thresh_sigma, minarea = (
            self._current_params()
        )

        try:
            # Run SEP
            objs, segmap = extract_with_sep(
                sci_bgsub,
                wht,
                deblend_nthresh=deblend_nthresh,
                deblend_cont=deblend_cont,
                detect_thresh_sigma=detect_thresh_sigma,
                minarea=minarea,
                return_segmap=True,
            )

            n_obj = len(objs)
            # Pick target closest to target_xy_px
            label, rec, dist = pick_target_label(
                objs, segmap, target_xy_px, verbose=False
            )

            # Robust formatting of dist (can be list or scalar)
            if isinstance(dist, (int, float, np.floating)):
                dist_str = f"{dist:.2f} px"
            else:
                dist_str = str(dist)

            # --- Segmentation using draw_segmentation ---
            self.ax_seg.clear()
            im, cmap = draw_segmentation(
                self.ax_seg,
                segmap,
                title="Segmentation",
                target_label=label,
                outline=False,
                origin="lower",
                extent=extent,
            )
            self.seg_im = im

            # --- Masked target ---
            self.ax_mask.clear()
            self.ax_mask.set_title("Masked Target")

            mask = (segmap != label) * (segmap != 0)
            sci_bgsub_masked = np.ma.masked_array(sci_bgsub, mask=mask)

            if DRAW_MODE == 'log10':
                im_mask = self.ax_mask.imshow(np.log10(sci_bgsub_masked), vmin=-6, origin='lower')
                if self.cbar is None:
                    self.cbar = plt.colorbar(im_mask, ax=self.ax_mask, fraction=0.046, pad=0.04)
                else:
                    self.cbar = plt.colorbar(im_mask, cax=self.cbar.ax, fraction=0.046, pad=0.04)
                mask_nan = copy.copy(~mask).astype(float)
                mask_nan[~mask] = np.nan
                self.ax_mask.imshow(mask_nan, cmap='binary', origin='lower')
                self.ax_mask.set_facecolor('k')

            elif DRAW_MODE == 'Asinh':
                if self.sci_norm is not None:
                    im_mask = self.ax_mask.imshow(
                        sci_bgsub_masked,
                        origin="lower",
                        norm=self.sci_norm,
                    )
                else:
                    raise ValueError(f"self.sci_norm is {self.sci_norm}")
            else:
                raise ValueError(f"DRAW_MODE must be 'log10' or 'Asinh'")

            self.ax_mask.set_aspect("equal")
            self.ax_mask.set_xlabel("x (px)")
            self.ax_mask.set_ylabel("y (px)")
            self.mask_im = im_mask

            # Info text (number of objects, chosen label, distance)
            info_text = (
                f"Objects detected: {n_obj}\n"
                f"Target label: {label}\n"
                f"Distance from center: {dist_str}\n\n"
                f"deblend_nthresh = {deblend_nthresh}\n"
                f"deblend_cont = {deblend_cont:.3e}\n"
                f"detect_thresh_sigma = {detect_thresh_sigma:.2f}\n"
                f"minarea = {minarea}"
            )
            self.info_label.config(text=info_text)

        except Exception as e:
            # Show brief message in GUI
            self.info_label.config(text=f"Error during SEP extraction:\n{e}")

            # Print full traceback to console and re-raise
            print("Error during SEP extraction:", file=sys.stderr)
            traceback.print_exc()
            raise

        # Redraw canvas and keep space for suptitle
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.canvas.draw_idle()
        print(f"Drawn! (time: {str(datetime.now())})")

    # ---------------- Slider & save handling ----------------

    def _update_param_labels(self):
        """Update the labels next to sliders to show real parameter values."""
        deblend_nthresh, deblend_cont, detect_thresh_sigma, minarea = (
            self._current_params()
        )

        self.label_nthresh_val.config(
            text=f"deblend_nthresh = {deblend_nthresh} (2^{int(self.var_log2_nthresh.get())})"
        )
        self.label_deblend_cont_val.config(
            text=(
                f"deblend_cont = 10^{int(self.var_log10_deblend_cont.get())}"
                f" = {deblend_cont:.3e}"
            )
        )
        self.label_detect_val.config(
            text=f"detect_thresh_sigma = {detect_thresh_sigma:.1f}"
        )
        self.label_minarea_val.config(text=f"minarea = {minarea}")

    def on_slider_change(self, _event=None):
        """
        Called whenever any slider moves.
        We only update the numeric labels; SEP is run only when clicking 'Run'.
        """
        self._update_param_labels()

    def save_figure(self):
        """
        Save current figure to file.
        We add a small text box with current SEP parameters on the figure.
        """
        dir = self.save_dir
        datetime_string_new = str(datetime.now()).replace(' ', '_').replace(':', '')
        datetime_string_new = datetime_string_new[:datetime_string_new.find('.')]
        deblend_nthresh, deblend_cont, detect_thresh_sigma, minarea = (
            self._current_params()
        )
        fname = os.path.join(dir, f"SEP_test-{datetime_string_new}.pdf")
        # fname = filedialog.asksaveasfilename(
        #     title="Save figure",
        #     defaultextension=".png",
        #     filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
        # )
        if not fname:
            return

        param_text = (
            f"deblend_nthresh={deblend_nthresh}, "
            f"deblend_cont={deblend_cont:.3e}, "
            f"detect_thresh_sigma={detect_thresh_sigma:.2f}, "
            f"minarea={minarea}"
        )

        # Temporarily add text to figure
        txt = self.fig.text(
            0.01,
            0.01,
            param_text,
            fontsize=8,
            ha="left",
            va="bottom",
        )

        self.fig.tight_layout(rect=[0, 0, 1, 0.95])
        self.fig.savefig(fname, dpi=300)
        txt.remove()
        self.canvas.draw_idle()


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------


def main():
    root = tk.Tk()
    app = SepGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
