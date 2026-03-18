import os
import time
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable

from optical_elliptical_multipole.nonjax.intensity_functions import sersic
from optical_elliptical_multipole.plotting.plot_tools import AsinhStretchPlot


# ============================================================
# User-configurable constants
# ============================================================
PIXEL_WIDTH = 0.03  # arcsec
X_MIN, X_MAX = -2.01, 2.01  # arcsec
Y_MIN, Y_MAX = -2.01, 2.01  # arcsec
AMPLITUDE = 1.0
EPS_FLOOR = 1.0e-12


# ============================================================
# Helper math
# ============================================================
def b_n_approx(n):
    n = float(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    return max(1.999 * n - 0.327, 1e-5)


def elliptical_radius_circularized(x, y, q):
    """
    Elliptical radius using a circularized-R_sersic convention.

    We use:
        R_ell = sqrt(q * x^2 + y^2 / q)

    so that isophote area pi*a*b = pi*R_ell^2, i.e. R_ell behaves as a
    circularized radius. For q=1, this reduces to the circular radius.

    This is consistent with the common lensing convention where the
    circularized radius preserves enclosed area, which is the key point
    behind using a 'circularized' Sersic radius.
    """
    q = max(float(q), 1e-4)
    return np.sqrt(q * x * x + (y * y) / q)


def dlnI_dr_sersic(r, R_sersic, n_sersic):
    """
    Analytic radial derivative of ln I for the Sersic profile.

    I(r) = A exp[-b_n((r/R_s)^(1/n) - 1)]

    d ln I / dr = -(b_n / (n R_s)) * (r/R_s)^(1/n - 1)

    The imported sersic() function internally floors R at 1e-4 for numerical
    stability, so we do the same here for consistent behavior.
    """
    r = np.maximum(np.asarray(r, dtype=float), 1.0e-4)
    bn = b_n_approx(n_sersic)
    x = r / float(R_sersic)
    return -(bn / (float(n_sersic) * float(R_sersic))) * np.power(x, 1.0 / float(n_sersic) - 1.0)


def threshold_mask_from_factor(r, factor, R_sersic, n_sersic, pixel_width=PIXEL_WIDTH):
    """
    Oversampling mask based on the log-slope criterion

        |d ln I / dr| * Delta_r > ln(factor)

    where the UI slider directly controls `factor`.

    Notes
    -----
    - factor <= 1 implies ln(factor) <= 0, so the criterion would be true
      everywhere because the left side is nonnegative. To keep behavior sane,
      we treat factor <= 1 as a threshold of 0, which still makes the mask all True.
    - This matches the user's requested slider semantics, although factor < 1
      is not physically meaningful as a "changes by this multiplicative factor"
      threshold.
    """
    eps_ln = np.log(max(float(factor), 1.0e-12))
    lhs = np.abs(dlnI_dr_sersic(r, R_sersic=R_sersic, n_sersic=n_sersic)) * pixel_width
    mask = lhs > eps_ln
    return mask, eps_ln, lhs


def max_radius_satisfying_threshold(factor, R_sersic, n_sersic, r_max_plot, pixel_width=PIXEL_WIDTH):
    """
    Numerically find the largest radius satisfying the criterion within the
    plotting domain. This is robust and avoids edge-case algebra.
    """
    r_dense = np.linspace(1.0e-4, r_max_plot, 20000)
    mask, eps_ln, _ = threshold_mask_from_factor(
        r_dense,
        factor=factor,
        R_sersic=R_sersic,
        n_sersic=n_sersic,
        pixel_width=pixel_width,
    )
    if np.any(mask):
        return float(r_dense[mask][-1]), eps_ln
    return None, eps_ln


def build_grid(xmin=X_MIN, xmax=X_MAX, ymin=Y_MIN, ymax=Y_MAX, dx=PIXEL_WIDTH, dy=PIXEL_WIDTH):
    x = np.arange(xmin, xmax + 0.5 * dx, dx)
    y = np.arange(ymin, ymax + 0.5 * dy, dy)
    xx, yy = np.meshgrid(x, y)
    return x, y, xx, yy


def save_current_figure_pdf(fig):
    ts = time.strftime("%Y%m%d_%H%M%S")
    outname = f"sersic_log_slope_gui_{ts}.pdf"
    fig.savefig(outname, dpi=300, bbox_inches="tight")
    return os.path.abspath(outname)


# ============================================================
# GUI app
# ============================================================
class SersicLogSlopeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Sersic log-slope threshold explorer")

        # ---------- state variables ----------
        self.factor_var = tk.DoubleVar(value=2.0)
        self.n_var = tk.DoubleVar(value=4.0)
        self.R_var = tk.DoubleVar(value=0.5)
        self.q_var = tk.DoubleVar(value=1.0)

        # ---------- layout ----------
        self.root_frame = ttk.Frame(master, padding=8)
        self.root_frame.pack(fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Frame(self.root_frame)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 8))

        self.plot_frame = ttk.Frame(self.root_frame)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._build_controls()
        self._build_figure()

        # precompute grid
        self.x, self.y, self.xx, self.yy = build_grid()
        self.extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]
        self.r_plot_max = np.sqrt(max(abs(X_MIN), abs(X_MAX)) ** 2 + max(abs(Y_MIN), abs(Y_MAX)) ** 2)
        self.r_1d = np.linspace(1.0e-4, self.r_plot_max, 3000)

        self.update_plot()

    def _build_controls(self):
        # Factor
        factor_frame = ttk.Frame(self.controls_frame)
        factor_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(factor_frame, text="Factor change across Δr = 0.03 arcsec").pack(anchor="w")
        factor_row = ttk.Frame(factor_frame)
        factor_row.pack(fill=tk.X)
        self.factor_scale = tk.Scale(
            factor_row,
            from_=1.0,
            to=3.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.factor_var,
            command=lambda _=None: self.update_plot(),
        )
        self.factor_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(factor_row, textvariable=self.factor_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        # n_sersic
        n_frame = ttk.Frame(self.controls_frame)
        n_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(n_frame, text="n_sersic").pack(anchor="w")
        n_row = ttk.Frame(n_frame)
        n_row.pack(fill=tk.X)
        self.n_scale = tk.Scale(
            n_row,
            from_=2.0,
            to=8.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.n_var,
            command=lambda _=None: self.update_plot(),
        )
        self.n_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(n_row, textvariable=self.n_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        # R_sersic
        R_frame = ttk.Frame(self.controls_frame)
        R_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(R_frame, text="R_sersic (circularized, arcsec)").pack(anchor="w")
        R_row = ttk.Frame(R_frame)
        R_row.pack(fill=tk.X)
        self.R_scale = tk.Scale(
            R_row,
            from_=0.2,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.R_var,
            command=lambda _=None: self.update_plot(),
        )
        self.R_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(R_row, textvariable=self.R_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        # q slider (added because user requested axis ratio q)
        q_frame = ttk.Frame(self.controls_frame)
        q_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(q_frame, text="Axis ratio q").pack(anchor="w")
        q_row = ttk.Frame(q_frame)
        q_row.pack(fill=tk.X)
        self.q_scale = tk.Scale(
            q_row,
            from_=0.2,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.q_var,
            command=lambda _=None: self.update_plot(),
        )
        self.q_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(q_row, textvariable=self.q_var, width=6).pack(side=tk.LEFT, padx=(6, 0))

        # buttons
        btn_frame = ttk.Frame(self.controls_frame)
        btn_frame.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(btn_frame, text="Save PDF", command=self.on_save_pdf).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="Quit", command=self.master.destroy).pack(fill=tk.X, pady=2)

    def _build_figure(self):
        self.fig = Figure(figsize=(15, 4.8), dpi=100)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.35)
        self.ax_img = self.fig.add_subplot(gs[0, 0])
        self.ax_mask = self.fig.add_subplot(gs[0, 1])
        self.ax_1d = self.fig.add_subplot(gs[0, 2])

        # Dedicated colorbar axis appended to the middle panel so it never overlaps.
        self.cax = make_axes_locatable(self.ax_mask).append_axes("right", size="5%", pad=0.08)
        self.cbar = None

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        factor = float(self.factor_var.get())
        n_sersic = float(self.n_var.get())
        R_sersic = float(self.R_var.get())
        q = float(self.q_var.get())

        # 2D image
        R_ell = elliptical_radius_circularized(self.xx, self.yy, q=q)
        img = np.asarray(sersic(R_ell, amplitude=AMPLITUDE, R_sersic=R_sersic, n_sersic=n_sersic), dtype=float)

        mask, eps_ln, _ = threshold_mask_from_factor(
            R_ell,
            factor=factor,
            R_sersic=R_sersic,
            n_sersic=n_sersic,
            pixel_width=PIXEL_WIDTH,
        )

        # 1D radial profile
        I_1d = np.asarray(sersic(self.r_1d, amplitude=AMPLITUDE, R_sersic=R_sersic, n_sersic=n_sersic), dtype=float)
        r_max, _ = max_radius_satisfying_threshold(
            factor=factor,
            R_sersic=R_sersic,
            n_sersic=n_sersic,
            r_max_plot=self.r_plot_max,
            pixel_width=PIXEL_WIDTH,
        )

        # clear axes
        self.ax_img.clear()
        self.ax_mask.clear()
        self.ax_1d.clear()

        # consistent asinh norm across both 2D panels
        vmin = np.nanmin(img)
        vmax = np.nanmax(img)

        im1, norm = AsinhStretchPlot(
            self.ax_img,
            img,
            a=0.1,
            vmin=vmin,
            vmax=vmax,
            return_norm=True,
            origin="lower",
            extent=self.extent,
            aspect="equal",
        )
        AsinhStretchPlot(
            self.ax_mask,
            img,
            a=0.1,
            norm=norm,
            origin="lower",
            extent=self.extent,
            aspect="equal",
        )

        # contour for mask boundary
        mask_float = mask.astype(float)
        self.ax_mask.contour(
            self.xx,
            self.yy,
            mask_float,
            levels=[0.5],
            linewidths=2.0,
        )

        # 1D profile
        self.ax_1d.plot(self.r_1d, I_1d, lw=2)
        self.ax_1d.set_yscale("log")
        self.ax_1d.set_xlim(0.0, self.r_plot_max)
        ymin = max(np.nanmin(I_1d[I_1d > 0]), EPS_FLOOR)
        ymax = np.nanmax(I_1d)
        self.ax_1d.set_ylim(ymin, ymax * 1.1)
        if r_max is not None:
            self.ax_1d.axvline(r_max, linestyle="--", linewidth=2)

        # Axis cosmetics
        self.ax_img.set_title("2D Sersic intensity")
        self.ax_mask.set_title("2D intensity + threshold contour")
        self.ax_1d.set_title("1D radial Sersic profile")

        for ax in (self.ax_img, self.ax_mask):
            ax.set_xlabel("x [arcsec]")
            ax.set_ylabel("y [arcsec]")

        self.ax_1d.set_xlabel("R [arcsec]")
        self.ax_1d.set_ylabel("Intensity")

        # show the outermost isophote satisfying the threshold for reference
        if r_max is not None and np.isfinite(r_max) and r_max > 0:
            a = r_max / np.sqrt(max(q, 1e-8))
            b = r_max * np.sqrt(max(q, 1e-8))
            ell = Ellipse(
                xy=(0.0, 0.0),
                width=2.0 * a,
                height=2.0 * b,
                angle=0.0,
                fill=False,
                linewidth=1.5,
                linestyle="--",
            )
            self.ax_mask.add_patch(ell)

        # Rebuild the colorbar inside its own dedicated axis.
        # Clearing cax is much more robust than calling Colorbar.remove(), which
        # can fail after repeated layout changes in Tkinter redraw cycles.
        self.cax.clear()
        self.cbar = self.fig.colorbar(im1, cax=self.cax)
        self.cbar.set_label("Intensity (asinh stretch)")

        title = (
            f"Sersic log-slope threshold explorer   "
            f"factor = {factor:.2f}   ln(factor) = {eps_ln:.4f}   "
            f"n = {n_sersic:.2f}   R_sersic = {R_sersic:.2f} arcsec   q = {q:.2f}"
        )
        if r_max is not None:
            title += f"   R_max = {r_max:.4f} arcsec"
        else:
            title += "   R_max = none in plotting range"
        self.fig.suptitle(title, fontsize=11)
        # Do not call tight_layout() here. It conflicts with the appended colorbar
        # axis and triggers warnings / unstable geometry updates in embedded Tk.
        self.canvas.draw_idle()

    def on_save_pdf(self):
        try:
            outpath = save_current_figure_pdf(self.fig)
            messagebox.showinfo("Saved", f"Saved PDF to:\n{outpath}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))


# ============================================================
# Main
# ============================================================
def main():
    root = tk.Tk()
    app = SersicLogSlopeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
