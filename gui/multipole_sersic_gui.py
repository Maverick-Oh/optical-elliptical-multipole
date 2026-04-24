"""
Interactive GUI for comparing Elliptical Multipole vs Circular Multipole
Sersic profiles side-by-side, with compound m=3 + m=4 multipoles.

Usage:
    python multipole_sersic_gui.py
    python multipole_sersic_gui.py --make-movie
    python multipole_sersic_gui.py --make-movie --movie-mode linear
    python multipole_sersic_gui.py --make-movie --movie-mode smooth
"""

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

import imageio
import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

# Add package root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optical_elliptical_multipole.nonjax import intensity_functions as intfun
from optical_elliptical_multipole.nonjax import profiles1D as p1d
from optical_elliptical_multipole.nonjax import profiles2D as p2d
from optical_elliptical_multipole.plotting.plot_tools import AsinhStretchPlot

hold_frames = 60

MOVIE_SETTINGS = {
    "fps": 20,
    "transition_modes": ["linear", "smooth"],
    "output_basename": "multipole_sersic_transition",
    "still_seconds": 0.5,
    "transition_steps": 50,
    "initial_state": {
        "n_sersic": 2.5,
        "R_sersic": 1.0,
        "amplitude": 1.0,
        "q": 1.0,
        "theta_ell": 0.0,
        "x0": 0.0,
        "y0": 0.0,
        "background": 0.0,
        "a_3": 0.0,
        "angle_3": 0.0,
        "a_4": 0.0,
        "angle_4": 0.0,
        "contour_r0": 1.0,
        "contour_r": 0.5,
        "contour_g": 1.0,
        "contour_b": 0.5,
        "grid_N": 150,
        "grid_range": 2.0,
        "show_contour": True,
    },
    "sequence": [
        {"kind": "hold", "title": "[Initial state (circle)]", "frames": hold_frames},
        {
            "kind": "transition",
            "param": "a_4",
            "start": 0.0,
            "end": 0.1,
            "title": "Increasing a_4",
            "highlight": "a_4",
        },
        {"kind": "hold", "title": "[m=4 Multipole on a circle]", "frames": hold_frames},
        {
            "kind": "transition",
            "param": "q",
            "start": 1.0,
            "end": 0.5,
            "title": "Decreasing q (ellipticizing)",
            "highlight": "q",
        },
        {"kind": "hold", "title": "[m=4 Multipole on an ellipse]", "frames": hold_frames},
        {
            "kind": "transition",
            "param": "angle_4",
            "start": 0.0,
            "end": np.pi / 4.0,
            "title": "Increasing angle_4",
            "highlight": "angle_4",
        },
        {"kind": "hold", "title": "[Rotating m=4 Multipole on an ellipse]", "frames": hold_frames},
        {
            "kind": "transition",
            "param": "q",
            "start": 0.5,
            "end": 1.0,
            "title": "Increasing q (circularizing)",
            "highlight": "q",
        },
        {"kind": "hold", "title": "[Rotated m=4 Multipole on a circle]", "frames": hold_frames},
        {
            "kind": "reset",
            "title": "Initial state (circle)",
            "frames": 10,
        },
        {
            "kind": "transition",
            "param": "a_3",
            "start": 0.0,
            "end": 0.1,
            "title": "Increasing a_3",
            "highlight": "a_3",
        },
        {"kind": "hold", "title": "[m=3 Multipole on a circle]", "frames": hold_frames},
        {
            "kind": "transition",
            "param": "q",
            "start": 1.0,
            "end": 0.5,
            "title": "Decreasing q (ellipticizing)",
            "highlight": "q",
        },
        {"kind": "hold", "title": "[m=3 Multipole on an ellipse]", "frames": hold_frames},
        {
            "kind": "transition",
            "param": "angle_3",
            "start": 0.0,
            "end": np.pi / 3.0,
            "title": "Increasing angle_3",
            "highlight": "angle_3",
        },
        {"kind": "hold", "title": "[Rotating m=3 Multipole on an ellipse]", "frames": hold_frames},
        {
            "kind": "transition",
            "param": "q",
            "start": 0.5,
            "end": 1.0,
            "title": "Increasing q (circularizing)",
            "highlight": "q",
        },
        {"kind": "hold", "title": "[Rotated m=3 Multipole on a circle]", "frames": hold_frames},
    ],
}


PARAMETER_TEXT_ORDER = [
    ("n_sersic", "n_sersic", "{:.1f}", True),
    ("R_sersic", "R_sersic", "{:.1f}", True),
    ("amplitude", "amplitude", "{:.1f}", True),
    ("q", "q", "{:.1f}", True),
    ("theta_ell", "theta_ell", "{:.1f}", True),
    ("x0", "x0", "{:.1f}", True),
    ("y0", "y0", "{:.1f}", True),
    ("background", "background", "{:.1f}", True),
    ("a_3", "a_3", "{:.3f}", True),
    ("angle_3", "angle_3", "{:.1f}", True),
    ("a_4", "a_4", "{:.3f}", True),
    ("angle_4", "angle_4", "{:.1f}", True),
    ("contour_r0", "contour_r0", "{:.1f}", False),
    ("contour_r", "contour_r", "{:.1f}", False),
    ("contour_g", "contour_g", "{:.1f}", False),
    ("contour_b", "contour_b", "{:.1f}", False),
    ("grid_N", "N pixels", "{:d}", False),
    ("grid_range", "Range", "{:.1f}", False),
]


def interpolate_value(start, end, u, mode):
    if mode == "smooth":
        weight = 0.5 * (1.0 - np.cos(np.pi * u))
    else:
        weight = u
    return start + (end - start) * weight


class MultipoleSeriscGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multipole Sersic Profile Explorer  (Elliptical vs Circular)")
        self.root.geometry("1680x920")

        self.defaults = {
            "n_sersic": 4.0,
            "R_sersic": 0.3,
            "amplitude": 0.05,
            "q": 0.7,
            "theta_ell": 0.0,
            "x0": 0.0,
            "y0": 0.0,
            "background": 0.001,
            "a_3": 0.0,
            "angle_3": 0.0,
            "a_4": 0.0,
            "angle_4": 0.0,
            "contour_r0": 1.0,
            "contour_r": 1.0,
            "contour_g": 0.0,
            "contour_b": 0.0,
        }

        self.params = {k: tk.DoubleVar(value=v) for k, v in self.defaults.items()}
        self.grid_N = tk.IntVar(value=150)
        self.grid_range = tk.DoubleVar(value=2.0)
        self.show_contour = tk.BooleanVar(value=True)

        self.left_frame = ttk.Frame(root, padding="10")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.right_frame = ttk.Frame(root, padding="5")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls()
        self._build_figure()
        self._bind_contour_controls()

        self.last_render_state = None
        self.last_extent = None
        self.ell_line = None
        self.circ_line = None
        self.shared_colorbar = None
        self.root.bind("<Configure>", self._on_resize)

        self.render_full()

    def _build_controls(self):
        lf = self.left_frame

        ttk.Label(
            lf, text="Sersic Parameters", font=("Helvetica", 13, "bold")
        ).pack(anchor=tk.W, pady=(0, 10))

        sersic_sliders = [
            ("n_sersic", 0.5, 10.0, 0.1),
            ("R_sersic", 0.01, 2.0, 0.01),
            ("amplitude", 0.001, 1.0, 0.001),
            ("q", 0.1, 1.0, 0.01),
            ("theta_ell", -np.pi, np.pi, 0.01),
            ("x0", -1.0, 1.0, 0.01),
            ("y0", -1.0, 1.0, 0.01),
            ("background", 0.0, 0.1, 0.0001),
        ]
        for label, lo, hi, res in sersic_sliders:
            self._create_slider(label, lo, hi, res)

        ttk.Separator(lf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(
            lf, text="Multipole Parameters", font=("Helvetica", 13, "bold")
        ).pack(anchor=tk.W, pady=(0, 5))
        ttk.Label(
            lf,
            text="(shared a_m and angle for both profiles)",
            font=("Helvetica", 9),
            foreground="gray",
        ).pack(anchor=tk.W, pady=(0, 8))

        multipole_sliders = [
            ("a_3", -0.5, 0.5, 0.005),
            ("angle_3", -np.pi, np.pi, 0.01),
            ("a_4", -0.5, 0.5, 0.005),
            ("angle_4", -np.pi, np.pi, 0.01),
        ]
        for label, lo, hi, res in multipole_sliders:
            self._create_slider(label, lo, hi, res)

        ttk.Separator(lf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(lf, text="1D Contour", font=("Helvetica", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 5)
        )

        ttk.Checkbutton(
            lf,
            text="Show 1D contour",
            variable=self.show_contour,
            command=self.update_contours_only,
        ).pack(anchor=tk.W, pady=(0, 6))

        contour_sliders = [
            ("contour_r0", 0.1, 3.0, 0.05),
            ("contour_r", 0.0, 1.0, 0.05),
            ("contour_g", 0.0, 1.0, 0.05),
            ("contour_b", 0.0, 1.0, 0.05),
        ]
        for label, lo, hi, res in contour_sliders:
            self._create_slider(label, lo, hi, res)

        ttk.Separator(lf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(lf, text="Grid Settings", font=("Helvetica", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 5)
        )

        f_grid = ttk.Frame(lf)
        f_grid.pack(fill=tk.X, pady=2)
        ttk.Label(f_grid, text="N pixels", width=12).pack(side=tk.LEFT)
        tk.Scale(
            f_grid,
            from_=50,
            to=400,
            resolution=10,
            orient=tk.HORIZONTAL,
            variable=self.grid_N,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        f_range = ttk.Frame(lf)
        f_range.pack(fill=tk.X, pady=2)
        ttk.Label(f_range, text='Range (")', width=12).pack(side=tk.LEFT)
        tk.Scale(
            f_range,
            from_=0.5,
            to=5.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.grid_range,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Separator(lf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        btn_style = ttk.Style()
        btn_style.configure("Action.TButton", font=("Helvetica", 11))

        ttk.Button(
            lf, text="Refresh Plot", style="Action.TButton", command=self.render_full
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            lf, text="Save as PDF", style="Action.TButton", command=self.save_pdf
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            lf, text="Reset Defaults", style="Action.TButton", command=self.reset_defaults
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            lf,
            text="Make Movies",
            style="Action.TButton",
            command=lambda: self.make_movies(MOVIE_SETTINGS["transition_modes"]),
        ).pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(lf, textvariable=self.status_var, foreground="gray").pack(
            side=tk.BOTTOM, anchor=tk.W, pady=5
        )

    def _build_figure(self):
        self.fig = Figure(figsize=(14.8, 5.7), dpi=100)
        outer = GridSpec(
            1,
            3,
            figure=self.fig,
            width_ratios=[0.82, 1.0, 1.0],
            left=0.04,
            right=0.92,
            bottom=0.10,
            top=0.90,
            wspace=0.10,
        )

        self.ax_info = self.fig.add_subplot(outer[0])
        self.ax_ell = self.fig.add_subplot(outer[1])
        self.ax_circ = self.fig.add_subplot(outer[2])
        self.ax_cbar = self.fig.add_axes([0.935, 0.18, 0.015, 0.60])

        self.ax_info.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame)
        self.toolbar.update()

    def _bind_contour_controls(self):
        for key in ["contour_r0", "contour_r", "contour_g", "contour_b"]:
            self.params[key].trace_add("write", self._contour_var_changed)

    def _contour_var_changed(self, *_args):
        self.update_contours_only()

    def _on_resize(self, _event=None):
        if self.last_render_state is None:
            return
        self._position_shared_colorbar()
        self.canvas.draw_idle()

    def _create_slider(self, label, lo, hi, res):
        frame = ttk.Frame(self.left_frame)
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)
        tk.Scale(
            frame,
            from_=lo,
            to=hi,
            resolution=res,
            orient=tk.HORIZONTAL,
            variable=self.params[label],
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def collect_state(self):
        state = {k: v.get() for k, v in self.params.items()}
        state["grid_N"] = self.grid_N.get()
        state["grid_range"] = self.grid_range.get()
        state["show_contour"] = self.show_contour.get()
        return state

    def apply_state_to_controls(self, state):
        for key, value in state.items():
            if key in self.params:
                self.params[key].set(value)
        if "grid_N" in state:
            self.grid_N.set(int(state["grid_N"]))
        if "grid_range" in state:
            self.grid_range.set(state["grid_range"])
        if "show_contour" in state:
            self.show_contour.set(bool(state["show_contour"]))

    def reset_defaults(self):
        for key, value in self.defaults.items():
            self.params[key].set(value)
        self.grid_N.set(150)
        self.grid_range.set(2.0)
        self.show_contour.set(True)
        self.render_full()

    def _compute_images(self, state):
        rng = state["grid_range"]
        N = int(state["grid_N"])
        x = np.linspace(-rng, rng, N)
        y = np.linspace(-rng, rng, N)
        X, Y = np.meshgrid(x, y)

        m_arr = np.array([3, 4])
        a_arr = np.array([state["a_3"], state["a_4"]], dtype=float)
        ang_arr = np.array([state["angle_3"], state["angle_4"]], dtype=float)

        sersic_kw = dict(
            amplitude=state["amplitude"],
            R_sersic=state["R_sersic"],
            n_sersic=state["n_sersic"],
        )

        img_ell = p2d.Elliptical_Multipole_Profile_2D(
            X,
            Y,
            intfun.sersic,
            q=state["q"],
            theta_ell=state["theta_ell"],
            m=m_arr,
            a_m=a_arr,
            phi_m=ang_arr,
            x0=state["x0"],
            y0=state["y0"],
            **sersic_kw,
        )
        img_ell += state["background"]

        img_circ = p2d.Circular_Multipole_Profile_2D(
            X,
            Y,
            intfun.sersic,
            q=state["q"],
            theta_ell=state["theta_ell"],
            m=m_arr,
            a_m=a_arr,
            theta_m=ang_arr,
            x0=state["x0"],
            y0=state["y0"],
            **sersic_kw,
        )
        img_circ += state["background"]
        return img_ell, img_circ, [-rng, rng, -rng, rng]

    def _compute_contours(self, state):
        m_arr = np.array([3, 4])
        a_arr = np.array([state["a_3"], state["a_4"]], dtype=float)
        ang_arr = np.array([state["angle_3"], state["angle_4"]], dtype=float)

        ell_xy = p1d.Elliptical_Multipole_Profile_1D(
            state["contour_r0"],
            q=state["q"],
            theta_ell=state["theta_ell"],
            m=m_arr,
            a_m=a_arr,
            phi_m=ang_arr,
            x0=state["x0"],
            y0=state["y0"],
            n_points=400,
        )
        circ_xy = p1d.Circular_Multipole_Profile_1D(
            state["contour_r0"],
            q=state["q"],
            theta_ell=state["theta_ell"],
            m=m_arr,
            a_m=a_arr,
            theta_m=ang_arr,
            x0=state["x0"],
            y0=state["y0"],
            n_points=400,
        )
        return ell_xy, circ_xy

    def _contour_color(self, state):
        return (state["contour_r"], state["contour_g"], state["contour_b"])

    def _position_shared_colorbar(self):
        bbox = self.ax_circ.get_position()
        cbar_height = 0.78 * bbox.height
        cbar_y0 = bbox.y0 + 0.11 * bbox.height
        cbar_x0 = bbox.x1 + 0.010
        self.ax_cbar.set_position([cbar_x0, cbar_y0, 0.012, cbar_height])

    def _draw_images(self, img_ell, img_circ, extent):
        self.ax_ell.clear()
        self.ax_circ.clear()
        self.ax_cbar.clear()

        vmin = min(np.nanmin(img_ell), np.nanmin(img_circ))
        vmax = max(np.nanmax(img_ell), np.nanmax(img_circ))

        im_ell, norm = AsinhStretchPlot(
            self.ax_ell,
            img_ell,
            a=0.1,
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=extent,
            cmap="afmhot",
            return_norm=True,
        )
        im_circ = AsinhStretchPlot(
            self.ax_circ,
            img_circ,
            a=0.1,
            origin="lower",
            extent=extent,
            cmap="afmhot",
            norm=norm,
        )
        self.shared_colorbar = self.fig.colorbar(im_ell, cax=self.ax_cbar)
        self._position_shared_colorbar()

        for ax in (self.ax_ell, self.ax_circ):
            ax.set_xticks([-1, 0, 1])
            ax.set_xticklabels(["-1", "0", "+1"])
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels(["-1", "0", "+1"])
            ax.set_xlabel("x (arcsec)")
            ax.set_aspect("equal")

        self.ax_ell.set_ylabel("y (arcsec)")
        self.ax_circ.set_ylabel("")
        self.ax_ell.set_title("Elliptical Multipole", fontsize=12)
        self.ax_circ.set_title("Circular Multipole", fontsize=12)

        self.ell_line = None
        self.circ_line = None

    def _draw_info_panel(self, state, highlight_key=None, movie_mode=False):
        self.ax_info.clear()
        self.ax_info.set_axis_off()

        y = 0.98
        dy = 0.070 if movie_mode else 0.055
        self.ax_info.text(
            0.02,
            y,
            "Parameters",
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
            transform=self.ax_info.transAxes,
        )
        y -= 1.2 * dy

        for key, label, fmt, include_in_movie in PARAMETER_TEXT_ORDER:
            if movie_mode and not include_in_movie:
                continue
            value = state[key]
            if key == "grid_N":
                rendered = fmt.format(int(round(value)))
            else:
                rendered = fmt.format(value)
            text = f"{label} = {rendered}"
            self.ax_info.text(
                0.02,
                y,
                text,
                ha="left",
                va="top",
                fontsize=12 if movie_mode else 11,
                fontweight="bold" if key == highlight_key else "normal",
                family="monospace",
                transform=self.ax_info.transAxes,
            )
            y -= dy

    def _apply_contours(self, state, redraw=True):
        if not state["show_contour"]:
            for line in (self.ell_line, self.circ_line):
                if line is not None:
                    line.set_visible(False)
            if redraw:
                self.canvas.draw_idle()
            return

        (ell_x, ell_y), (circ_x, circ_y) = self._compute_contours(state)
        color = self._contour_color(state)

        if self.ell_line is None:
            (self.ell_line,) = self.ax_ell.plot(ell_x, ell_y, color=color, lw=1.5)
        else:
            self.ell_line.set_data(ell_x, ell_y)
            self.ell_line.set_color(color)
            self.ell_line.set_visible(True)

        if self.circ_line is None:
            (self.circ_line,) = self.ax_circ.plot(circ_x, circ_y, color=color, lw=1.5)
        else:
            self.circ_line.set_data(circ_x, circ_y)
            self.circ_line.set_color(color)
            self.circ_line.set_visible(True)

        if redraw:
            self.canvas.draw_idle()

    def render_full(
        self,
        state_override=None,
        redraw=True,
        suptitle=None,
        highlight_key=None,
        movie_mode=False,
    ):
        self.status_var.set("Computing...")
        self.root.update_idletasks()

        state = self.collect_state()
        if state_override is not None:
            state.update(state_override)

        img_ell, img_circ, extent = self._compute_images(state)
        self._draw_images(img_ell, img_circ, extent)
        self._draw_info_panel(state, highlight_key=highlight_key, movie_mode=movie_mode)
        self._apply_contours(state, redraw=False)

        self.fig.suptitle("" if suptitle is None else suptitle, fontsize=15)
        self.last_render_state = dict(state)
        self.last_extent = extent

        if redraw:
            self.canvas.draw()
        self.status_var.set("Ready")

    def update_contours_only(self):
        if self.last_render_state is None:
            return
        state = dict(self.last_render_state)
        state.update(self.collect_state())
        self.last_render_state = state
        self._draw_info_panel(state, movie_mode=False)
        self._apply_contours(state, redraw=True)
        self.status_var.set("Updated contour")

    def save_pdf(self):
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(output_dir, "multipole_sersic_gui.pdf")
        self.fig.savefig(output_path, bbox_inches="tight")
        self.status_var.set(f"Saved: {os.path.basename(output_path)}")

    def _capture_frame(self):
        self.canvas.draw()
        width, height = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape((height, width, 4))[..., :3].copy()

    def make_movies(self, modes):
        output_dir = os.path.dirname(os.path.abspath(__file__))
        for mode in modes:
            path = os.path.join(
                output_dir, f"{MOVIE_SETTINGS['output_basename']}_{mode}.mp4"
            )
            self._write_movie(path, transition_mode=mode)
        self.status_var.set("Saved movie files")

    def _write_movie(self, output_path, transition_mode):
        fps = MOVIE_SETTINGS["fps"]
        steps = MOVIE_SETTINGS["transition_steps"]
        state = dict(MOVIE_SETTINGS["initial_state"])

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        try:
            for item in MOVIE_SETTINGS["sequence"]:
                kind = item["kind"]
                if kind == "reset":
                    state = dict(MOVIE_SETTINGS["initial_state"])
                    for _ in range(item["frames"]):
                        self.render_full(
                            state_override=state,
                            redraw=True,
                            suptitle=item["title"],
                            highlight_key=None,
                            movie_mode=True,
                        )
                        writer.append_data(self._capture_frame())
                    continue

                if kind == "hold":
                    for _ in range(item["frames"]):
                        self.render_full(
                            state_override=state,
                            redraw=True,
                            suptitle=item["title"],
                            highlight_key=None,
                            movie_mode=True,
                        )
                        writer.append_data(self._capture_frame())
                    continue

                if kind == "transition":
                    param = item["param"]
                    for i in range(steps):
                        u = 0.0 if steps == 1 else i / (steps - 1)
                        state[param] = interpolate_value(
                            item["start"], item["end"], u, transition_mode
                        )
                        self.render_full(
                            state_override=state,
                            redraw=True,
                            suptitle=item["title"],
                            highlight_key=item["highlight"],
                            movie_mode=True,
                        )
                        writer.append_data(self._capture_frame())
                    state[param] = item["end"]
        finally:
            writer.close()


def build_app():
    root = tk.Tk()
    return root, MultipoleSeriscGUI(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-movie", action="store_true")
    parser.add_argument(
        "--movie-mode",
        choices=["linear", "smooth", "all"],
        default="all",
    )
    args = parser.parse_args()

    root, app = build_app()
    if args.make_movie:
        root.withdraw()
        if args.movie_mode == "all":
            modes = MOVIE_SETTINGS["transition_modes"]
        else:
            modes = [args.movie_mode]
        app.make_movies(modes)
        root.destroy()
        return

    root.mainloop()


if __name__ == "__main__":
    main()
