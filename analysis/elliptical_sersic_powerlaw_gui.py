import datetime as dt
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_elliptical_multipole.nonjax.intensity_functions import sersic
from optical_elliptical_multipole.nonjax.profiles2D import Elliptical_Profile_2D


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures"
COLORMAP_OPTIONS = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
    "berlin",
    "managua",
    "vanimo",
    "twilight",
    "twilight_shifted",
    "hsv",
    "flag",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "turbo",
    "nipy_spectral",
    "gist_ncar",
]


def elliptical_power_law_kappa(X, Y, b, gamma, q, phi, center_x=0.0, center_y=0.0):
    """Lenstronomy EPLMajorAxis convergence with a q/phi sky rotation."""
    t = gamma - 1.0
    x = X - center_x
    y = Y - center_y
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x_rot = x * cos_phi + y * sin_phi
    y_rot = -x * sin_phi + y * cos_phi
    radius = np.hypot(q * x_rot, y_rot)
    radius = np.maximum(radius, 1e-6)
    kappa = 0.5 * (2.0 - t) * (b / radius) ** t
    return np.nan_to_num(kappa, posinf=1e10, neginf=0.0)


class EllipticalSersicPowerLawGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Elliptical Sersic and Power-Law Profiles")
        self.root.geometry("1700x950")

        self.q_value = tk.DoubleVar(value=0.72)
        self.phi_deg_value = tk.DoubleVar(value=0.0)
        self.light_cmap_value = tk.StringVar(value="afmhot")
        self.mass_cmap_value = tk.StringVar(value="binary")
        self.hide_axis_labels_value = tk.BooleanVar(value=False)
        self.profile_x_log_value = tk.BooleanVar(value=True)
        self.profile_y_log_value = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="Ready")

        self.grid_extent = 4.0
        self.grid_size = 1000
        axis = np.linspace(-self.grid_extent, self.grid_extent, self.grid_size)
        self.X, self.Y = np.meshgrid(axis, axis)

        self.sersic_params = {
            "amplitude": 1.0,
            "R_sersic": 1.0,
            "n_sersic": 4.0,
        }
        self.power_law_params = {
            "b": 1.2,
            "gamma": 2.0,
        }

        self.controls_frame = tk.Frame(root, padx=10, pady=10)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.figure_frame = tk.Frame(root, padx=10, pady=10)
        self.figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls()
        self._build_figure()
        self.run_plot()

    def _build_controls(self):
        tk.Label(
            self.controls_frame,
            text="Shared Ellipse Parameters",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor=tk.W, pady=(0, 8))

        self._add_slider(
            label="q",
            variable=self.q_value,
            from_=0.2,
            to=1.0,
            resolution=0.01,
        )
        self._add_slider(
            label="phi [deg]",
            variable=self.phi_deg_value,
            from_=-90.0,
            to=90.0,
            resolution=1.0,
        )
        self._add_colormap_menu("Light cmap", self.light_cmap_value)
        self._add_colormap_menu("Mass cmap", self.mass_cmap_value)
        tk.Checkbutton(
            self.controls_frame,
            text="Remove axis ticks and labels",
            variable=self.hide_axis_labels_value,
        ).pack(anchor=tk.W, pady=(8, 0))
        tk.Checkbutton(
            self.controls_frame,
            text="1D profiles: log x-axis",
            variable=self.profile_x_log_value,
        ).pack(anchor=tk.W, pady=(4, 0))
        tk.Checkbutton(
            self.controls_frame,
            text="1D profiles: log y-axis",
            variable=self.profile_y_log_value,
        ).pack(anchor=tk.W, pady=(4, 0))

        tk.Button(self.controls_frame, text="Run", command=self.run_plot).pack(
            fill=tk.X, pady=(12, 8)
        )
        tk.Button(
            self.controls_frame, text="Save as PDF", command=self.save_as_pdf
        ).pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            self.controls_frame,
            text=(
                "Representative fixed parameters\n"
                "Light: Sersic n=4, R_sersic=1 arcsec\n"
                "Mass: EPL b=1.2 arcsec, gamma=2\n"
                "Grid: 8 x 8 arcsec"
            ),
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(16, 8))

        tk.Label(
            self.controls_frame,
            textvariable=self.status_text,
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=280,
        ).pack(fill=tk.X, pady=(8, 0))

    def _add_slider(self, label, variable, from_, to, resolution):
        row = tk.Frame(self.controls_frame)
        row.pack(fill=tk.X, pady=5)
        tk.Label(row, text=label, width=10, anchor=tk.W).pack(side=tk.LEFT)
        value_label = tk.Label(row, text=f"{variable.get():.3g}", width=8)
        value_label.pack(side=tk.RIGHT)

        def update_label(_value=None):
            value_label.configure(text=f"{variable.get():.3g}")

        slider = tk.Scale(
            row,
            from_=from_,
            to=to,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=variable,
            length=260,
            command=update_label,
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _add_colormap_menu(self, label, variable):
        row = tk.Frame(self.controls_frame)
        row.pack(fill=tk.X, pady=5)
        tk.Label(row, text=label, width=10, anchor=tk.W).pack(side=tk.LEFT)
        menu = tk.OptionMenu(row, variable, *COLORMAP_OPTIONS)
        menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_figure(self):
        self.fig = Figure(figsize=(13.5, 9.5), dpi=100)
        outer_gridspec = GridSpec(
            2,
            2,
            figure=self.fig,
            width_ratios=[1.0, 1.0],
            wspace=0.28,
            hspace=0.35,
            left=0.07,
            right=0.97,
            bottom=0.07,
            top=0.95,
        )
        light_gridspec = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gridspec[0, 0], width_ratios=[1.0, 0.04], wspace=0.03
        )
        mass_gridspec = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gridspec[1, 0], width_ratios=[1.0, 0.04], wspace=0.03
        )
        self.light_axis = self.fig.add_subplot(light_gridspec[0, 0])
        self.light_cbar_axis = self.fig.add_subplot(light_gridspec[0, 1])
        self.mass_axis = self.fig.add_subplot(mass_gridspec[0, 0])
        self.mass_cbar_axis = self.fig.add_subplot(mass_gridspec[0, 1])
        self.sersic_profile_axis = self.fig.add_subplot(outer_gridspec[0, 1])
        self.epl_profile_axis = self.fig.add_subplot(outer_gridspec[1, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.figure_frame)
        self.toolbar.update()

    def _profiles(self):
        q = self.q_value.get()
        phi = np.deg2rad(self.phi_deg_value.get())

        light = Elliptical_Profile_2D(
            self.X,
            self.Y,
            sersic,
            q=q,
            theta_ell=phi,
            **self.sersic_params,
        )
        mass = elliptical_power_law_kappa(
            self.X,
            self.Y,
            q=q,
            phi=phi,
            **self.power_law_params,
        )
        return light, mass

    def run_plot(self):
        light, mass = self._profiles()
        self.cmap_fallback_messages = []
        self._draw_profile(
            self.light_axis,
            self.light_cbar_axis,
            light,
            "Elliptical Sersic Light Profile",
            cmap=self.light_cmap_value.get(),
            colorbar_label="surface brightness [arbitrary]",
        )
        self._draw_profile(
            self.mass_axis,
            self.mass_cbar_axis,
            mass,
            "Elliptical Power-Law Mass Profile",
            cmap=self.mass_cmap_value.get(),
            colorbar_label=r"$\kappa$",
        )
        self._draw_sersic_radius_profile(self.sersic_profile_axis)
        self._draw_epl_radius_profile(self.epl_profile_axis)
        self.canvas.draw_idle()
        status = f"Rendered q={self.q_value.get():.2f}, phi={self.phi_deg_value.get():.0f} deg"
        if self.cmap_fallback_messages:
            status += "; " + "; ".join(self.cmap_fallback_messages)
        self.status_text.set(status)

    def _draw_profile(self, axis, colorbar_axis, image, title, cmap, colorbar_label):
        axis.clear()
        colorbar_axis.clear()
        cmap = self._resolve_colormap(cmap)
        vmax = np.nanpercentile(image, 99.5)
        norm = ImageNormalize(vmin=0.0, vmax=vmax, stretch=AsinhStretch(a=0.08))
        image_artist = axis.imshow(
            image,
            origin="lower",
            cmap=cmap,
            norm=norm,
            extent=[
                -self.grid_extent,
                self.grid_extent,
                -self.grid_extent,
                self.grid_extent,
            ],
            interpolation="nearest",
        )
        axis.set_title(title)
        if self.hide_axis_labels_value.get():
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel("")
            axis.set_ylabel("")
        else:
            axis.set_xlabel("x [arcsec]")
            axis.set_ylabel("y [arcsec]")
        axis.set_aspect("equal")
        colorbar = self.fig.colorbar(image_artist, cax=colorbar_axis)
        colorbar.set_label(colorbar_label)

    def _draw_sersic_radius_profile(self, axis):
        axis.clear()
        radius = np.geomspace(0.02, self.grid_extent, 500)
        n_values = np.arange(1, 11)
        cmap = matplotlib.cm.get_cmap("viridis")
        for n_sersic in n_values:
            intensity = sersic(
                radius,
                amplitude=self.sersic_params["amplitude"],
                R_sersic=self.sersic_params["R_sersic"],
                n_sersic=float(n_sersic),
            )
            color = cmap((n_sersic - 1) / (len(n_values) - 1))
            axis.plot(radius, intensity, color=color, lw=1.6)
            if n_sersic in (1, 10):
                idx = 50 if n_sersic == 1 else 50
                axis.text(
                    radius[idx],
                    intensity[idx],
                    f"n={n_sersic}",
                    color=color,
                    ha="left",
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5},
                )
        self._format_radius_profile_axis(
            axis,
            title="Sersic Radius Profile",
            ylabel="intensity [arbitrary]",
        )

    def _draw_epl_radius_profile(self, axis):
        axis.clear()
        radius = np.geomspace(0.02, self.grid_extent, 500)
        gamma_values = [1.0, 1.5, 2.0, 2.5]
        cmap = matplotlib.cm.get_cmap("plasma")
        for idx_gamma, gamma in enumerate(gamma_values):
            density = self._circular_power_law_density(radius, gamma)
            color = cmap(idx_gamma / (len(gamma_values) - 1))
            axis.plot(radius, density, color=color, lw=1.6)
            if gamma in (1.0, 2.0):
                idx = {1.0: 50, 2.0: 50}[gamma]
                axis.text(
                    radius[idx],
                    density[idx],
                    f"gamma={gamma:.1f}",
                    color=color,
                    ha="left",
                    va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5},
                )
        self._format_radius_profile_axis(
            axis,
            title="EPL Radius Profile",
            ylabel=r"$\kappa(r)$",
        )

    def _format_radius_profile_axis(self, axis, title, ylabel):
        axis.set_title(title)
        axis.set_xscale("log" if self.profile_x_log_value.get() else "linear")
        axis.set_yscale("log" if self.profile_y_log_value.get() else "linear")
        axis.grid(True, which="both", alpha=0.25)
        if self.hide_axis_labels_value.get():
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_xlabel("")
            axis.set_ylabel("")
        else:
            axis.set_xlabel("radius [arcsec]")
            axis.set_ylabel(ylabel)

    def _circular_power_law_density(self, radius, gamma):
        t = gamma - 1.0
        density = 0.5 * (2.0 - t) * (self.power_law_params["b"] / radius) ** t
        if self.profile_y_log_value.get():
            return np.maximum(density, 1e-12)
        return density

    def _resolve_colormap(self, cmap):
        try:
            matplotlib.cm.get_cmap(cmap)
        except ValueError:
            fallback = "viridis"
            self.cmap_fallback_messages.append(
                f"{cmap} unavailable in this Matplotlib; used {fallback}"
            )
            return fallback
        return cmap

    def save_as_pdf(self):
        selected_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save figure as PDF",
            initialdir=str(DEFAULT_OUTPUT_DIR),
            initialfile="elliptical_sersic_powerlaw.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not selected_path:
            return

        output_path = self._timestamped_pdf_path(Path(selected_path))
        self.run_plot()
        self.fig.savefig(output_path, format="pdf", bbox_inches="tight")
        self.status_text.set(f"Saved PDF: {output_path}")

    @staticmethod
    def _timestamped_pdf_path(path):
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = ".pdf"
        if path.suffix.lower() == suffix:
            return path.with_name(f"{path.stem}_{timestamp}{suffix}")
        return path.with_name(f"{path.name}_{timestamp}{suffix}")


def main():
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    EllipticalSersicPowerLawGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
