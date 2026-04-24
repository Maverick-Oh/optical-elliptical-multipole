import datetime as dt
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optical_elliptical_multipole.plotting import AsinhStretchPlot


FITS_PATH = PROJECT_ROOT / "data" / "HDUL_test6-1000" / "5519-SCI.fits"
DEFAULT_FULL_PNG_PATH = (
    PROJECT_ROOT / "data" / "HDUL_test6-1000" / "5519-SCI-full.png"
)


class Asinh5519SciGui:
    def __init__(self, root):
        self.root = root
        self.root.title("5519-SCI Asinh Stretch")
        self.root.geometry("1200x900")

        self.raw_data = fits.getdata(FITS_PATH)
        background_pixels = self.raw_data[np.isfinite(self.raw_data) & (self.raw_data != 0)]
        if background_pixels.size == 0:
            raise ValueError("No finite nonzero pixels are available for background estimation.")

        self.background_mean, self.background, self.background_std = sigma_clipped_stats(
            background_pixels, sigma=3.0, maxiters=None
        )
        self.data = self.raw_data - self.background
        self.height, self.width = self.data.shape
        self.a_value = tk.DoubleVar(value=0.1)
        self.status_text = tk.StringVar(
            value=f"Loaded {FITS_PATH.name}; subtracted background={self.background:.6g}"
        )

        self.controls_frame = tk.Frame(root, padx=10, pady=10)
        self.controls_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.figure_frame = tk.Frame(root, padx=10, pady=10)
        self.figure_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls()
        self._build_figure()
        self.run_plot()

    def _build_controls(self):
        tk.Label(self.controls_frame, text="Asinh stretch parameter a").pack(
            anchor=tk.W
        )

        slider_row = tk.Frame(self.controls_frame)
        slider_row.pack(fill=tk.X, pady=(4, 12))

        self.a_label = tk.Label(slider_row, text=f"{self.a_value.get():.4f}", width=10)
        self.a_label.pack(side=tk.RIGHT)

        self.a_slider = tk.Scale(
            slider_row,
            from_=0.0001,
            to=0.1,
            resolution=0.0001,
            orient=tk.HORIZONTAL,
            variable=self.a_value,
            length=260,
            command=self._update_a_label,
        )
        self.a_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(self.controls_frame, text="Run", command=self.run_plot).pack(
            fill=tk.X, pady=(0, 8)
        )
        tk.Button(
            self.controls_frame, text="Save as PDF", command=self.save_as_pdf
        ).pack(fill=tk.X, pady=(0, 8))
        tk.Button(
            self.controls_frame,
            text="Save as full PNG",
            command=self.save_as_full_png,
        ).pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            self.controls_frame,
            text=f"Image size: {self.width} x {self.height} px",
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(12, 4))
        tk.Label(
            self.controls_frame,
            text=(
                "Background removed with 3-sigma clipping\n"
                f"median: {self.background:.6g}\n"
                f"std: {self.background_std:.6g}"
            ),
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(4, 4))
        tk.Label(
            self.controls_frame,
            textvariable=self.status_text,
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=280,
        ).pack(fill=tk.X, pady=(4, 0))

    def _build_figure(self):
        self.fig = Figure(figsize=(7.5, 9.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.figure_frame)
        self.toolbar.update()

    def _update_a_label(self, _value=None):
        self.a_label.configure(text=f"{self.a_value.get():.4f}")

    def _draw_on_axis(self, axis):
        axis.clear()
        axis.set_axis_off()
        AsinhStretchPlot(
            axis,
            self.data,
            a=self.a_value.get(),
            origin="lower",
            cmap="binary",
            interpolation="nearest",
            aspect="equal",
            vmin=0,
        )

    def run_plot(self):
        self._draw_on_axis(self.ax)
        self.fig.tight_layout(pad=0)
        self.canvas.draw_idle()
        self.status_text.set(f"Rendered with a={self.a_value.get():.4f}")

    def save_as_pdf(self):
        selected_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save figure as PDF",
            initialdir=str(DEFAULT_FULL_PNG_PATH.parent),
            initialfile="5519-SCI-full.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
        )
        if not selected_path:
            return

        output_path = self._timestamped_pdf_path(Path(selected_path))
        self._draw_on_axis(self.ax)
        self.fig.tight_layout(pad=0)
        self.fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0)
        self.canvas.draw_idle()
        self.status_text.set(f"Saved PDF: {output_path}")

    def save_as_full_png(self):
        selected_path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save full-resolution PNG",
            initialdir=str(DEFAULT_FULL_PNG_PATH.parent),
            initialfile=DEFAULT_FULL_PNG_PATH.name,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
        )
        if not selected_path:
            return

        output_path = Path(selected_path)
        self._save_full_resolution_png(output_path)
        self.status_text.set(f"Saved full PNG: {output_path}")
        messagebox.showinfo("Saved", f"Saved full-resolution PNG:\n{output_path}")

    def _save_full_resolution_png(self, output_path):
        dpi = 100
        full_fig = Figure(
            figsize=(self.width / dpi, self.height / dpi),
            dpi=dpi,
            frameon=False,
        )
        full_ax = full_fig.add_axes([0, 0, 1, 1])
        full_ax.set_axis_off()
        AsinhStretchPlot(
            full_ax,
            self.data,
            a=self.a_value.get(),
            origin="lower",
            cmap="binary",
            interpolation="nearest",
            aspect="auto",
            vmin=0,
        )
        full_fig.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)

    @staticmethod
    def _timestamped_pdf_path(path):
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = ".pdf"
        if path.suffix.lower() == suffix:
            return path.with_name(f"{path.stem}_{timestamp}{suffix}")
        return path.with_name(f"{path.name}_{timestamp}{suffix}")


def main():
    if not FITS_PATH.exists():
        raise FileNotFoundError(f"FITS file not found: {FITS_PATH}")

    root = tk.Tk()
    Asinh5519SciGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
