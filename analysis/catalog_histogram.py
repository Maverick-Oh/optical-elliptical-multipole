import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

def launch_cosmos_hist_gui(csv_path="../data/HDUL_ALL/cosmos_sample_N=4433_ALL.csv"):
    """
    Read COSMOS sample CSV and launch a Tkinter GUI with:
      1) histogram of r50 converted to arcsec
      2) histogram of r_gim2d
      3) histogram of n_sersic_gim2d (or sersic_n_gim2d if that is the actual column name)
      4) 2D histogram of r_gim2d vs n_sersic_gim2d with cell annotations

    Notes
    -----
    - r50 is converted from pixel to arcsec using:
          r50_arcsec = r50 * 0.03
    - Rows with blank/missing r_gim2d and Sérsic-n values are dropped only
      for the GIM2D-based plots.
    - The 2D histogram text in each cell shows:
          count
          % of valid GIM2D subsample
          % of full sample (N=4433)
    """

    # -----------------------------
    # 1) Read CSV
    # -----------------------------
    df = pd.read_csv(csv_path)
    total_N = len(df)

    # -----------------------------
    # 2) Resolve column names
    # -----------------------------
    r50_col = "r50"
    r_gim2d_col = "r_gim2d"

    possible_n_cols = ["n_sersic_gim2d", "sersic_n_gim2d"]
    n_gim2d_col = None
    for c in possible_n_cols:
        if c in df.columns:
            n_gim2d_col = c
            break

    if r50_col not in df.columns:
        raise ValueError(f"Column '{r50_col}' not found in CSV.")

    if r_gim2d_col not in df.columns:
        raise ValueError(f"Column '{r_gim2d_col}' not found in CSV.")

    if n_gim2d_col is None:
        raise ValueError(
            "Could not find Sérsic index column. "
            "Expected one of: 'n_sersic_gim2d', 'sersic_n_gim2d'."
        )

    # -----------------------------
    # 3) Convert to numeric
    # -----------------------------
    df[r50_col] = pd.to_numeric(df[r50_col], errors="coerce")
    df[r_gim2d_col] = pd.to_numeric(df[r_gim2d_col], errors="coerce")
    df[n_gim2d_col] = pd.to_numeric(df[n_gim2d_col], errors="coerce")

    # -----------------------------
    # 4) Derived column: r50_arcsec
    # -----------------------------
    df["r50_arcsec"] = df[r50_col] * 0.03

    r50_arcsec = df["r50_arcsec"].dropna().to_numpy()

    valid_gim2d = df[[r_gim2d_col, n_gim2d_col]].dropna().copy()
    r_gim2d = valid_gim2d[r_gim2d_col].to_numpy()
    n_gim2d = valid_gim2d[n_gim2d_col].to_numpy()
    valid_N = len(valid_gim2d)

    print(f"Total sample size              : {total_N}")
    print(f"Valid r50 values               : {len(r50_arcsec)}")
    print(f"Valid GIM2D rows (both present): {valid_N}")

    # -----------------------------
    # 5) Build Tkinter GUI
    # -----------------------------
    root = tk.Tk()
    root.title("COSMOS Histogram Explorer")

    main_frame = ttk.Frame(root, padding=8)
    main_frame.pack(fill="both", expand=True)

    controls_frame = ttk.Frame(main_frame, padding=8)
    controls_frame.pack(side="left", fill="y")

    plot_frame = ttk.Frame(main_frame, padding=8)
    plot_frame.pack(side="right", fill="both", expand=True)

    # -----------------------------
    # 6) Slider variables
    # -----------------------------
    bins_r50_var = tk.IntVar(value=30)
    bins_rgim_var = tk.IntVar(value=30)
    bins_ngim_var = tk.IntVar(value=30)
    bins2d_x_var = tk.IntVar(value=4)
    bins2d_y_var = tk.IntVar(value=5)

    # -----------------------------
    # 7) Matplotlib figure
    # -----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(hspace=0.35, wspace=0.28, right=0.88)

    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # Create one dedicated colorbar axis only once
    cbar_ax = fig.add_axes([0.90, 0.11, 0.02, 0.33])  # [left, bottom, width, height]
    cbar = None

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    # -----------------------------
    # 8) Plot updater
    # -----------------------------
    def update_plots(*args):
        nonlocal cbar

        for ax in axes.ravel():
            ax.clear()

        # ---- Histogram 1: r50_arcsec ----
        bins_r50 = bins_r50_var.get()
        ax1.hist(r50_arcsec, bins=bins_r50, edgecolor="black")
        ax1.set_title("Histogram of r50_arcsec")
        ax1.set_xlabel("r50_arcsec")
        ax1.set_ylabel("Count")
        ax1.grid(alpha=0.25)

        # ---- Histogram 2: r_gim2d ----
        bins_rgim = bins_rgim_var.get()
        ax2.hist(r_gim2d, bins=bins_rgim, edgecolor="black")
        ax2.set_title("Histogram of r_gim2d")
        ax2.set_xlabel("r_gim2d")
        ax2.set_ylabel("Count")
        ax2.grid(alpha=0.25)

        # ---- Histogram 3: n_gim2d ----
        bins_ngim = bins_ngim_var.get()
        ax3.hist(n_gim2d, bins=bins_ngim, edgecolor="black")
        ax3.set_title(f"Histogram of {n_gim2d_col}")
        ax3.set_xlabel(n_gim2d_col)
        ax3.set_ylabel("Count")
        ax3.grid(alpha=0.25)

        # ---- 2D Histogram ----
        bins2d_x = bins2d_x_var.get()
        bins2d_y = bins2d_y_var.get()

        h, xedges, yedges, img = ax4.hist2d(
            r_gim2d,
            n_gim2d,
            bins=[bins2d_x, bins2d_y],
            cmap="Greens"
        )

        ax4.set_title(f"2D Histogram: r_gim2d vs {n_gim2d_col}")
        ax4.set_xlabel("r_gim2d")
        ax4.set_ylabel(n_gim2d_col)

        # Update colorbar without recreating its axis
        if cbar is None:
            cbar = fig.colorbar(img, cax=cbar_ax)
            cbar.set_label("Count")
        else:
            cbar.update_normal(img)
            cbar.set_label("Count")

        # Annotate each cell
        for ix in range(len(xedges) - 1):
            for iy in range(len(yedges) - 1):
                count = int(h[ix, iy])

                x_center = 0.5 * (xedges[ix] + xedges[ix + 1])
                y_center = 0.5 * (yedges[iy] + yedges[iy + 1])

                pct_valid = 100.0 * count / valid_N if valid_N > 0 else 0.0
                pct_full = 100.0 * count / total_N if total_N > 0 else 0.0
                est_out_of_4433 = pct_valid / 100.0 * total_N if total_N > 0 else 0.0

                label = (
                    f"{count}\n"
                    f"{pct_valid:.1f}% valid\n"
                    f"{pct_full:.1f}% of 4433\n"
                    f"~{est_out_of_4433:.0f}/4433"
                )

                ax4.text(
                    x_center,
                    y_center,
                    label,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8
                )

        ax4.text(
            0.01,
            0.99,
            f"Valid GIM2D rows: {valid_N}/{total_N}",
            transform=ax4.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
        )

        canvas.draw_idle()

    def save_as_pdf():
        # timestamp for unique filename
        ts = time.strftime("%Y%m%d_%H%M%S")

        filename = (
            f"cosmos_hist_"
            f"r50b{bins_r50_var.get()}_"
            f"rg{bins_rgim_var.get()}_"
            f"ng{bins_ngim_var.get()}_"
            f"2d{bins2d_x_var.get()}x{bins2d_y_var.get()}_"
            f"{ts}.pdf"
        )

        fig.savefig(filename, format="pdf", bbox_inches="tight")
        print(f"[Saved] {filename}")

    # -----------------------------
    # 9) Controls
    # -----------------------------
    ttk.Label(
        controls_frame,
        text="Histogram Controls",
        font=("Arial", 11, "bold")
    ).pack(anchor="w", pady=(0, 8))

    ttk.Button(
        controls_frame,
        text="Save as PDF",
        command=save_as_pdf
    ).pack(fill="x", pady=4)

    def add_slider(parent, label, variable, from_, to_, resolution=1):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=4)

        ttk.Label(frame, text=label).pack(anchor="w")
        scale = tk.Scale(
            frame,
            from_=from_,
            to=to_,
            orient="horizontal",
            variable=variable,
            resolution=resolution,
            showvalue=True,
            command=lambda _=None: update_plots()
        )
        scale.pack(fill="x")
        return scale

    add_slider(controls_frame, "Bins: r50_arcsec", bins_r50_var, 5, 100, 1)
    add_slider(controls_frame, "Bins: r_gim2d", bins_rgim_var, 5, 100, 1)
    add_slider(controls_frame, f"Bins: {n_gim2d_col}", bins_ngim_var, 5, 100, 1)
    add_slider(controls_frame, "2D bins in x (r_gim2d)", bins2d_x_var, 2, 15, 1)
    add_slider(controls_frame, f"2D bins in y ({n_gim2d_col})", bins2d_y_var, 2, 15, 1)

    ttk.Separator(controls_frame, orient="horizontal").pack(fill="x", pady=8)

    def print_current_info():
        print("=" * 60)
        print(f"Total rows in CSV              : {total_N}")
        print(f"Valid r50_arcsec rows          : {len(r50_arcsec)}")
        print(f"Valid GIM2D rows               : {valid_N}")
        print(f"Current r50 bins               : {bins_r50_var.get()}")
        print(f"Current r_gim2d bins           : {bins_rgim_var.get()}")
        print(f"Current {n_gim2d_col} bins     : {bins_ngim_var.get()}")
        print(f"Current 2D x bins              : {bins2d_x_var.get()}")
        print(f"Current 2D y bins              : {bins2d_y_var.get()}")

    ttk.Button(
        controls_frame,
        text="Print current settings",
        command=print_current_info
    ).pack(fill="x", pady=4)

    ttk.Button(
        controls_frame,
        text="Refresh plot",
        command=update_plots
    ).pack(fill="x", pady=4)

    ttk.Button(
        controls_frame,
        text="Quit",
        command=root.destroy
    ).pack(fill="x", pady=(12, 4))

    update_plots()
    root.mainloop()


if __name__ == "__main__":
    launch_cosmos_hist_gui("../data/HDUL_ALL/cosmos_sample_N=4433_ALL.csv")