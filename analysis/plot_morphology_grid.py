#!/usr/bin/env python
"""
plot_morphology_grid.py
=======================
Generate a grid figure comparing elliptical vs circular multipole morphology
across different axis ratios q, for both positive and negative amplitudes.

Produces one figure per multipole order m (default: m=3 and m=4).

Usage
-----
python plot_morphology_grid.py
python plot_morphology_grid.py --a-m 0.15 --m-list 4
python plot_morphology_grid.py --q-list 0.3 0.5 0.7 1.0
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tools_fitting import (
    simulate_model_elliptical_multipole,
    simulate_model_circular_multipole,
    build_arcsec_grid,
)
from optical_elliptical_multipole.nonjax.profiles1D import (
    Elliptical_Multipole_Profile_1D,
    Circular_Multipole_Profile_1D,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)


def make_morphology_grid(m_order, a_m_val=0.1, q_list=None,
                         n_sersic=2.0, R_sersic=1.0, amplitude=1.0,
                         theta_ell=0.0, phi_m=0.0, x0=0.0, y0=0.0,
                         background=0.001, pix_scale=0.03,
                         supersample=5, out_dir=None):
    """
    Create a 5-row × len(q_list)-column grid figure.

    Rows:  Unperturbed | Elliptical a_m>0 | Elliptical a_m<0 | Circular a_m>0 | Circular a_m<0
    Cols:  one per q value
    """
    if q_list is None:
        q_list = [0.4, 0.7, 1.0]
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, "data", "morphology_grid")
    os.makedirs(out_dir, exist_ok=True)

    n_cols = len(q_list)
    n_rows = 5

    # Amplitude label: use m subscript
    amp_label = f"a_{m_order}"

    # --- Build grid ---
    half_extent = 2.5  # arcsec
    nx = int(np.ceil(2 * half_extent / pix_scale))
    if nx % 2 == 0:
        nx += 1
    ny = nx

    ss = supersample
    X_f, Y_f, _ = build_arcsec_grid((ny * ss, nx * ss), pixscale=pix_scale / ss)
    _, _, extent = build_arcsec_grid((ny, nx), pixscale=pix_scale)

    m = np.array([m_order])
    phi_m_arr = np.array([phi_m])

    # Row definitions: (model_type, a_m_multiplier)
    # Row 0 = unperturbed elliptical Sérsic (a_m = 0)
    row_defs = [
        ('elliptical',  0),   # unperturbed
        ('elliptical', +1),
        ('elliptical', -1),
        ('circular',   +1),
        ('circular',   -1),
    ]

    # --- Create figure ---
    # Extra space on left for row labels, top for column headers
    fig_width = 3.2 * n_cols + 1.8   # extra left margin for labels
    fig_height = 3.2 * n_rows + 0.8  # extra top margin for q labels

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Define grid spec with left margin for labels
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        left=0.14, right=0.96, bottom=0.04, top=0.92,
        wspace=0.08, hspace=0.08,
    )

    # Compute shared color limits from all images
    all_images = []
    for row_idx, (model_type, sign) in enumerate(row_defs):
        for col_idx, q_val in enumerate(q_list):
            a_m_arr = np.array([sign * a_m_val])
            if model_type == 'elliptical':
                I_fine = simulate_model_elliptical_multipole(
                    X_f, Y_f, n_sersic=n_sersic, R_sersic=R_sersic,
                    amplitude=amplitude, q=q_val, theta_ell=theta_ell,
                    m=m, a_m=a_m_arr, phi_m=phi_m_arr,
                    x0=x0, y0=y0, background=background,
                )
            else:
                I_fine = simulate_model_circular_multipole(
                    X_f, Y_f, n_sersic=n_sersic, R_sersic=R_sersic,
                    amplitude=amplitude, q=q_val, theta_ell=theta_ell,
                    m=m, a_m=a_m_arr, theta_m=phi_m_arr,
                    x0=x0, y0=y0, background=background,
                )
            I_coarse = I_fine.reshape(ny, ss, nx, ss).mean(axis=(1, 3))
            all_images.append(I_coarse)

    vmin = min(img.min() for img in all_images)
    vmax = max(img.max() for img in all_images)

    # Apply asinh stretch manually for compatibility with older matplotlib
    linear_width = 0.05 * vmax
    all_images_stretched = [np.arcsinh(img / linear_width) for img in all_images]
    s_min = np.arcsinh(vmin / linear_width)
    s_max = np.arcsinh(vmax / linear_width)
    norm = mcolors.Normalize(vmin=s_min, vmax=s_max)

    img_idx = 0
    for row_idx, (model_type, sign) in enumerate(row_defs):
        for col_idx, q_val in enumerate(q_list):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            I_stretched = all_images_stretched[img_idx]
            img_idx += 1

            ax.imshow(I_stretched, origin='lower', extent=extent,
                      norm=norm, aspect='equal')

            # Overlay 1D contour (skip for unperturbed row)
            a_m_arr = np.array([sign * a_m_val])
            if sign == 0:
                # Unperturbed: draw the plain ellipse contour (a_m=0)
                x_c, y_c = Elliptical_Multipole_Profile_1D(
                    r0=R_sersic, q=q_val, theta_ell=theta_ell,
                    m=m, a_m=np.array([0.0]), phi_m=phi_m_arr,
                    x0=x0, y0=y0, return_type='xy',
                    include_end=True, n_points=500,
                )
            elif model_type == 'elliptical':
                x_c, y_c = Elliptical_Multipole_Profile_1D(
                    r0=R_sersic, q=q_val, theta_ell=theta_ell,
                    m=m, a_m=a_m_arr, phi_m=phi_m_arr,
                    x0=x0, y0=y0, return_type='xy',
                    include_end=True, n_points=500,
                )
            else:
                x_c, y_c = Circular_Multipole_Profile_1D(
                    r0=R_sersic, q=q_val, theta_ell=theta_ell,
                    m=m, a_m=a_m_arr, theta_m=phi_m_arr,
                    x0=x0, y0=y0, return_type='xy',
                    include_end=True, n_points=500,
                )
            ax.plot(x_c, y_c, ls='--', color=(0, 1, 0), lw=1.2)

            # Axis formatting
            lim = 1.8
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{int(v)}'))

            if row_idx < n_rows - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('$x$', fontsize=10)
            if col_idx > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel('$y$', fontsize=10)

            ax.tick_params(labelsize=8)

            # Column header (q value) on top row
            if row_idx == 0:
                ax.set_title(f'$q = {q_val}$', fontsize=12, fontweight='bold')

    # --- Row labels (left side) ---
    fig.canvas.draw()

    # Row 0: unperturbed — label on left
    y_row0 = 0.5 * (gs[0, 0].get_position(fig).y0 + gs[0, 0].get_position(fig).y1)
    fig.text(0.04, y_row0, f'${amp_label} = 0$\n(unperturbed)',
             fontsize=11, ha='center', va='center', rotation=90)

    # Group label: "Elliptical Multipole" for rows 1-2
    y_ellip = 0.5 * (gs[1, 0].get_position(fig).y0 + gs[2, 0].get_position(fig).y1)
    fig.text(0.015, y_ellip, 'Elliptical\nMultipole', fontsize=12, fontweight='bold',
             ha='center', va='center', rotation=90)

    # Group label: "Circular Multipole" for rows 3-4
    y_circ = 0.5 * (gs[3, 0].get_position(fig).y0 + gs[4, 0].get_position(fig).y1)
    fig.text(0.015, y_circ, 'Circular\nMultipole', fontsize=12, fontweight='bold',
             ha='center', va='center', rotation=90)

    # Sub-labels with exact values (rows 1-4 only)
    for row_idx, (_, sign) in enumerate(row_defs):
        if row_idx == 0:
            continue  # unperturbed row already labelled
        y_row = 0.5 * (gs[row_idx, 0].get_position(fig).y0 +
                        gs[row_idx, 0].get_position(fig).y1)
        if sign > 0:
            sign_text = f'${amp_label} = +{a_m_val}$'
        else:
            sign_text = f'${amp_label} = -{a_m_val}$'
        fig.text(0.065, y_row, sign_text, fontsize=11,
                 ha='center', va='center', rotation=90)

    # Separator lines
    # Between unperturbed (row 0) and elliptical group (rows 1-2)
    y_sep0 = 0.5 * (gs[0, 0].get_position(fig).y0 + gs[1, 0].get_position(fig).y1)
    fig.add_artist(plt.Line2D(
        [0.05, 0.96], [y_sep0, y_sep0],
        transform=fig.transFigure, color='gray', lw=1.0, ls='-',
    ))
    # Between elliptical group (rows 1-2) and circular group (rows 3-4)
    y_sep1 = 0.5 * (gs[2, 0].get_position(fig).y0 + gs[3, 0].get_position(fig).y1)
    fig.add_artist(plt.Line2D(
        [0.05, 0.96], [y_sep1, y_sep1],
        transform=fig.transFigure, color='gray', lw=1.0, ls='-',
    ))

    # Suptitle
    fig.suptitle(f'Multipole Morphology Comparison — $m = {m_order}$',
                 fontsize=14, fontweight='bold', y=0.97)

    # Save
    out_path = os.path.join(out_dir, f"morphology_grid_m{m_order}.pdf")
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate multipole morphology grid figure")
    parser.add_argument("--m-list", type=int, nargs='+', default=[1, 3, 4],
                        help="Multipole orders to plot (default: 3 4)")
    parser.add_argument("--a-m", type=float, default=0.1,
                        help="Multipole amplitude (default: 0.1)")
    parser.add_argument("--q-list", type=float, nargs='+', default=[1.0, 0.7, 0.4],
                        help="Axis ratios for columns (default: 1.0 0.7 0.4)")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "morphology_grid"),
                        help="Output directory")
    args = parser.parse_args()

    for m_order in args.m_list:
        make_morphology_grid(
            m_order=m_order,
            a_m_val=args.a_m,
            q_list=args.q_list,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
