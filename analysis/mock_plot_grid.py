import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import seaborn as sns
from datetime import datetime

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(base_dir, subdir='mock_varying_all'):
    truth_file = os.path.join(base_dir, subdir, 'simulation_truth.csv')
    fit_file = os.path.join(base_dir, subdir, 'fitting_results.csv')

    if not os.path.exists(truth_file) or not os.path.exists(fit_file):
        print(f"Skipping {base_dir}/{subdir} (Files missing)")
        return pd.DataFrame()

    df_truth = pd.read_csv(truth_file)
    df_fit = pd.read_csv(fit_file)

    # Harmonize ID columns
    df_truth['seqid'] = df_truth['seqid'].astype(str)
    if 'sequentialid' in df_fit.columns:
        df_fit['seqid'] = df_fit['sequentialid'].astype(str)
    elif 'id' in df_fit.columns:
        df_fit['seqid'] = df_fit['id'].astype(str)

    # Merge
    df = pd.merge(df_truth, df_fit, on='seqid', suffixes=('_true', '_rec'))

    # Ensure columns exist
    for m in [3, 4]:
        if f'a_m{m}' in df.columns:
            df[f'a_m{m}_true'] = df[f'a_m{m}']
        if f'phi_m{m}' in df.columns:
            df[f'phi_m{m}_true'] = df[f'phi_m{m}']
    if 'R_sersic' in df.columns:
        df['R_sersic_true'] = df['R_sersic']
    if 'n_sersic' in df.columns:
        df['n_sersic_true'] = df['n_sersic']

    return df


def collect_all_dataframes():
    """Load data from the consolidated grid directory."""
    dfs = []
    df1 = load_data('../data', 'mock_grid_validation/mock_varying_all')
    if not df1.empty:
        dfs.append(df1)
    return dfs


# ---------------------------------------------------------------------------
# Static PDF plot
# ---------------------------------------------------------------------------

def _pi_frac_formatter(m):
    """Return a FuncFormatter that renders values as multiples of π/(2m)."""
    unit = np.pi / (2 * m)

    def fmt(x, _pos):
        k = x / unit
        if abs(k) < 1e-6:
            return "0"
        if abs(abs(k) - 1) < 1e-6:
            sign = "−" if k < 0 else ""
            return f"{sign}π/{2*m}"
        if abs(abs(k) - 0.5) < 1e-6:
            sign = "−" if k < 0 else ""
            return f"{sign}π/{4*m}"
        return f"{x:.3f}"

    return ticker.FuncFormatter(fmt)


def plot_grid(dfs, save=True):
    if not dfs:
        print("No valid dataframes found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    if 'a_m3_true' not in df.columns:
        df['a_m3_true'] = df['a_m3']

    df['a_m_group'] = df['a_m3_true'].apply(lambda x: round(x, 4))

    a_m_groups = sorted(df['a_m_group'].unique())
    R_grid = sorted(df['R_sersic_true'].unique())
    n_grid = sorted(df['n_sersic_true'].unique())

    print(f"Found {len(a_m_groups)} a_m groups: {a_m_groups}")

    for m_mode in [3, 4]:
        n_rows = 6
        # 6 columns: 5 data + 1 thin colorbar column
        fig, axes = plt.subplots(n_rows, len(a_m_groups) + 1,
                                 figsize=(4.5 * len(a_m_groups) + 1.2, 4 * n_rows),
                                 gridspec_kw={'width_ratios': [1]*len(a_m_groups) + [0.06]},
                                 constrained_layout=True)
        fig.suptitle(f"Multipole Recovery Grid Validation (m={m_mode})", fontsize=20)

        phi_unit = np.pi / (2 * m_mode)

        for i, a_m in enumerate(a_m_groups):
            dg = df[df['a_m_group'] == a_m].copy()

            dg['res_am'] = dg[f'a_m{m_mode}_best'] - dg[f'a_m{m_mode}_true']
            dg['unc_am'] = dg[f'a_m{m_mode}_err']
            dg['sig_am'] = dg['res_am'] / dg['unc_am']

            raw_diff = dg[f'phi_m{m_mode}_best'] - dg[f'phi_m{m_mode}_true']
            period = 2 * np.pi / m_mode
            dg['res_phi'] = (raw_diff + period / 2) % period - period / 2
            dg['unc_phi'] = dg[f'phi_m{m_mode}_err']
            dg['sig_phi'] = dg['res_phi'] / dg['unc_phi']

            def _pivot(col):
                return dg.pivot_table(index='R_sersic_true',
                                      columns='n_sersic_true',
                                      values=col, aggfunc='mean')

            specs = [
                ('res_am', -0.01, 0.01, 'bwr',
                 f'a_m{m_mode} residual\n(est − true)', '.4f',
                 None),
                ('unc_am', 0, 0.01, 'magma',
                 f'a_m{m_mode} uncertainty (1σ)', '.4f',
                 None),
                ('sig_am', -2, 2, 'bwr',
                 f'a_m{m_mode} σ from truth\n(est − true)/σ', '.2f',
                 None),
                ('res_phi', -phi_unit, phi_unit, 'bwr',
                 f'ϕ_m{m_mode} residual\n(est − true)', None,
                 [-phi_unit, -phi_unit / 2, 0, phi_unit / 2, phi_unit]),
                ('unc_phi', 0, phi_unit, 'magma',
                 f'ϕ_m{m_mode} uncertainty (1σ)', None,
                 [0, phi_unit / 2, phi_unit]),
                ('sig_phi', -2, 2, 'bwr',
                 f'ϕ_m{m_mode} σ from truth\n(est − true)/σ', '.2f',
                 None),
            ]

            for row_idx, (col, vmin, vmax, cmap, title, cfmt, cticks) in enumerate(specs):
                ax = axes[row_idx, i]
                piv = _pivot(col)

                for rr in R_grid:
                    if rr not in piv.index:
                        piv.loc[rr] = np.nan
                piv.sort_index(inplace=True)

                annot_fmt = cfmt if cfmt else '.3f'
                sns.heatmap(piv, ax=ax, cmap=cmap, annot=True,
                            fmt=annot_fmt, vmin=vmin, vmax=vmax,
                            cbar=False)
                ax.set_title(f'a_m={a_m}\n{title}' if row_idx == 0 else title)
                ax.invert_yaxis()
                ax.set_ylabel('R_sersic' if i == 0 else '')
                ax.set_xlabel('n_sersic' if row_idx == n_rows - 1 else '')

        # Fill colorbar column
        phi_unit = np.pi / (2 * m_mode)
        cbar_specs = [
            (-0.01, 0.01, 'bwr', None),
            (0, 0.01, 'magma', None),
            (-2, 2, 'bwr', None),
            (-phi_unit, phi_unit, 'bwr',
             [-phi_unit, -phi_unit/2, 0, phi_unit/2, phi_unit]),
            (0, phi_unit, 'magma',
             [0, phi_unit/2, phi_unit]),
            (-2, 2, 'bwr', None),
        ]
        for row_idx, (vmin, vmax, cmap, cticks) in enumerate(cbar_specs):
            cax = axes[row_idx, -1]
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax)
            if cticks:
                cb.set_ticks(cticks)
                cb.ax.yaxis.set_major_formatter(_pi_frac_formatter(m_mode))

        if save:
            out = f'../data/multipole_grid_validation_m{m_mode}.pdf'
            plt.savefig(out, bbox_inches='tight')
            print(f'Saved {out}')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tkinter interactive GUI
# ---------------------------------------------------------------------------

# Per-criterion colors and x-offsets
CRITERION_STYLES = [
    # (key suffix, color, x_offset, label)
    ('res_am',  'red',    -0.20, 'a_m residual'),
    ('unc_am',  'orange', -0.12, 'a_m uncertainty'),
    ('sig_am',  'gold',   -0.04, 'a_m σ'),
    ('res_phi', 'green',   0.04, 'ϕ_m residual'),
    ('unc_phi', 'dodgerblue', 0.12, 'ϕ_m uncertainty'),
    ('sig_phi', 'purple',  0.20, 'ϕ_m σ'),
]


def launch_gui(dfs):
    matplotlib.use('TkAgg')
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    if not dfs:
        print("No data to display.")
        return

    df = pd.concat(dfs, ignore_index=True)
    if 'a_m3_true' not in df.columns:
        df['a_m3_true'] = df['a_m3']
    df['a_m_group'] = df['a_m3_true'].apply(lambda x: round(x, 4))

    a_m_groups = sorted(df['a_m_group'].unique())
    R_grid = sorted(df['R_sersic_true'].unique())
    n_grid = sorted(df['n_sersic_true'].unique())

    n_R = len(R_grid)
    n_n = len(n_grid)

    # ------------------------------------------------------------------
    # Pre-compute all metric arrays once
    # ------------------------------------------------------------------
    precomp = {}
    raw_per_cell = {}

    metric_names = ['res_am', 'unc_am', 'sig_am', 'res_phi', 'unc_phi', 'sig_phi', 'loss']

    for m_idx, m_mode in enumerate([3, 4]):
        phi_unit = np.pi / (2 * m_mode)
        period = 2 * np.pi / m_mode

        for ai, a_m in enumerate(a_m_groups):
            dg = df[df['a_m_group'] == a_m].copy()

            dg['res_am'] = dg[f'a_m{m_mode}_best'] - dg[f'a_m{m_mode}_true']
            dg['unc_am'] = dg[f'a_m{m_mode}_err']
            dg['sig_am'] = dg['res_am'] / dg['unc_am']

            raw_diff = dg[f'phi_m{m_mode}_best'] - dg[f'phi_m{m_mode}_true']
            dg['res_phi'] = (raw_diff + period / 2) % period - period / 2
            dg['unc_phi'] = dg[f'phi_m{m_mode}_err']
            dg['sig_phi'] = dg['res_phi'] / dg['unc_phi']
            dg['loss'] = dg['loss_non_pso']

            cell_data = {mn: np.full((n_R, n_n), np.nan) for mn in metric_names}
            for _, row in dg.iterrows():
                ri = R_grid.index(row['R_sersic_true'])
                ni = n_grid.index(row['n_sersic_true'])
                for mn in metric_names:
                    cell_data[mn][ri, ni] = row[mn]

            raw_per_cell[(m_idx, ai)] = cell_data
            for mi, mn in enumerate(metric_names):
                precomp[(m_idx, ai, mi)] = cell_data[mn]

    # ------------------------------------------------------------------
    # Build Tkinter window
    # ------------------------------------------------------------------
    root = tk.Tk()
    root.title("Multipole Grid Validation – Threshold Explorer")

    ctrl_frame = tk.Frame(root, padx=10, pady=10)
    ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)

    fig_frame = tk.Frame(root)
    fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    slider_defs = [
        ('a_m_factor1', 'a_m residual factor',     0, 2.0, 0.05, 1.0,  'red'),
        ('a_m_factor2', 'a_m uncertainty factor',   0, 2.0, 0.05, 1.0,  'orange'),
        ('a_m_factor3', 'a_m σ factor',             0, 4.0, 0.1,  2.0,  'gold'),
        ('phi_m_factor1', 'ϕ_m residual factor',    0, 1.0, 0.05, 0.5,  'green'),
        ('phi_m_factor2', 'ϕ_m uncertainty factor',  0, 2.0, 0.05, 1.0, 'dodgerblue'),
        ('phi_m_factor3', 'ϕ_m σ factor',           0, 4.0, 0.1,  2.0,  'purple'),
    ]

    sliders = {}
    for key, label, lo, hi, res, default, color in slider_defs:
        frm = tk.Frame(ctrl_frame)
        frm.pack(fill=tk.X)
        canvas_dot = tk.Canvas(frm, width=14, height=14, highlightthickness=0)
        canvas_dot.create_rectangle(1, 1, 13, 13, fill=color, outline='black')
        canvas_dot.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(frm, text=label, anchor='w').pack(side=tk.LEFT)

        s = tk.Scale(ctrl_frame, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, length=220)
        s.set(default)
        s.pack(fill=tk.X, pady=(0, 4))
        sliders[key] = s

    summary_var = tk.StringVar(value="")
    tk.Label(ctrl_frame, textvariable=summary_var, justify=tk.LEFT,
             anchor='nw', wraplength=240, font=('Courier', 9)).pack(
                 fill=tk.X, pady=(10, 0))

    # ------------------------------------------------------------------
    # Layout:
    # m=3 (0-6), Spacer (7), m=4 (8-14), Spacer (15), Loss (16), Total Sum (17)
    # Total = 18 rows
    # ------------------------------------------------------------------
    n_cols = len(a_m_groups)
    n_metric_rows = 6
    total_rows = 18

    # Height ratios: metric rows 1, summary rows 1.2, spacer rows 0.3
    height_ratios = [1]*6 + [1.2] + [0.3] + [1]*6 + [1.2] + [0.3] + [1.2]*2

    fig, all_axes = plt.subplots(
        total_rows, n_cols + 1,
        figsize=(3.6 * n_cols + 1.0, 2.7 * total_rows),
        gridspec_kw={
            'width_ratios': [1] * n_cols + [0.05],
            'height_ratios': height_ratios,
        }
    )
    # Increased hspace from 0.4 to 0.7 to avoid title/tick overlaps
    fig.subplots_adjust(hspace=0.7, wspace=0.45, top=0.90, bottom=0.02,
                        left=0.07, right=0.96)

    header_artists = []
    x_markers = []

    spec_meta = [
        (-0.01, 0.01, 'bwr',    'a_m{m} residual (est−true)'),
        (0,     0.01, 'Greens', 'a_m{m} unc (1σ)'),
        (-2,    2,    'bwr',    'a_m{m} σ from truth'),
        (None,  None, 'bwr',    'ϕ_m{m} residual (est−true)'),
        (None,  None, 'Greens', 'ϕ_m{m} unc (1σ)'),
        (-2,    2,    'bwr',    'ϕ_m{m} σ from truth'),
        (0,     1.2,  'Blues',  'Fitting Loss (loss_non_pso)'),
    ]

    # Draw static heatmaps
    for m_idx, m_mode in enumerate([3, 4]):
        row_off = m_idx * 8  # 0 or 8
        phi_unit = np.pi / (2 * m_mode)

        for ai in range(n_cols):
            for mi in range(n_metric_rows):
                ax = all_axes[row_off + mi, ai]
                arr = precomp[(m_idx, ai, mi)]

                vmin, vmax, cmap, ttpl = spec_meta[mi]
                if mi == 3:
                    vmin, vmax = -phi_unit, phi_unit
                elif mi == 4:
                    vmin, vmax = 0, phi_unit

                ax.imshow(arr, aspect='auto', origin='upper',
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation='nearest')

                for ri in range(n_R):
                    for ni in range(n_n):
                        v = arr[ri, ni]
                        if np.isfinite(v):
                            txt = f'{v:.3f}'
                            ax.text(ni, ri, txt, ha='center', va='center',
                                    fontsize=6, color='black')

                ax.set_xticks(range(n_n))
                ax.set_xticklabels([f'{v:.1f}' for v in n_grid], fontsize=6)
                ax.set_yticks(range(n_R))
                ax.set_yticklabels([f'{v:.1f}' for v in R_grid], fontsize=6)

                ax.set_title(ttpl.format(m=m_mode), fontsize=7)
                if mi == 0:
                    ax.text(0.5, 1.3, f'a_m = {a_m_groups[ai]}', 
                            transform=ax.transAxes, ha='center', va='bottom',
                            fontsize=9, fontweight='bold')
                
                ax.set_ylabel('R_sersic' if ai == 0 else '', fontsize=8, fontweight='bold')

            # Combined summary row (6 and 14)
            summary_row = row_off + n_metric_rows
            ax_sum = all_axes[summary_row, ai]
            ax_sum.set_xlim(-0.5, n_n - 0.5)
            ax_sum.set_ylim(-0.5, n_R - 0.5)
            ax_sum.invert_yaxis()
            ax_sum.set_xticks(range(n_n))
            ax_sum.set_xticklabels([f'{v:.1f}' for v in n_grid], fontsize=6)
            ax_sum.set_yticks(range(n_R))
            ax_sum.set_yticklabels([f'{v:.1f}' for v in R_grid], fontsize=6)
            ax_sum.set_facecolor('#f0f0f0')

            # Draw cell boundaries
            for edge in [-0.5, 0.5, 1.5, 2.5, 3.5]:
                ax_sum.axvline(edge, color='black', linewidth=1)
                ax_sum.axhline(edge, color='black', linewidth=1)

            ax_sum.set_title(f'm={m_mode} Combined Threshold Map', fontsize=8, fontweight='normal')
            if ai == 0:
                ax_sum.set_ylabel('R_sersic', fontsize=8, fontweight='bold')
            ax_sum.set_xlabel('n_sersic', fontsize=8, fontweight='bold', labelpad=0)

        # Colorbars for metrics
        for mi in range(n_metric_rows):
            cax = all_axes[row_off + mi, -1]
            vmin, vmax, cmap, _ = spec_meta[mi]
            if mi == 3:
                vmin, vmax = -phi_unit, phi_unit
            elif mi == 4:
                vmin, vmax = 0, phi_unit

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax)
            if mi in [3, 4]:
                cb.ax.yaxis.set_major_formatter(_pi_frac_formatter(m_mode))
            cb.ax.tick_params(labelsize=6)
        all_axes[row_off + n_metric_rows, -1].set_visible(False)

    # Row 16: Loss heatmap (average over m=3/m=4 for display, or just m=3)
    loss_row = 16
    for ai in range(n_cols):
        ax = all_axes[loss_row, ai]
        arr = precomp[(0, ai, 6)] # loss
        vmin, vmax, cmap, ttpl = spec_meta[6]
        ax.imshow(arr, aspect='auto', origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
        for ri in range(n_R):
            for ni in range(n_n):
                v = arr[ri, ni]
                if np.isfinite(v):
                    ax.text(ni, ri, f'{v:.3f}', ha='center', va='center', fontsize=6, color='black')
        ax.set_xticks(range(n_n))
        ax.set_xticklabels([f'{v:.1f}' for v in n_grid], fontsize=6)
        ax.set_yticks(range(n_R))
        ax.set_yticklabels([f'{v:.1f}' for v in R_grid], fontsize=6)
        ax.set_title(ttpl, fontsize=8, fontweight='bold')
        if ai == 0:
            ax.set_ylabel('R_sersic', fontsize=8, fontweight='bold')
        ax.set_xlabel('n_sersic', fontsize=8, fontweight='bold', labelpad=0)

    # Colorbar for Loss
    cax_loss = all_axes[loss_row, -1]
    norm_l = mcolors.Normalize(vmin=0, vmax=1.2)
    sm_l = plt.cm.ScalarMappable(cmap='Blues', norm=norm_l)
    sm_l.set_array([])
    fig.colorbar(sm_l, cax=cax_loss).ax.tick_params(labelsize=6)

    # Row 17: Total Combined Threshold Map (m=3 & m=4)
    total_sum_row = 17
    for ai in range(n_cols):
        ax = all_axes[total_sum_row, ai]
        ax.set_xlim(-0.5, n_n - 0.5)
        ax.set_ylim(-0.5, n_R - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(n_n))
        ax.set_xticklabels([f'{v:.1f}' for v in n_grid], fontsize=6)
        ax.set_yticks(range(n_R))
        ax.set_yticklabels([f'{v:.1f}' for v in R_grid], fontsize=6)
        ax.set_facecolor('#f0f0f0')
        for edge in [-0.5, 0.5, 1.5, 2.5, 3.5]:
            ax.axvline(edge, color='black', linewidth=1)
            ax.axhline(edge, color='black', linewidth=1)
        ax.set_title('Combined Threshold Map (m=3 & m=4)', fontsize=8, fontweight='bold')
        if ai == 0:
            ax.set_ylabel('R_sersic', fontsize=8, fontweight='bold')
        ax.set_xlabel('n_sersic', fontsize=8, fontweight='bold', labelpad=0)
    all_axes[total_sum_row, -1].set_visible(False)

    # Spacers and Separators
    for row in [7, 15]:
        for ax in all_axes[row, :]:
            ax.set_visible(False)
        bbox = all_axes[row, 0].get_position()
        sep_y = bbox.y0 + bbox.height / 2
        line = plt.Line2D([0.04, 0.98], [sep_y, sep_y], transform=fig.transFigure,
                          color='black', linewidth=2.5)
        fig.add_artist(line)

    # Section labels
    for m_idx, label in enumerate(['m=3', 'm=4', 'Global']):
        row_off = [0, 8, 16][m_idx]
        y_center_fig = 0.02 + (0.90 - 0.02) * (1 - (row_off + 3.5)/total_rows)
        if m_idx == 2: y_center_fig = 0.02 + (0.90 - 0.02) * (1 - (17)/total_rows)
        fig.text(0.01, y_center_fig, label, fontsize=14, fontweight='bold', va='center', rotation=90)

    canvas = FigureCanvasTkAgg(fig, master=fig_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Reevaluate
    # ------------------------------------------------------------------
    def reevaluate():
        for mk in x_markers: mk.remove()
        x_markers.clear()
        for art in header_artists: art.remove()
        header_artists.clear()

        f1, f2, f3 = sliders['a_m_factor1'].get(), sliders['a_m_factor2'].get(), sliders['a_m_factor3'].get()
        pf1, pf2, pf3 = sliders['phi_m_factor1'].get(), sliders['phi_m_factor2'].get(), sliders['phi_m_factor3'].get()

        total_filtered, total_cells = 0, 0
        all_bads = {} # store bad masks for global summary

        for m_idx, m_mode in enumerate([3, 4]):
            row_off = m_idx * 8
            phi_unit = np.pi / (2 * m_mode)
            marker = '2' if m_mode == 3 else 'x'
            summary_row = row_off + n_metric_rows

            for ai in range(n_cols):
                cd = raw_per_cell[(m_idx, ai)]
                valid = np.isfinite(cd['res_am'])

                bads = [np.abs(cd['res_am']) > f1*0.01, cd['unc_am'] > f2*0.01, np.abs(cd['sig_am']) > f3,
                        np.abs(cd['res_phi']) > pf1*phi_unit, cd['unc_phi'] > pf2*phi_unit, np.abs(cd['sig_phi']) > pf3]
                all_bads[(m_idx, ai)] = bads

                any_bad = bads[0].copy()
                for b in bads[1:]: any_bad |= b
                total_cells += valid.sum()
                total_filtered += (any_bad & valid).sum()

                # Draw on metrics and m-specific summary
                for ci, (_, color, dx, _) in enumerate(CRITERION_STYLES):
                    # Metric axis
                    ax = all_axes[row_off + ci, ai]
                    # Summary axis
                    ax_s = all_axes[summary_row, ai]
                    for ri in range(n_R):
                        for ni in range(n_n):
                            if bads[ci][ri, ni] and valid[ri, ni]:
                                for target_ax, msize, mwidth in [(ax, 10, 1), (ax_s, 12, 1.5)]:
                                    ln, = target_ax.plot(ni+dx, ri, marker, color=color,
                                                       markeredgecolor=color, markersize=msize,
                                                       markeredgewidth=mwidth, zorder=10)
                                    x_markers.append(ln)

        # Global Combined summary (row 17)
        for ai in range(n_cols):
            ax_g = all_axes[17, ai]
            valid_g = np.isfinite(raw_per_cell[(0, ai)]['res_am'])
            for m_idx, m_mode in enumerate([3, 4]):
                marker = '2' if m_mode == 3 else 'x'
                bads = all_bads[(m_idx, ai)]
                for ci, (_, color, dx, _) in enumerate(CRITERION_STYLES):
                    for ri in range(n_R):
                        for ni in range(n_n):
                            if bads[ci][ri, ni] and valid_g[ri, ni]:
                                ln, = ax_g.plot(ni+dx, ri, marker, color=color,
                                              markeredgecolor=color, markersize=12,
                                              markeredgewidth=1.5, zorder=10)
                                x_markers.append(ln)

        summary_var.set(f"Filtered: {total_filtered}/{total_cells} ({100*total_filtered/max(total_cells,1):.1f}%)")

        # Header Info
        vals = [f"{f:.2f}×0.01" for f in [f1, f2]] + [f"{f3:.1f}"] + \
               [f"{f:.2f}×π/2m" for f in [pf1, pf2]] + [f"{pf3:.1f}"]
        for i, (_, color, _, label) in enumerate(CRITERION_STYLES):
            x = 0.12 + i * 0.14
            rect = plt.Rectangle((x-0.008, 0.957), 0.01, 0.015, transform=fig.transFigure, color=color)
            fig.add_artist(rect); header_artists.append(rect)
            txt = fig.text(x+0.005, 0.965, f"{label} > {vals[i]}", fontsize=9, fontweight='bold', va='center')
            header_artists.append(txt)
        t_stats = fig.text(0.5, 0.940, f"FILTERED: {total_filtered}/{total_cells} ({100*total_filtered/max(total_cells,1):.1f}%)",
                           fontsize=12, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='white', boxstyle='round,pad=0.3'))
        header_artists.append(t_stats)
        canvas.draw_idle()

    tk.Button(ctrl_frame, text="Reevaluate", command=reevaluate, bg='#2196F3', fg='gray', font=('Helvetica', 11, 'bold')).pack(fill=tk.X, pady=(15, 0))

    def save_pdf():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = f'../data/multipole_grid_validation_interactive_{stamp}'
        fig.savefig(base + '.pdf', bbox_inches='tight')
        with open(base + '.txt', 'w') as f:
            for key in sliders: f.write(f"{key}: {sliders[key].get()}\n")
    tk.Button(ctrl_frame, text="Save as PDF", command=save_pdf, bg='#4CAF50', fg='gray', font=('Helvetica', 11, 'bold')).pack(fill=tk.X, pady=(8, 0))

    legend_frame = tk.LabelFrame(ctrl_frame, text="Marker Legend", padx=5, pady=5)
    legend_frame.pack(fill=tk.X, pady=(15, 0))
    for _key, color, _dx, label in CRITERION_STYLES:
        frm = tk.Frame(legend_frame); frm.pack(fill=tk.X, pady=1)
        c = tk.Canvas(frm, width=14, height=14, highlightthickness=0)
        c.create_text(7, 7, text='✕', fill=color, font=('Helvetica', 10, 'bold'))
        c.pack(side=tk.LEFT, padx=(0, 4))
        tk.Label(frm, text=label, font=('Helvetica', 9)).pack(side=tk.LEFT)

    reevaluate()
    root.mainloop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gui', action='store_true',
                        help='Launch interactive Tkinter GUI')
    args = parser.parse_args()

    dfs = collect_all_dataframes()

    if args.gui:
        launch_gui(dfs)
    else:
        plot_grid(dfs)
