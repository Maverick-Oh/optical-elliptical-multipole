#!/usr/bin/env python
"""
compare_circ_vs_ellip.py
========================
Compare circular vs elliptical multipole fitting on synthetic data.

Generates an elliptical-multipole mock, then fits it twice:
  1. Elliptical multipole model  (recovers phi_m / a_m)
  2. Circular  multipole model  (recovers theta_m / a_m)

Produces before/after comparison plots with contour overlays and a
summary figure of true vs measured a_m for a sweep of true amplitudes.

Usage examples
--------------
# Single a_m4=0.1 (defaults):
python compare_circ_vs_ellip.py

# Sweep over several a_m4 values:
python compare_circ_vs_ellip.py --a-m-list 0.001 0.003 0.01 0.03 0.1

# Override parameters via JSON:
python compare_circ_vs_ellip.py --config my_config.json

# Force re-fit (ignore cached CSV):
python compare_circ_vs_ellip.py --overwrite
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

from tools_fitting import (
    simulate_model_elliptical_multipole,
    build_arcsec_grid,
    pack_params,
    unpack_params,
    process_one_target_optimize,
    residual_map_sigma,
    reduced_chi_squared,
    downsample,
)
from optical_elliptical_multipole.nonjax.profiles1D import (
    Elliptical_Multipole_Profile_1D,
    Circular_Multipole_Profile_1D,
)
from optical_elliptical_multipole.plotting.plot_tools import detailed_comparison_plot

# ──────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

DEFAULT_CFG = dict(
    # Multipole
    m=[4],
    a_m=[0.1],          # one per m
    phi_m=[0.0],        # one per m (radians)
    # Ellipse
    q=0.8,
    theta_ell=0.0,
    # Sérsic
    n_sersic=2.0,
    R_sersic=1.0,       # arcsec
    amplitude=1.0,
    # Position / background
    x0=0.0,
    y0=0.0,
    background=0.001,
    # Observation settings
    PIX_SCALE=0.03,          # arcsec/pixel
    SUPERSAMPLE_FACTOR=10,   # for mock generation
    FIT_SUPERSAMPLE_FACTOR=1,
    EXPTIME=4056.0,
    RMS_NOISE=0.005,
    # Grid
    grid_half_extent=2.01,   # arcsec
)

CSV_FILENAME = "comparison_results.csv"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_config(json_path=None):
    """Return config dict: defaults merged with JSON overrides."""
    cfg = DEFAULT_CFG.copy()
    if json_path is not None:
        with open(json_path) as f:
            overrides = json.load(f)
        cfg.update(overrides)
    # Ensure arrays
    cfg['m'] = list(cfg['m'])
    cfg['a_m'] = list(cfg['a_m'])
    cfg['phi_m'] = list(cfg['phi_m'])
    return cfg


def generate_mock(cfg, a_m_override=None, seed=42):
    """
    Generate a noisy mock observation from an elliptical-multipole Sérsic.

    Returns
    -------
    sci, wht : 2-d arrays (counts/sec scale)
    extent : imshow extent in arcsec
    truth_params : dict of ground-truth parameters
    """
    rng = np.random.RandomState(seed)

    a_m = np.array(cfg['a_m'] if a_m_override is None else a_m_override, dtype=float)
    phi_m = np.array(cfg['phi_m'], dtype=float)
    m = np.array(cfg['m'], dtype=int)

    pix = cfg['PIX_SCALE']
    half = cfg['grid_half_extent']
    nx = int(np.ceil(2 * half / pix))
    if nx % 2 == 0:
        nx += 1
    ny = nx

    ss = cfg['SUPERSAMPLE_FACTOR']

    # Fine grid
    X_f, Y_f, ext_f = build_arcsec_grid((ny * ss, nx * ss), pixscale=pix / ss)

    I_fine = simulate_model_elliptical_multipole(
        X_f, Y_f,
        n_sersic=cfg['n_sersic'], R_sersic=cfg['R_sersic'], amplitude=cfg['amplitude'],
        q=cfg['q'], theta_ell=cfg['theta_ell'],
        m=m, a_m=a_m, phi_m=phi_m,
        x0=cfg['x0'], y0=cfg['y0'],
        background=cfg.get('background', 0.001),
    )

    # Downsample
    I_coarse = I_fine.reshape(ny, ss, nx, ss).mean(axis=(1, 3))

    # Add noise (Poisson + Gaussian)
    exptime = cfg['EXPTIME']
    rms = cfg['RMS_NOISE']
    wht_val = 1.0 / rms**2

    counts_true = I_coarse * exptime
    counts_noisy = rng.poisson(np.abs(counts_true)).astype(float)
    I_poisson = counts_noisy / exptime
    bkg_noise = rng.normal(0, rms, size=I_coarse.shape)
    sci = I_poisson + bkg_noise
    wht = np.full_like(sci, wht_val)

    _, _, extent = build_arcsec_grid((ny, nx), pixscale=pix)

    truth = dict(
        n_sersic=cfg['n_sersic'], R_sersic=cfg['R_sersic'], amplitude=cfg['amplitude'],
        q=cfg['q'], theta_ell=cfg['theta_ell'],
        x0=cfg['x0'], y0=cfg['y0'], background=cfg.get('background', 0.001),
        EXPTIME_SCI=exptime,
    )
    for i, mi in enumerate(m):
        truth[f'a_m{mi}'] = a_m[i]
        truth[f'phi_m{mi}'] = phi_m[i]

    return sci, wht, extent, truth


def build_fixed_params_for_multipole_only(truth, cfg):
    """
    Build a fixed_params dict that freezes everything EXCEPT
    the multipole amplitudes (a_m) and angles (phi_m).
    """
    fixed = dict(
        n_sersic=truth['n_sersic'],
        R_sersic=truth['R_sersic'],
        amplitude=truth['amplitude'],
        q=truth['q'],
        theta_ell=truth['theta_ell'],
        x0=truth['x0'],
        y0=truth['y0'],
        background=truth['background'],
    )
    return fixed


def build_row_query(truth, seq_id):
    """Build a fake row_query Series for process_one_target_optimize."""
    return pd.Series(dict(
        sequentialid=seq_id,
        EXPTIME_SCI=truth['EXPTIME_SCI'],
        EXPTIME_WHT=0,
        sersic_n_gim2d=truth['n_sersic'],
        r50=truth['R_sersic'],
    ))


def build_row_sep(sci, truth, seq_id, pix_scale):
    ny, nx = sci.shape
    return pd.Series(dict(
        seqid=seq_id,
        image_width=nx,
        image_height=ny,
        q=truth['q'],
        theta=truth['theta_ell'],
        x=0, y=0,
        xcpeak=nx / 2 - 0.5,
        ycpeak=ny / 2 - 0.5,
        R50=truth['R_sersic'] / pix_scale,   # SEP R50 is in pixel units
        flux=truth['amplitude'] * 100,
    ))


def fit_and_plot(sci, wht, extent, truth, cfg, model_type, out_dir, label, seq_id=0,
                 fit_all_params=False, wrong_start=False):
    """
    Fit with the given model_type and produce comparison plots.

    Returns
    -------
    rec : dict  — fitting results (including a_m and uncertainties)
    """
    m = cfg['m']
    k = len(m)

    # Fixed params
    if fit_all_params:
        fixed_params = None
    else:
        fixed_params = build_fixed_params_for_multipole_only(truth, cfg)

    # Prepare initial guess override (if wrong_start)
    initial_guess = None
    if wrong_start:
        initial_guess = dict(
            n_sersic=truth['n_sersic'] * 1.3,
            R_sersic=truth['R_sersic'] * 0.7,
            amplitude=truth['amplitude'] * 1.5,
            q=min(truth['q'] + 0.1, 0.99),
            theta_ell=truth['theta_ell'] + 0.1,
            x0=0.05, y0=-0.05,
            background=0.0,
            a_m=np.zeros(k),
            phi_m=np.zeros(k),
        )
        if not fit_all_params:
            warnings.warn("--wrong-start given without --fit-all-params: enabling full-param fitting")
            fixed_params = None

    row_query = build_row_query(truth, seq_id)
    row_sep = build_row_sep(sci, truth, seq_id, pix_scale=cfg['PIX_SCALE'])

    mask = np.zeros(sci.shape, dtype=bool)
    segmap = np.zeros(sci.shape, dtype=int)
    psf = np.ones((5, 5))  # Placeholder

    rec = process_one_target_optimize(
        row_query=row_query,
        data_dir=out_dir,
        row_sep=row_sep,
        sci=sci, wht=wht, mask=mask, segmap=segmap, psf=psf,
        m=m,
        opt_method='L-BFGS-B',
        PIX_SCALE=cfg['PIX_SCALE'],
        plot_initial_contour=False,
        plot_final_contour=False,  # We'll do our own plotting below
        fit_model=True,
        verbose=True,
        target_loss=1.2,
        supersample_factor=cfg['FIT_SUPERSAMPLE_FACTOR'],
        truth_row=truth,
        plot_name=os.path.join(out_dir, f"{label}"),
        initial_guess=initial_guess,
        model_type=model_type,
        fixed_params=fixed_params,
        use_jax=False,  # numpy for all model types
    )

    # ── Build after-fitting comparison plot with contours ──
    ss = cfg['FIT_SUPERSAMPLE_FACTOR']
    ny, nx = sci.shape
    X_ss, Y_ss, _ = build_arcsec_grid((ny * ss, nx * ss), pixscale=cfg['PIX_SCALE'] / ss)

    # Reconstruct best model
    from tools_fitting import _build_model_image
    p_best = unpack_params(
        pack_params(dict(
            n_sersic=rec.get('n_sersic_best', truth['n_sersic']),
            R_sersic=rec.get('R_sersic_best', truth['R_sersic']),
            amplitude=rec.get('amplitude_best', truth['amplitude']),
            q=rec.get('q_best', truth['q']),
            theta_ell=rec.get('theta_ell_best', truth['theta_ell']),
            a_m=np.array([rec.get(f'a_m{mi}_best', 0.) for mi in m]),
            phi_m=np.array([rec.get(f'phi_m{mi}_best', 0.) for mi in m]),
            x0=rec.get('x0_best', 0.),
            y0=rec.get('y0_best', 0.),
            background=rec.get('background_best', 0.),
        ), k),
        k,
    )
    mod_ss = _build_model_image(model_type, X_ss, Y_ss, p_best, np.array(m))
    if ss > 1:
        mod_final = downsample(mod_ss, ss)
    else:
        mod_final = mod_ss
    mod_final = mod_final + p_best['background']

    exptime = truth['EXPTIME_SCI']
    n_param = 5 + 3 * k
    res_map = residual_map_sigma(sci, wht, mod_final, exptime, mask=mask)

    # Param dicts for plot
    p_best_flat = {}
    p_unc_flat = {}
    for key in ['n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell', 'x0', 'y0', 'background']:
        p_best_flat[key] = p_best[key]
        p_unc_flat[key] = rec.get(f"{key}_err", np.nan)
    for i, mi in enumerate(m):
        p_best_flat[f"a_m{mi}"] = p_best['a_m'][i]
        p_unc_flat[f"a_m{mi}"] = rec.get(f"a_m{mi}_err", np.nan)
        if model_type == 'circular_multipole':
            p_best_flat[f"theta_m{mi}"] = p_best['phi_m'][i]
            p_unc_flat[f"theta_m{mi}"] = rec.get(f"phi_m{mi}_err", np.nan)
        else:
            p_best_flat[f"phi_m{mi}"] = p_best['phi_m'][i]
            p_unc_flat[f"phi_m{mi}"] = rec.get(f"phi_m{mi}_err", np.nan)

    p_true_flat = {k.replace('_true', ''): v for k, v in truth.items()
                   if k not in ('EXPTIME_SCI',)}

    chi2_init = rec.get('loss_initial', np.nan)
    chi2_final = rec.get('loss_final', np.nan)
    meta = (f"Model: {model_type}\n"
            f"Loss Init: {chi2_init:.3f}\n"
            f"Loss Final: {chi2_final:.3f}\n"
            f"SS Factor: {ss}")

    fig, axs = detailed_comparison_plot(
        np.ma.masked_array(sci, mask=mask), mod_final, res_map,
        extent=extent,
        param_best=p_best_flat, param_unc=p_unc_flat, param_true=p_true_flat,
        meta_info_str=meta,
        scale='asinh',
    )

    # ── Overlay contours ──
    R_s = truth['R_sersic']
    q_t = truth['q']
    th_t = truth['theta_ell']
    x0_t = truth['x0']
    y0_t = truth['y0']

    # TRUE contour on Observed panel (axs[0,0])
    a_m_true = np.array([truth[f'a_m{mi}'] for mi in m])
    phi_m_true = np.array([truth[f'phi_m{mi}'] for mi in m])
    x_true, y_true = Elliptical_Multipole_Profile_1D(
        r0=R_s, q=q_t, theta_ell=th_t, m=np.array(m), a_m=a_m_true,
        phi_m=phi_m_true, x0=x0_t, y0=y0_t, return_type='xy',
        include_end=True, n_points=300,
    )
    axs[0, 0].plot(x_true, y_true, color='lime', lw=1.2, label='True contour')

    # FITTED contour on Model panel (axs[0,1])
    a_m_fit = p_best['a_m']
    phi_m_fit = p_best['phi_m']
    R_s_fit = p_best['R_sersic']
    q_fit = p_best['q']
    th_fit = p_best['theta_ell']
    x0_fit = p_best['x0']
    y0_fit = p_best['y0']

    if model_type in ('elliptical_multipole', 'elliptical'):
        x_fit, y_fit = Elliptical_Multipole_Profile_1D(
            r0=R_s_fit, q=q_fit, theta_ell=th_fit, m=np.array(m), a_m=a_m_fit,
            phi_m=phi_m_fit, x0=x0_fit, y0=y0_fit, return_type='xy',
            include_end=True, n_points=300,
        )
    else:  # circular_multipole
        x_fit, y_fit = Circular_Multipole_Profile_1D(
            r0=R_s_fit, q=q_fit, theta_ell=th_fit, m=np.array(m), a_m=a_m_fit,
            theta_m=phi_m_fit, x0=x0_fit, y0=y0_fit, return_type='xy',
            include_end=True, n_points=300,
        )
    axs[0, 1].plot(x_fit, y_fit, color='k', lw=1.0, label='Fitted contour')

    out_pdf = os.path.join(out_dir, f"{seq_id}-05-after_fitting_{model_type}.pdf")
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf}")

    return rec


def fit_slope(true_vals, meas_vals):
    """
    Least-squares fit of y = slope * x (no intercept).
    Returns slope.
    """
    # slope = sum(x*y) / sum(x^2)
    return np.sum(true_vals * meas_vals) / np.sum(true_vals ** 2)


def make_summary_figure(df_results, m_list, out_dir, q_value=None):
    """
    Plot true a_m vs measured a_m for elliptical (blue) and circular (red).
    One subplot per multipole order. Fits y = slope * x and annotates.

    Returns
    -------
    slopes : dict  — {model_type: {f'slope_m{mi}': float}}
    """
    n_m = len(m_list)
    fig, axes = plt.subplots(1, n_m, figsize=(6 * n_m, 5), squeeze=False)
    axes = axes[0]

    slopes = {}  # {model_type: {f'slope_m{mi}': val}}

    for ax_idx, mi in enumerate(m_list):
        ax = axes[ax_idx]

        for model_type, color, marker in [
            ('elliptical_multipole', 'tab:blue', 'o'),
            ('circular_multipole', 'tab:red', 's'),
        ]:
            sub = df_results[df_results['model_type'] == model_type]
            if sub.empty:
                continue

            true_vals = sub[f'a_m{mi}_true'].values
            meas_vals = sub[f'a_m{mi}_best'].values
            err_vals = sub[f'a_m{mi}_err'].values

            ax.errorbar(
                true_vals, meas_vals, yerr=err_vals,
                fmt=marker, color=color, capsize=3,
                label=model_type.replace('_', ' '),
                markersize=6,
            )

            # Linear fit y = slope * x
            if len(true_vals) >= 2:
                slope = fit_slope(true_vals, meas_vals)
            else:
                slope = meas_vals[0] / true_vals[0] if true_vals[0] != 0 else np.nan

            if model_type not in slopes:
                slopes[model_type] = {}
            slopes[model_type][f'slope_m{mi}'] = slope

            # Draw best-fit line
            x_line = np.linspace(min(true_vals)*1.2, max(true_vals) * 1.2, 100) if min(true_vals) < 0 else np.linspace(0, max(true_vals) * 1.2, 100)
            ax.plot(x_line, slope * x_line, color=color, ls='--', lw=1.0, alpha=0.7)

            # Annotate slope
            label_short = 'Ellip' if 'elliptical' in model_type else 'Circ'
            ax.text(
                0.05 if 'elliptical' in model_type else 0.05,
                0.92 if 'elliptical' in model_type else 0.84,
                f'{label_short} slope = {slope:.4f}',
                transform=ax.transAxes, fontsize=9, color=color,
                fontweight='bold',
            )

        # 1:1 line
        all_true = df_results[f'a_m{mi}_true'].values
        lims = [min(all_true) * 1.2, max(all_true) * 1.2] if min(all_true) < 0 else [0, max(all_true) * 1.2]
        ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='1:1')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f'True $a_{{m={mi}}}$')
        ax.set_ylabel(f'Measured $a_{{m={mi}}}$')
        title = f'Multipole m={mi}'
        if q_value is not None:
            title += f'  (q={q_value:.2f})'
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.grid(True)

    fig.tight_layout()
    out_fig = os.path.join(out_dir, "summary_circ_vs_ellip.pdf")
    fig.savefig(out_fig, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved summary figure: {out_fig}")
    return slopes


def make_slope_vs_q_figure(slope_records, m_list, out_dir):
    """
    Plot the fitted slope as a function of q for each model type.

    Parameters
    ----------
    slope_records : list of dicts with keys 'q', 'model_type', 'slope_m{mi}'
    """
    df = pd.DataFrame(slope_records)
    n_m = len(m_list)
    fig, axes = plt.subplots(1, n_m, figsize=(6 * n_m, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, mi in enumerate(m_list):
        ax = axes[ax_idx]
        col = f'slope_m{mi}'

        for model_type, color, marker in [
            ('elliptical_multipole', 'tab:blue', 'o'),
            ('circular_multipole', 'tab:red', 's'),
        ]:
            sub = df[df['model_type'] == model_type].sort_values('q')
            if sub.empty:
                continue

            ax.plot(sub['q'], sub[col], marker=marker, color=color, lw=1.5,
                    label=model_type.replace('_', ' '), markersize=7)

        ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5, label='slope = 1')
        ax.set_xlabel('$q$ (axis ratio)', fontsize=12)
        ax.set_ylabel(f'Slope ($a_{{m={mi},\\mathrm{{meas}}}} / a_{{m={mi},\\mathrm{{true}}}}$)', fontsize=12)
        ax.set_title(f'Slope vs q — m={mi}')
        ax.legend(fontsize=9)
        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(top=1.1)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_fig = os.path.join(out_dir, "slope_vs_q.pdf")
    fig.savefig(out_fig, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved slope vs q figure: {out_fig}")

    # Also save slope data as CSV
    csv_out = os.path.join(out_dir, "slope_vs_q.csv")
    df.to_csv(csv_out, index=False)
    print(f"Saved slope vs q data: {csv_out}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare circular vs elliptical multipole fitting")
    parser.add_argument("--config", type=str, default=None, help="JSON config file to override defaults")
    parser.add_argument("--a-m-list", type=float, nargs='+', default=None,
                        help="Sweep over these true a_m values (applied to all m)")
    parser.add_argument("--q-list", type=float, nargs='+', default=None,
                        help="Sweep over these q values (produces slope-vs-q figure)")
    parser.add_argument("--fit-all-params", action='store_true',
                        help="Fit all parameters (not just multipole)")
    parser.add_argument("--wrong-start", action='store_true',
                        help="Start from perturbed initial guess (implies --fit-all-params)")
    parser.add_argument("--supersample", type=int, default=None,
                        help="Fitting supersampling factor (overrides config)")
    parser.add_argument("--overwrite", action='store_true',
                        help="Recompute even if CSV results exist")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "comparison_circ_ellip_fitting"),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mock generation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.supersample is not None:
        cfg['FIT_SUPERSAMPLE_FACTOR'] = args.supersample

    out_dir_base = args.out_dir
    os.makedirs(out_dir_base, exist_ok=True)

    # Determine a_m values to sweep
    if args.a_m_list is not None:
        a_m_sweep = args.a_m_list
    else:
        a_m_sweep = [cfg['a_m'][0]]  # single default value

    m_list = cfg['m']
    k = len(m_list)

    # Determine q values to sweep
    if args.q_list is not None:
        q_sweep = args.q_list
    else:
        q_sweep = [cfg['q']]  # single default value

    # Collect slopes for slope-vs-q analysis
    all_slope_records = []

    for q_val in q_sweep:
        cfg['q'] = q_val
        if len(q_sweep) > 1:
            # Per-q subdirectory
            out_dir = os.path.join(out_dir_base, f"q_{q_val:.2f}")
        else:
            out_dir = out_dir_base
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, CSV_FILENAME)

        print(f"\n{'#'*60}")
        print(f"  q = {q_val}")
        print(f"{'#'*60}")

        # Load existing results (cache)
        if os.path.exists(csv_path) and not args.overwrite:
            df_existing = pd.read_csv(csv_path)
            print(f"Loaded {len(df_existing)} cached results from {csv_path}")
        else:
            df_existing = pd.DataFrame()

        all_records = []
        if not df_existing.empty:
            all_records = df_existing.to_dict('records')

        for idx_am, a_m_val in enumerate(a_m_sweep):
            a_m_array = [a_m_val] * k

            print(f"\n{'='*60}")
            print(f"a_m = {a_m_val} (sweep {idx_am+1}/{len(a_m_sweep)})  q = {q_val}")
            print(f"{'='*60}")

            for model_type in ['elliptical_multipole', 'circular_multipole']:
                label = f"am{a_m_val:.4f}_{model_type}"

                # Skip if cached
                if not df_existing.empty:
                    match = df_existing[
                        (df_existing['model_type'] == model_type) &
                        (np.isclose(df_existing[f'a_m{m_list[0]}_true'], a_m_val, atol=1e-8))
                    ]
                    if not match.empty:
                        print(f"  [{model_type}] Cached result found, skipping fit.")
                        continue

                print(f"\n  Fitting with model_type = {model_type}")
                t0 = time.time()

                sci, wht, extent, truth = generate_mock(cfg, a_m_override=a_m_array, seed=args.seed + idx_am)

                rec = fit_and_plot(
                    sci, wht, extent, truth, cfg,
                    model_type=model_type,
                    out_dir=out_dir,
                    label=label,
                    seq_id=idx_am,
                    fit_all_params=args.fit_all_params,
                    wrong_start=args.wrong_start,
                )
                rec['model_type'] = model_type
                rec['fit_time_total'] = time.time() - t0
                rec['q_true'] = q_val

                for mi in m_list:
                    rec[f'a_m{mi}_true'] = truth[f'a_m{mi}']
                    rec[f'phi_m{mi}_true'] = truth[f'phi_m{mi}']

                all_records.append(rec)

                # Incremental save
                df_save = pd.DataFrame(all_records)
                df_save.to_csv(csv_path, index=False)

        # ── Summary Figure (per q) ──
        df_all = pd.DataFrame(all_records)
        if len(df_all) > 0:
            slopes = make_summary_figure(df_all, m_list, out_dir, q_value=q_val)
            print(f"\nAll results saved to: {csv_path}")

            # Save slopes to CSV within the per-q directory
            for model_type, slope_dict in slopes.items():
                slope_rec = {'q': q_val, 'model_type': model_type}
                slope_rec.update(slope_dict)
                all_slope_records.append(slope_rec)
        else:
            print("No results to plot.")

    # ── Slope vs q Figure (always regenerate from available data) ──
    if all_slope_records:
        make_slope_vs_q_figure(all_slope_records, m_list, out_dir_base)


if __name__ == "__main__":
    main()
