import numpy as np
from matplotlib import pyplot as plt
from astropy.visualization import (ImageNormalize, AsinhStretch, PercentileInterval)

from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from scipy.ndimage import binary_dilation

def _discrete_cmap(n, target_index=1, randomize=False, seed=42):
    """Build a discrete ListedColormap with n+1 entries (0..n).
    - Keeps label indices unchanged.
    - Sets color at target_index to white.
    - If randomize=True, shuffles colors for indices 1..n excluding target_index.
    """
    # choose a base palette
    # if n <= 20 and base_palette in ('tab20', 'tab20b', 'tab20c'):
    #     base = plt.get_cmap(base_palette, n + 1)
    #     colors = base(np.arange(n + 1))
    # else:
    base = plt.get_cmap('hsv', n + 1)
    colors = base(np.arange(n + 1))

    # Optionally randomize all but the target index (and 0 which is unused)
    # if randomize and n >= 2:
    #     rng = np.random.default_rng(seed)
    #     idx = np.arange(1, n + 1)
    #     # exclude the target index from shuffling
    #     idx = idx[idx != target_index]
    #     shuf = idx.copy()
    #     rng.shuffle(shuf)
    #     colors[idx] = colors[shuf]

    # Force the target index to white if it is within range
    # if 1 <= target_index <= n:
    colors[target_index] = np.array([1.0, 1.0, 1.0, 1.0])
    cmap = ListedColormap(colors)
    # plt.figure(); plt.imshow(np.random.randint(n, size=(10,10)), cmap=cmap,); plt.colorbar(); plt.show()
    return cmap

def draw_segmentation(ax, segmap, target_label: int, title: str|None = None, zeros_mask=None,
                      a50=None, a90=None, a99=None, q=None, theta=None, center_xy=None, seed=42, outline=False,
                      randomize_cmap=False, **kwargs):
    # base layer: grayscale of segmap (masked zeros)
    # base = segmap.astype(float)
    base = segmap.astype(float)
    base[segmap == 0] = np.nan
    if zeros_mask is not None:
        base[zeros_mask] = np.nan

    nlab = int(np.nanmax(base)) if np.isfinite(base).any() else 1

    # Build discrete cmap; highlight target_label; do not randomize by default
    tl = int(target_label) if target_label is not None else 1
    cmap = _discrete_cmap(nlab, target_index=tl, randomize=randomize_cmap, seed=seed)
    cmap.set_bad('k')  # zeros/NaNs shown in blue

    im = ax.imshow(base, cmap=cmap, vmin=0, vmax=nlab, **kwargs)

    # Optional outline to enhance boundaries
    if outline and np.isfinite(base).any():
        mask_nonzero = np.isfinite(base)
        edges = binary_dilation(mask_nonzero) ^ mask_nonzero
        ax.imshow(np.where(edges, 1, np.nan), origin='lower', cmap='gray', alpha=0.25)

    # Percent-light ellipses in black for contrast against white target
    if all(v is not None for v in [a50, q, theta, center_xy]):
        cx, cy = center_xy
        for a, lw, lab, color, ls in [(a50, 1.8, 'R50', 'gray', '-'),
                                  (a90, 1.4, 'R90', 'gray', '--'),
                                  (a99, 1.0, 'R99', 'gray', ':')]:
            if a is None or np.isnan(a):
                continue
            b = q * a
            e = Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(theta),
                        fill=False, linewidth=lw, color=color, ls=ls, label=lab)
            ax.add_patch(e)
            # ax.text(cx + 1.2 * a, cy, lab, color=color, fontsize=8, weight='bold')
    if title is not None:
        ax.set_title(title)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.legend()
    return im, cmap

def AsinhStretchPlot(ax, img:np.ma.core.MaskedArray,
                      title: str|None=None, percentile=99.5, norm=None, return_norm=False, a=0.1, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    # normalize with percentile stretch
    interval = PercentileInterval(percentile)
    v = interval.get_limits(img[np.isfinite(img)]) if np.isfinite(img).any() else (-1, 1)
    # AsinhStretch makes things brighter: https://docs.astropy.org/en/stable/api/astropy.visualization.AsinhStretch.html
    if norm is None:
        norm = ImageNormalize(vmin=v[0], vmax=v[1], stretch=AsinhStretch(a=a))
    # with norm argument, AsinhStretch is accounted correctly in plotting and also colorbar scaling later.
    im = ax.imshow(img, norm=norm, **kwargs)#, norm=norm)
    if title is not None:
        ax.set_title(title)
    # ax.set_xticks([]); ax.set_yticks([])
    if return_norm:
        return im, norm
    else:
        return im

def polar_plot(r, theta, fig=None, ax=None, **kwargs):
    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    else:
        if fig is None:
            pass
        if ax is None:
            ax = fig.add_subplot(111)
    ax.plot(theta, r, **kwargs)
    ax.grid(True)
    return fig, ax

def polar_plot_contourf(R, Theta, Z, fig=None, ax=None, title=''):
    if fig is None and ax is None:
        fig, ax = plt.subplots(subplot_kw={})
    else:
        if fig is None:
            pass
        if ax is None:
            ax = fig.add_subplot(111)
    X, Y = R * np.cos(Theta), R * np.sin(Theta)
    ax.contourf(X, Y, Z, cmap=plt.cm.YlGnBu_r, levels=100)

    # Tweak the limits and add latex math labels.
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    # ax.grid(True)
    ax.set_title(title)
    ax.axis('equal')
    return fig, ax

# plot_tools.py
import numpy as np
import matplotlib.pyplot as plt

def _prep_log_scale(img, scale='log', eps=1e-6):
    if scale == 'log':
        # mask non-positive for display; add eps to avoid -inf
        disp = np.log10(np.maximum(img, 0.0) + eps)
    else:
        disp = img
    return disp

def comparison_plot(im1, im2, residual_map=None, *,
                    scale='asinh', labels=None, extent=None, vminvmax_standard='im1',
                    residual_vmin=None, residual_vmax=None, extra_text=None, extra_text_fontsize=10., a=0.1):
    """
    Three-panel comparison plot: data, model, residual.
    Residual uses bwr with symmetric limits so white == 0.
    Axes are equal by default. Data & model share the same color scale.

    scale None, 'log', 'asinh'
    """
    if residual_map is None:
        residual_map = im1 - im2 # default residual: difference
        lim = np.nanmax(np.abs(residual_map))
    else:
        lim = np.nanmax(np.abs(residual_map))
    ncols = 3 if extra_text is None else 4
    fig, axs = plt.subplots(1, ncols, figsize=(4*ncols, 3))
    if (extra_text is not None) and (extra_text != ''):
        axs[-1].text(0.,1.0, extra_text, horizontalalignment='left', verticalalignment='top', fontsize=extra_text_fontsize)
        axs[-1].set_xticks([])
        axs[-1].set_yticks([])
    im1_label = labels[0] if labels is not None else 'image1'
    im2_label = labels[1] if labels is not None else 'image2'
    res_label = labels[2] if (labels is not None and len(labels)>2) else 'image1 - image2'
    im1_lab_d = f'log({im1_label})' if scale == 'log' else im1_label
    im2_lab_d = f'log({im2_label})' if scale == 'log' else im2_label

    im1_disp = _prep_log_scale(im1, scale=scale)
    im2_disp = _prep_log_scale(im2, scale=scale)

    # Shared color limits for data & model
    if vminvmax_standard == 'im1':
        vmin = np.nanmin(im1_disp)
        vmax = np.nanmax(im1_disp)
    elif vminvmax_standard == 'im2':
        vmin = np.nanmin(im2_disp)
        vmax = np.nanmax(im2_disp)
    elif vminvmax_standard == 'both':
        vmin = np.nanmin([np.nanmin(im1_disp), np.nanmin(im2_disp)])
        vmax = np.nanmax([np.nanmax(im1_disp), np.nanmax(im2_disp)])
    else:
        raise ValueError(f"Invalid vminvmax_standard: {vminvmax_standard}; it should be either 'im1' or 'im2' or 'both'")

    # 1) left: data
    if scale == 'asinh':
        if vminvmax_standard == 'im1':
            h1, norm = AsinhStretchPlot(axs[0], im1_disp, a=a, extent=extent, origin='lower', aspect='equal',
                                        return_norm=True)
            h2 = AsinhStretchPlot(axs[1], im2_disp, a=a, extent=extent, origin='lower', aspect='equal', norm=norm)
        elif vminvmax_standard == 'im2':
            h2, norm = AsinhStretchPlot(axs[1], im2_disp, a=a, extent=extent, origin='lower', aspect='equal', return_norm=True)
            h1 = AsinhStretchPlot(axs[0], im1_disp, a=a, extent=extent, origin='lower', aspect='equal', norm=norm)
    else:
        h1 = axs[0].imshow(im1_disp, extent=extent, origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax)
        h2 = axs[1].imshow(im2_disp, extent=extent, origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax)
    h3= axs[2].imshow(residual_map, extent=extent, origin='lower',
                      cmap='bwr', aspect='equal',
                      vmin=-lim if residual_vmin is None else residual_vmin,
                      vmax=+lim if residual_vmax is None else residual_vmax)
    # color bars and titles
    fig.colorbar(h1, ax=axs[0], location='right')
    axs[0].set_title(im1_lab_d)
    fig.colorbar(h2, ax=axs[1], location='right')
    axs[1].set_title(im2_lab_d)
    fig.colorbar(h3, ax=axs[2], location='right')
    axs[2].set_title(res_label)
    for ax in axs[0:3]:
        ax.set_facecolor('k')

    plt.tight_layout()
    return fig, axs

def detailed_comparison_plot(im1, im2, residual_map, *,
                             extent=None,
                             param_best=None, param_unc=None, param_true=None,
                             meta_info_str=None,
                             residual_vmin=None, residual_vmax=None,
                             scale='asinh', a=0.1):
    """
    Requested 2x3 layout:
    Row 1: Observed, Model, Residual
    Row 2: Best Fit params (w/ uncertainties), True params, Meta info
    """
    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    
    # --- ROW 1: Images ---
    # Call internal helper or duplicate logic?
    # Let's duplicate/adapt from comparison_plot for direct control
    
    im1_disp = _prep_log_scale(im1, scale=scale)
    im2_disp = _prep_log_scale(im2, scale=scale)
    
    # Shared vmin/vmax
    vmin = np.nanmin([np.nanmin(im1_disp), np.nanmin(im2_disp)])
    vmax = np.nanmax([np.nanmax(im1_disp), np.nanmax(im2_disp)])
    
    if scale == 'asinh':
        h1, norm = AsinhStretchPlot(axs[0,0], im1_disp, a=a, extent=extent, origin='lower', aspect='equal', return_norm=True)
        h2 = AsinhStretchPlot(axs[0,1], im2_disp, a=a, extent=extent, origin='lower', aspect='equal', norm=norm)
    else:
        h1 = axs[0,0].imshow(im1_disp, extent=extent, origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
        h2 = axs[0,1].imshow(im2_disp, extent=extent, origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
        
    lim = np.nanmax(np.abs(residual_map))
    h3 = axs[0,2].imshow(residual_map, extent=extent, origin='lower', cmap='bwr', aspect='equal',
                         vmin=-lim if residual_vmin is None else residual_vmin,
                         vmax=+lim if residual_vmax is None else residual_vmax)
                         
    fig.colorbar(h1, ax=axs[0,0], location='right', fraction=0.046, pad=0.04)
    axs[0,0].set_title('Observed')
    fig.colorbar(h2, ax=axs[0,1], location='right', fraction=0.046, pad=0.04)
    axs[0,1].set_title('Model')
    fig.colorbar(h3, ax=axs[0,2], location='right', fraction=0.046, pad=0.04)
    axs[0,2].set_title('Residual')
    
    # --- ROW 2: Info ---
    
    # Helper to formatting params
    # param_best, param_unc, param_true are expected to be Dicts or compatible
    
    def format_params(p_dict, u_dict=None):
        lines = []
        if p_dict is None: return ""
        for k, v in p_dict.items():
            if isinstance(v, (float, np.floating)):
                val_str = f"{v:.4g}"
            else:
                val_str = str(v)
            
            if u_dict and k in u_dict:
                 u = u_dict[k]
                 if isinstance(u, (float, np.floating)):
                     lines.append(f"{k}: {val_str} ± {u:.4g}")
                 else:
                     lines.append(f"{k}: {val_str} ± {u}")
            else:
                 lines.append(f"{k}: {val_str}")
        return "\n".join(lines)

    # Ax[1,0]: Best Fit + Uncertainty
    text_best = "Best Fit Parameters:\n------------------\n" + format_params(param_best, param_unc)
    axs[1,0].text(0.05, 0.95, text_best, transform=axs[1,0].transAxes, 
                  verticalalignment='top', fontsize=9, family='monospace')
    axs[1,0].axis('off')
    # axs[1,0].set_title("Best Fit")

    # Ax[1,1]: True Parameters
    text_true = "True Parameters:\n----------------\n" + format_params(param_true)
    axs[1,1].text(0.05, 0.95, text_true, transform=axs[1,1].transAxes, 
                  verticalalignment='top', fontsize=9, family='monospace')
    axs[1,1].axis('off')
    # axs[1,1].set_title("Ground Truth")

    # Ax[1,2]: Meta Info
    text_meta = "Run Info:\n---------\n"
    if meta_info_str:
        text_meta += meta_info_str
    axs[1,2].text(0.05, 0.95, text_meta, transform=axs[1,2].transAxes, 
                  verticalalignment='top', fontsize=9, family='monospace')
    axs[1,2].axis('off')

    plt.tight_layout()
    return fig, axs
