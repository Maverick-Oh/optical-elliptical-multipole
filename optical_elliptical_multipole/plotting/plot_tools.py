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

def draw_segmentation(ax, segmap, title: str, target_label: int, zeros_mask=None,
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
    ax.set_title(title)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.legend()
    return im, cmap

def AsinhStretchPlot(ax, img:np.ma.core.MaskedArray,
                      title: str|None=None, percentile=99.5, norm=None, return_norm=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    # normalize with percentile stretch
    interval = PercentileInterval(percentile)
    v = interval.get_limits(img[np.isfinite(img)]) if np.isfinite(img).any() else (-1, 1)
    # AsinhStretch makes things brighter: https://docs.astropy.org/en/stable/api/astropy.visualization.AsinhStretch.html
    if norm is None:
        norm = ImageNormalize(vmin=v[0], vmax=v[1], stretch=AsinhStretch())
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

def _prep_scale(img, scale='log', eps=1e-6):
    if scale == 'log':
        # mask non-positive for display; add eps to avoid -inf
        disp = np.log10(np.maximum(img, 0.0) + eps)
    else:
        disp = img
    return disp

def comparison_plot(im1, im2, *, scale='log', labels=None, extent=None, vminvmax_standard='im1'):
    """
    Three-panel comparison plot: data, model, residual.
    Residual uses bwr with symmetric limits so white == 0.
    Axes are equal by default. Data & model share the same color scale.
    """
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))

    im1_label = labels[0] if labels is not None else 'image1'
    im2_label = labels[1] if labels is not None else 'image2'
    im1_lab_d = f'log({im1_label})' if scale == 'log' else im1_label
    im2_lab_d = f'log({im2_label})' if scale == 'log' else im2_label

    im1_disp = _prep_scale(im1, scale=scale)
    im2_disp = _prep_scale(im2, scale=scale)

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
    h1 = axs[0].imshow(im1_disp, extent=extent, origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax)
    fig.colorbar(h1, ax=axs[0], location='right')
    axs[0].set_title(im1_lab_d)

    # 2) middle: model
    h2 = axs[1].imshow(im2_disp, extent=extent, origin='lower', aspect='equal',
                       vmin=vmin, vmax=vmax)
    fig.colorbar(h2, ax=axs[1], location='right')
    axs[1].set_title(im2_lab_d)

    # 3) right: residual = data - model
    diff = im1 - im2
    lim = np.nanmax(np.abs(diff))
    h3 = axs[2].imshow(
        diff, extent=extent, origin='lower',
        cmap='bwr', vmin=-lim, vmax=+lim, aspect='equal'
    )
    fig.colorbar(h3, ax=axs[2], location='right')
    axs[2].set_title(f"Difference ({im1_label} - {im2_label})")

    plt.tight_layout()
    return fig, axs
