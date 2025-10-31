import numpy as np
from matplotlib import pyplot as plt

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

def comparison_plot(im1, im2, *, scale='log', labels=None, extent=None):
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
    vmin = np.nanmin([np.nanmin(im1_disp), np.nanmin(im2_disp)])
    vmax = np.nanmax([np.nanmax(im1_disp), np.nanmax(im2_disp)])

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
