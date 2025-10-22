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


def comparison_plot(im1, im2, scale='log', labels=None, extent=None):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    # prep
    im1_ = np.log10(im1) if scale=='log' else im1
    im2_ = np.log10(im2) if scale=='log' else im2
    im1_label = labels[0] if labels is not None else 'image1'; im1_label_ = 'log('+im1_label+')' if scale=='log' else im1_label
    im2_label = labels[1] if labels is not None else 'image2'; im2_label_ = 'log('+im2_label+')' if scale=='log' else im2_label
    # 1) I_oem
    imshow1 = axs[0].imshow(im1_, extent=extent, origin='lower', aspect='equal')
    fig.colorbar(imshow1, ax=axs[0], location='right')
    axs[0].set_title(im1_label_)
    # 2) I_lenstronomy
    imshow2 = axs[1].imshow(im2_, extent=extent, origin='lower', aspect='equal')
    fig.colorbar(imshow2, ax=axs[1], location='right')
    axs[1].set_title(im2_label_)
    # 3) Difference with fixed ±10 and white at 0
    diff = im1 - im2
    diff_scale = max(abs(diff.min()), abs(diff.max()))
    imshow3 = axs[2].imshow(diff, extent=extent, origin='lower',
                        cmap='bwr', vmin=-diff_scale, vmax=diff_scale, aspect='equal')
    fig.colorbar(imshow3, ax=axs[2], location='right')
    axs[2].set_title(f"Difference ({im1_label} - {im2_label})")
    # show
    plt.tight_layout()
    return fig, axs