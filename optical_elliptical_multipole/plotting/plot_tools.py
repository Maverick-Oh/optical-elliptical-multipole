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