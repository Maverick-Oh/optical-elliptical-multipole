import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import os

from matplotlib.ticker import MaxNLocator

def AsinhStretchPlot(axis, data, a=0.1, vmin=None, vmax=None, return_norm=False, *args, **kwargs):
    # For asinh normalized plot
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    norm = kwargs.pop('norm', None)
    if norm is None:
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(a=a))
    
    im = axis.imshow(data, *args, norm=norm, **kwargs)
    if return_norm:
        return im, norm
    else:
        return im

def _prep_log_scale(image, scale='asinh'):
    """Prepare data for log or asinh scaling."""
    # If already a MaskedArray, keep it
    if isinstance(image, np.ma.MaskedArray):
        return image
    # Otherwise, just return
    return image

def comparison_plot(im1, im2, *,
                    residual_map=None, labels=('Image1', 'Image2', 'Residual'),
                    extent=None, scale='asinh', a=0.1,
                    residual_vmin=None, residual_vmax=None,
                    extra_text=None):
    """
    3x1 comparison plot for quick checks.
    (For final outputs, use detailed_comparison_plot instead.)
    """
    im1_disp = _prep_log_scale(im1, scale=scale)
    im2_disp = _prep_log_scale(im2, scale=scale)

    if residual_map is None:
        residual_map = im1 - im2
    else:
        residual_map = _prep_log_scale(residual_map, scale='linear')

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Shared vmin/vmax for im1, im2
    vmin = np.nanmin([np.nanmin(im1_disp), np.nanmin(im2_disp)])
    vmax = np.nanmax([np.nanmax(im1_disp), np.nanmax(im2_disp)])

    if scale == 'asinh':
        h1, norm = AsinhStretchPlot(axs[0], im1_disp, a=a, extent=extent, origin='lower', aspect='equal', return_norm=True)
        h2 = AsinhStretchPlot(axs[1], im2_disp, a=a, extent=extent, origin='lower', aspect='equal', norm=norm)
    else:
        h1 = axs[0].imshow(im1_disp, extent=extent, origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
        h2 = axs[1].imshow(im2_disp, extent=extent, origin='lower', aspect='equal', vmin=vmin, vmax=vmax)

    lim = np.nanmax(np.abs(residual_map))
    h3 = axs[2].imshow(residual_map, extent=extent, origin='lower', cmap='bwr', aspect='equal',
                       vmin=-lim if residual_vmin is None else residual_vmin,
                       vmax=+lim if residual_vmax is None else residual_vmax)

    fig.colorbar(h1, ax=axs[0], fraction=0.046, pad=0.04)
    axs[0].set_title(labels[0])
    fig.colorbar(h2, ax=axs[1], fraction=0.046, pad=0.04)
    axs[1].set_title(labels[1])
    fig.colorbar(h3, ax=axs[2], fraction=0.046, pad=0.04)
    axs[2].set_title(labels[2])

    if extra_text:
        axs[1].text(1.05, 0.5, extra_text, transform=axs[1].transAxes,
                    verticalalignment='center', fontsize=10, family='monospace')

    plt.tight_layout()
    return fig, axs

def detailed_comparison_plot(im1, im2, residual_map, *,
                             extent=None,
                             param_best=None, param_unc=None, param_true=None,
                             meta_info_str=None,
                             filename_sci=None,
                             residual_vmin=-5, residual_vmax=5,
                             scale='asinh', a=0.1):
    """
    2x3 layout for detailed mock fitting validation:
    Row 1: Observed, Model, Residual
    Row 2: Best/Initial Params (w/ uncertainties), True Params, Meta Info
    
    Parameters
    ----------
    im1 : array-like
        Observed image (masked)
    im2 : array-like
        Model image
    residual_map : array-like
        Residual map in sigma units
    extent : list
        Matplotlib extent for imshow
    param_best : dict
        Best-fit or initial parameters (ordered dict recommended)
    param_unc : dict, optional
        Uncertainties on parameters (same keys as param_best)
    param_true : dict, optional
        True parameters from simulation
    meta_info_str : str, optional
        Meta information to display
    filename_sci : str, optional
        Source filename (will display basename only)
    residual_vmin, residual_vmax : float
        Colormap limits for residual plot
    scale : str
        Scaling for image display ('asinh' or 'linear')
    a : float
        AsinhStretch parameter
    """
    nrows, ncols = 2, 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.5*nrows))
    
    # --- ROW 1: Images ---
    im1_disp = _prep_log_scale(im1, scale=scale)
    im2_disp = _prep_log_scale(im2, scale=scale)
    
    # Shared vmin/vmax
    vmin = np.nanmin([np.nanmin(im1_disp), np.nanmin(im2_disp)])
    vmax = np.nanmax([np.nanmax(im1_disp), np.nanmax(im2_disp)])
    
    if scale == 'asinh':
        h1, norm = AsinhStretchPlot(axs[0,0], im1_disp, a=a, vmin=vmin, vmax=vmax, extent=extent, origin='lower', aspect='equal', return_norm=True)
        h2 = AsinhStretchPlot(axs[0,1], im2_disp, a=a, vmin=vmin, vmax=vmax, extent=extent, origin='lower', aspect='equal', norm=norm)
    else:
        h1 = axs[0,0].imshow(im1_disp, extent=extent, origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
        h2 = axs[0,1].imshow(im2_disp, extent=extent, origin='lower', aspect='equal', vmin=vmin, vmax=vmax)
    
    # Residual with extend='both' to show values outside cmap range
    h3 = axs[0,2].imshow(residual_map, extent=extent, origin='lower', cmap='bwr', aspect='equal',
                         vmin=residual_vmin, vmax=residual_vmax)
    
    fig.colorbar(h1, ax=axs[0,0], fraction=0.046, pad=0.04)
    axs[0,0].set_title('Observed')
    fig.colorbar(h2, ax=axs[0,1], fraction=0.046, pad=0.04)
    axs[0,1].set_title('Model')
    cbar3 = fig.colorbar(h3, ax=axs[0,2], fraction=0.046, pad=0.04, extend='both')
    axs[0,2].set_title('Residual (σ)')
    
    # --- ROW 2: Parameters and Info ---
    
    # Define canonical parameter order (matching physics/fitting order)
    param_order = [
        'n_sersic', 'R_sersic', 'amplitude', 
        'q', 'theta_ell',
        'a_m3', 'phi_m3', 'a_m4', 'phi_m4',
        'x0', 'y0', 'background'
    ]
    
    def format_params(p_dict, u_dict=None, order=param_order):
        """Format parameters in canonical order with optional uncertainties."""
        lines = []
        if p_dict is None: 
            return ""
        
        # First add params in canonical order (if present)
        for k in order:
            if k in p_dict:
                v = p_dict[k]
                if isinstance(v, (float, np.floating)):
                    val_str = f"{v:.4g}"
                else:
                    val_str = str(v)
                
                if u_dict and k in u_dict:
                    u = u_dict[k]
                    if isinstance(u, (float, np.floating)):
                        lines.append(f"{k:12s}: {val_str:>10s} ± {u:.4g}")
                    else:
                        lines.append(f"{k:12s}: {val_str}")
                else:
                    lines.append(f"{k:12s}: {val_str:>10s}")
        
        # Then add any remaining params not in canonical order
        for k, v in p_dict.items():
            if k in ['filename_sci', 'filename_wht', 'segmap_file']: continue # Skip filename paths in param list
            if k not in order:
                if isinstance(v, (float, np.floating)):
                    val_str = f"{v:.4g}"
                else:
                    val_str = str(v)
                lines.append(f"{k:12s}: {val_str:>10s}")
        
        return "\n".join(lines)
    
    # Ax[1,0]: True Parameters
    text_true = "True Parameters:\n" + "-" * 16 + "\n" + format_params(param_true)
    axs[1,0].text(0.05, 0.95, text_true, transform=axs[1,0].transAxes, 
                  verticalalignment='top', fontsize=9, family='monospace')
    axs[1,0].axis('off')

    # Ax[1,1]: Best Fit or Initial Parameters + Uncertainty
    title_left = "Best Fit Parameters:" if param_unc else "Initial Parameters:"
    text_best = f"{title_left}\n" + "-" * len(title_left) + "\n" + format_params(param_best, param_unc)
    axs[1,1].text(0.05, 0.95, text_best, transform=axs[1,1].transAxes, 
                  verticalalignment='top', fontsize=9, family='monospace')
    axs[1,1].axis('off')

    # Ax[1,2]: Meta Info
    text_meta = "Run Info:\n" + "-" * 9 + "\n"
    if meta_info_str:
        text_meta += meta_info_str
    if filename_sci:
        # Display basename only, not full path
        text_meta += f"\nFile: {os.path.basename(filename_sci)}"
    elif param_true and 'filename_sci' in param_true:
        # Fallback: Try to get filename from param_true if not provided as arg
        text_meta += f"\nFile: {os.path.basename(param_true['filename_sci'])}"
    axs[1,2].text(0.05, 0.95, text_meta, transform=axs[1,2].transAxes, 
                  verticalalignment='top', fontsize=9, family='monospace')
    axs[1,2].axis('off')

    plt.tight_layout()
    return fig, axs

# Additional helper functions for draw_segmentation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from scipy.ndimage import binary_dilation

def _discrete_cmap(n, target_index=1, randomize=False, seed=42):
    """Build a discrete ListedColormap with n+1 entries (0..n).
    - Keeps label indices unchanged.
    - Sets color at target_index to white.
    - If randomize=True, shuffles colors for indices 1..n excluding target_index.
    """
    base = plt.get_cmap('hsv', n + 1)
    colors = base(np.arange(n + 1))
    colors[target_index] = np.array([1.0, 1.0, 1.0, 1.0])
    cmap = ListedColormap(colors)
    return cmap

def draw_segmentation(ax, segmap, target_label: int, title=None, zeros_mask=None,
                      a50=None, a90=None, a99=None, q=None, theta=None, center_xy=None, seed=42, outline=False,
                      randomize_cmap=False, **kwargs):
    # base layer: grayscale of segmap (masked zeros)
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
    if title is not None:
        ax.set_title(title)
    return im, cmap

def plot_masked_and_cropped(sci, mask, wht=None, extent=None, filename_sci=None, out_path=None, cropped=True):
    """
    Generate the *-02-bg_and_segmap.pdf and *-03-masked_and_cropped.pdf plot.
    Shows the Science image with mask applied (NaNs) to verify input data.
    """
    fig, axs = plt.subplots(1, 2 if wht is not None else 1, figsize=(10 if wht is not None else 5, 5))
    if wht is None:
        axs = [axs]
        
    # Prepare masked array
    sci_masked = np.ma.masked_array(sci, mask=mask)
    
    # Plot Science
    h1 = AsinhStretchPlot(axs[0], sci_masked, a=0.1, extent=extent, origin='lower')
    if cropped:
        axs[0].set_title('Masked & Cropped SCI')
    else:
        axs[0].set_title('Masked SCI')
    fig.colorbar(h1, ax=axs[0], fraction=0.046, pad=0.04)
    
    # Plot Weight if provided
    if wht is not None:
        # Weight usually visualized linear or asinh
        h2 = AsinhStretchPlot(axs[1], wht, a=0.1, extent=extent, origin='lower')
        if cropped:
            axs[1].set_title('Cropped WHT')
        else:
            axs[1].set_title('WHT')
        fig.colorbar(h2, ax=axs[1], fraction=0.046, pad=0.04)

    if filename_sci:
        plt.suptitle(f"File: {os.path.basename(filename_sci)}", fontsize=10)
        
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig, axs

def plot_sep_steps(sci, sci_bgsub, wht, segmap, 
                   target_label=None, target_xy=None, 
                   extent=None, filename_sci=None, out_path=None):
    """
    Generate the 5-panel plot showing SCI, SCI-BKG, WHT, 1/sqrt(WHT), and Segmentation.
    Replicates the style from preprocess_COSMOS_w_source_extractor.py.
    """
    fig_hw_unit_inch = 5
    fig, axes = plt.subplots(1, 5, figsize=(fig_hw_unit_inch*5, fig_hw_unit_inch))
    
    # Set titles
    axes[0].set_title("SCI")
    axes[1].set_title("SCI - BKG")
    axes[2].set_title("WHT")
    axes[3].set_title("1/sqrt(WHT)")
    
    # 1. SCI
    im_img, norm = AsinhStretchPlot(axes[0], sci, origin="lower", return_norm=True, extent=extent)
    plt.colorbar(mappable=im_img, ax=axes[0], fraction=0.046, pad=0.04)
    if target_xy is not None:
        axes[0].axvline(target_xy[0], ymin=0., ymax=sci.shape[0], color='w',
                         linewidth=0.5, linestyle='--')
        axes[0].axhline(target_xy[1], xmin=0., xmax=sci.shape[1], color='w',
                         linewidth=0.5, linestyle='--')

    # 2. SCI - BKG
    im_img_bkgsub = axes[1].imshow(sci_bgsub, origin="lower", norm=norm, extent=extent)
    plt.colorbar(mappable=im_img_bkgsub, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. WHT
    im_wht = AsinhStretchPlot(axes[2], wht, origin='lower', extent=extent)
    plt.colorbar(mappable=im_wht, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. 1/sqrt(WHT) aka RMS
    with np.errstate(divide='ignore', invalid='ignore'):
        rms = 1.0/np.sqrt(wht)
    im_wht_sqrt_inverse = AsinhStretchPlot(axes[3], rms, origin='lower', extent=extent)
    plt.colorbar(mappable=im_wht_sqrt_inverse, ax=axes[3], fraction=0.046, pad=0.04)

    # 5. Segmentation
    im, cmap = draw_segmentation(axes[4], segmap, title='Segmentation', target_label=target_label,
                        outline = False, origin='lower', extent=extent)
    cbar = plt.colorbar(mappable=im, ax=axes[4], fraction=0.046, pad=0.04)
    cbar.locator = MaxNLocator(integer=True, nbins=5)
    cbar.update_ticks()

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x (px)')

    if filename_sci:
        fig.suptitle(f"File: {os.path.basename(filename_sci)}", fontsize=16)

    fig.tight_layout()
    
    if out_path:
        fig.savefig(out_path)
        plt.close(fig)
    else:
        return fig, axes

