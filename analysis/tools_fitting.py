import numpy as np
# OEM (non-JAX) imports — keep these paths as requested
from optical_elliptical_multipole.nonjax.intensity_functions import sersic
from optical_elliptical_multipole.nonjax.profiles1D import Elliptical_Multipole_Profile_1D
from optical_elliptical_multipole.nonjax.profiles2D import Elliptical_Multipole_Profile_2D
from optical_elliptical_multipole.plotting.plot_tools import comparison_plot, detailed_comparison_plot
import warnings
import os
import matplotlib.pyplot as plt
# from astropy.io import fits
import h5py
from scipy.optimize import minimize, Bounds
import time
import copy
from tools_misc import dict2str_newline
import emcee
import os
import h5py

def build_arcsec_grid(shape, pixscale=0.03):
    """
    Half-pixel centering:
      odd N: centers land exactly at 0
      even N: symmetric about 0, no pixel at 0
    Returns X, Y (arcsec), and imshow extent.
    """
    ny, nx = shape
    # x
    if nx % 2 == 1:
        x = (np.arange(nx) - (nx - 1) / 2.0) * pixscale
    else:
        x = (np.arange(nx) - nx / 2.0 + 0.5) * pixscale
    # y
    if ny % 2 == 1:
        y = (np.arange(ny) - (ny - 1) / 2.0) * pixscale
    else:
        y = (np.arange(ny) - ny / 2.0 + 0.5) * pixscale

    X, Y = np.meshgrid(x, y)
    extent = [x.min() - pixscale / 2, x.max() + pixscale / 2,
              y.min() - pixscale / 2, y.max() + pixscale / 2]
    return X, Y, extent

def l2_mean(a, b, mask=None):
    """Mean squared residuals over finite pixels (ignores NaN/Inf automatically)."""
    diff = (a - b)
    if mask is None:
        mask = np.isfinite(diff)
    else:
        mask = mask & np.isfinite(diff)
    if not np.any(mask):
        return np.nan
    d = diff[mask]
    return np.mean(d * d)

def residual_map_sigma(SCI, WHT, model, EXP_TIME, mask=None):
    sigma_tot_sq = sigma_total_squared(WHT, SCI, EXP_TIME, poisson_threshold_n=3., verbose=True)
    diff = SCI - model
    diff_sigma = diff / np.sqrt(sigma_tot_sq)
    if mask is not None:
        diff_sigma = np.ma.masked_array(diff_sigma, mask=mask)
    return diff_sigma

def chi_squared(SCI, model, sigma_tot_squared, mask=None):
    dif = SCI - model
    chi2_map = dif*dif / sigma_tot_squared
    if mask is not None:
        chi2_map = np.ma.masked_array(chi2_map, mask=mask)
    return np.sum(chi2_map)

def reduced_chi_squared(SCI, WHT, model, n_param_model, EXP_TIME, mask=None):
    sigma_tot_sq = sigma_total_squared(WHT, SCI, EXP_TIME, poisson_threshold_n=3., verbose=True, mask=mask)
    chi2 = chi_squared(SCI, model, sigma_tot_sq, mask=mask)
    pixel_tot = np.prod(SCI.shape) - np.sum(mask) # total number of pixels
    chi2_reduced_map = chi2 / (pixel_tot - n_param_model)
    return np.sum(chi2_reduced_map)

def sigma_total_squared(WHT, SCI, EXP_TIME, poisson_threshold_n=3., mask=None, verbose=True):
    # sigma_tot^2 = 1/WHT + SCI/EXP_TIME
    # poisson_mask: True to mask out; False to let values in. typically values below three sigma.
    # Apply poisson mask only for SCI > poisson_threshold_n * sigma_BKG; poisson_threshold_n is 3 and sigma_BKG is 1/sqrt(WHT)
    # So faster calculation becomes the vollowing:
    if np.isnan(EXP_TIME):
        raise ValueError(f'EXP_TIME cannot be NaN! Currently it is: {EXP_TIME}')
    poisson_mask = (SCI<0) + (SCI * SCI < poisson_threshold_n * poisson_threshold_n * (1 / WHT)) # True to mask out
    sigma_background_squared = 1/WHT
    sigma_poisson_squared = SCI/EXP_TIME # in flux
    sigma_poisson_squared[poisson_mask] = 0.
    sigma_tot_squared = sigma_background_squared + sigma_poisson_squared
    if mask is not None:
        sigma_tot_squared = np.ma.masked_array(sigma_tot_squared, mask=mask)
    return sigma_tot_squared

def analytic_amplitude(data, model_unit, background, mask=None):
    """
    Best scalar a for D ≈ a*M + B (in least squares), with B fixed.
    a = <D - B, M> / <M, M>
    Ref: normal equations for linear least squares (one-parameter).
    """
    D = data - background
    M = model_unit
    if mask is None:
        mask = ~(np.isfinite(D) & np.isfinite(M))
    else:
        pass
    num = np.sum(D[~mask] * M[~mask])
    den = np.sum(M[~mask] * M[~mask])
    if den <= 0:
        return 1.0
    return float(num / den)

def masked_from_image(img):
    """Mask non-finite pixels."""
    return np.isfinite(img)

def _warn_and_write_missing(missing_ids, data_dir):
    if len(missing_ids) == 0:
        return
    miss_path = os.path.join(data_dir, "__missing_fits.txt")
    with open(miss_path, "w") as f:
        for sid in missing_ids:
            f.write(f"{sid}\n")
    warnings.warn(f"{len(missing_ids)} FITS files missing; wrote {miss_path}")

# ---------------------------
# model evaluation (elliptical multipoles)
# ---------------------------
def simulate_model_elliptical_multipole(
    X, Y, *,
    n_sersic, R_sersic, amplitude,
    q, theta_ell,
    m, a_m, phi_m,
    x0, y0,
    background=0.0,
    psf=None  # placeholder
):
    """
    Build model image: elliptical multipole Sérsic + background.
    PSF hook is not implemented yet (placeholder).
    """
    if psf is not None:
        raise NotImplementedError("PSF convolution pending (hook present).")

    # Elliptical Multipole profile (removes multipole from circularized radius, then sersic)
    I = Elliptical_Multipole_Profile_2D(
        X, Y, sersic,
        q=q, theta_ell=theta_ell,
        m=np.asarray(m), a_m=np.asarray(a_m), phi_m=np.asarray(phi_m),
        x0=x0, y0=y0,
        amplitude=amplitude, R_sersic=R_sersic, n_sersic=n_sersic
    )
    return I + background

# ---------------------------
# parameter packing
# ---------------------------
def pack_params(p, m_len):
    """
    Pack dict -> vector in stable order.
    p must have: n_sersic, R_sersic, amplitude, q, theta_ell,
                 a_m (len=m_len), phi_m (len=m_len), x0, y0, background
    """
    vec = [
        p['n_sersic'],
        p['R_sersic'],
        p['amplitude'],
        p['q'],
        p['theta_ell'],
    ]
    vec.extend(p['a_m'][:m_len])
    vec.extend(p['phi_m'][:m_len])
    vec.extend([p['x0'], p['y0'], p['background']])
    return np.asarray(vec, dtype=float)

def unpack_params(vec, m_len):
    """
    Inverse of pack_params.
    """
    idx = 0
    n_sersic  = vec[idx]; idx += 1
    R_sersic  = vec[idx]; idx += 1
    amplitude = vec[idx]; idx += 1
    q         = vec[idx]; idx += 1
    theta_ell = vec[idx]; idx += 1

    a_m   = np.array(vec[idx:idx+m_len]); idx += m_len
    phi_m = np.array(vec[idx:idx+m_len]); idx += m_len

    x0 = vec[idx]; idx += 1
    y0 = vec[idx]; idx += 1
    background = vec[idx]; idx += 1

    return dict(
        n_sersic=n_sersic, R_sersic=R_sersic, amplitude=amplitude,
        q=q, theta_ell=theta_ell, a_m=a_m, phi_m=phi_m,
        x0=x0, y0=y0, background=background
    )

def default_sigma(m_array, img_half_extent):
    # 'n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell',
    sigma = [0.5, 0.5, 0.01, 0.1, np.pi/5]
    # a_m
    sigma += [0.01]*len(m_array)
    # phi_m
    sigma += (np.pi/2/m_array/5).tolist()
    # x0, y0
    sigma += [img_half_extent/20, img_half_extent/20]
    # background
    sigma += [0.001]
    return sigma

def default_bounds(m_array, img_half_extent):
    """
    Build lower/upper bounds for least_squares.
    a_m bounds: [-0.1, 0.1] per instruction.
    """
    m_array = np.asarray(m_array) if type(m_array)==list else m_array
    # core scalars
    # 'n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell',
    lo = [0.2, 1e-6, 0.0, 1e-6, -np.pi] # q minimum 1e-6 to avoid being zero
    hi = [20.0,  np.inf, np.inf, 1.0,  np.pi]
    # a_m
    lo += [-0.1]*len(m_array)
    hi += [ 0.1]*len(m_array)
    # phi_m
    lo += (np.array([-np.pi]*len(m_array))/(2 * m_array)).tolist()
    hi += (np.array([ np.pi]*len(m_array))/(2 * m_array)).tolist()
    # x0, y0
    lo += [-img_half_extent, -img_half_extent]
    hi += [ img_half_extent,  img_half_extent]
    # background
    lo += [-0.1]
    hi += [ 0.1]
    return np.array(lo, float), np.array(hi, float)

def jacobian_error_estimate(v_best, residual_fn, rel_step=1e-6, abs_step=0.0, verbose=False):
    """
    Estimate 1σ uncertainties for parameters at v_best using a numerical Jacobian
    of the residual vector.

    Parameters
    ----------
    v_best : array-like, shape (P,)
        Best-fit parameter vector.
    residual_fn : callable
        Function residual_fn(v) -> 1D array of residuals r (already normalized
        by per-pixel sigma, e.g. (data - model)/sigma). Shape (N,).
    rel_step : float
        Relative step size for finite differences.
    abs_step : float
        Absolute additive step (in case some v_best are near zero).
    verbose : bool
        If True, prints some diagnostic info.

    Returns
    -------
    v_err : ndarray, shape (P,)
        Approximate 1σ uncertainties for each parameter.
        np.nan if covariance cannot be estimated (e.g. N <= P).
    """
    v_best = np.asarray(v_best, dtype=float)
    r0 = residual_fn(v_best)

    r0 = np.asarray(r0, dtype=float).ravel()
    N = r0.size
    P = v_best.size

    if verbose:
        print(f"[jacobian_error_estimate] N={N} residuals, P={P} params")

    # If not enough data points, cannot estimate covariance reliably
    if N <= P:
        if verbose:
            print("[jacobian_error_estimate] N <= P, returning NaNs")
        return np.full(P, np.nan, dtype=float)

    # Build Jacobian via central finite differences
    J = np.empty((N, P), dtype=float)
    step = rel_step * (np.abs(v_best) + 1.0) + abs_step

    for j in range(P):
        h = step[j]
        if h == 0 or not np.isfinite(h):
            # If step is zero/NaN, treat as zero-sensitivity
            J[:, j] = 0.0
            continue

        v_plus = v_best.copy()
        v_minus = v_best.copy()
        v_plus[j] += h
        v_minus[j] -= h

        r_plus = np.asarray(residual_fn(v_plus), dtype=float).ravel()
        r_minus = np.asarray(residual_fn(v_minus), dtype=float).ravel()

        # Sanity check on residual length
        if r_plus.size != N or r_minus.size != N:
            raise ValueError(
                "residual_fn returned arrays of inconsistent length when "
                f"perturbing parameter {j} (expected {N}, got {r_plus.size} / {r_minus.size})"
            )

        J[:, j] = (r_plus - r_minus) / (2.0 * h)

    # chi^2 = sum r^2 (because residuals are already normalized by sigma)
    chi2 = float(np.dot(r0, r0))
    dof = max(N - P, 1)
    sigma2 = chi2 / dof

    if verbose:
        print(f"[jacobian_error_estimate] chi2={chi2:.3g}, dof={dof}, sigma2={sigma2:.3g}")

    JTJ = J.T @ J

    # Use pseudo-inverse for robustness
    try:
        cov = sigma2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        if verbose:
            print("[jacobian_error_estimate] JTJ not invertible, using pseudo-inverse")
        cov = sigma2 * np.linalg.pinv(JTJ)

    # 1σ uncertainties are sqrt of diagonal elements
    diag = np.diag(cov)
    diag = np.clip(diag, 0.0, np.inf)  # guard against tiny negative numerics
    v_err = np.sqrt(diag)
    return v_err


# ---------------------------
# per-target workflow
# ---------------------------
def configured_optimizer(loss, v0, lo, hi, opt_method):
    bounds_list = list(zip(lo, hi))  # for methods that accept list-of-pairs
    if opt_method=='COBYQA':
        opt = minimize(loss, v0, bounds=bounds_list,
                       method='COBYQA',
                       options=dict(disp=False))  # bounds already computed above as lo, hi; v0 is your initial vector
    elif opt_method=='L-BFGS-B':
        opt = minimize(
            loss,
            v0,
            method="L-BFGS-B",
            bounds=bounds_list,
            jac=None,  # or pass a gradient function if you have one
            options=dict(disp=False, maxiter=500)
        )
    elif opt_method=='SLSQP':
        opt = minimize(
            loss,
            v0,
            method="SLSQP",
            bounds=bounds_list,
            jac=None,  # optional gradient
            options=dict(disp=False, maxiter=500)
        )
    elif opt_method=='Powell':
        opt = minimize(
            loss,
            v0,
            method="Powell",
            bounds=bounds_list,
            options=dict(disp=False, maxiter=500)
        )
    elif opt_method=='trust-constr':
        bounds_obj = Bounds(lo, hi)  # for trust-constr
        opt = minimize(
            loss,
            v0,
            method="trust-constr",
            bounds=bounds_obj,  # needs a Bounds object
            jac=None,  # can pass gradient if available
            options=dict(verbose=0, maxiter=300)
        )
    else:
        raise ValueError(f"unknown optimization method: {opt_method}")
    return opt

def downsample(img, factor):
    """
    Downsample image by block averaging.
    img shape must be (h*factor, w*factor)
    """
    if factor == 1:
        return img
    h, w = img.shape
    return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def process_one_target_optimize(row_query, data_dir, row_sep, m,
                                opt_method,
                                PIX_SCALE=0.03,
                                plot_initial_contour=True,
                                plot_final_contour=True,
                                fit_model=True,
                                verbose=False,
                                debug=False,
                                target_loss=1.0,
                                supersample_factor=1,
                                truth_row=None): # Added truth_row
    # processing with updated method using weight map
    # ... (same docstring) ...
    seqid_str = str(int(row_query['sequentialid']))

    rec = dict(sequentialid=seqid_str) 

    # Load Data (HDF5)
    filename_hdf5 = os.path.join(data_dir, seqid_str + '-cropped.hdf5')
    assert os.path.exists(filename_hdf5)
    with h5py.File(filename_hdf5, "r") as data_file:
        sci_bgsub = np.array(data_file['sci_bgsub_crop'])
        wht       = np.array(data_file['wht_crop'])
        mask      = np.array(data_file['mask_crop'])
        seg       = np.array(data_file['segmap_crop'])

    # Standard Grid
    X, Y, extent = build_arcsec_grid(sci_bgsub.shape, pixscale=PIX_SCALE)
    
    # Supersampled Grid
    if supersample_factor > 1:
        ny, nx = sci_bgsub.shape
        shape_ss = (ny * supersample_factor, nx * supersample_factor)
        pixscale_ss = PIX_SCALE / supersample_factor
        X_ss, Y_ss, _ = build_arcsec_grid(shape_ss, pixscale=pixscale_ss)
    else:
        X_ss, Y_ss = X, Y

    q = row_sep['q']
    if q<=0. or q>=1.:
        raise ValueError("q must be between 0 and 1.")
    theta_ell = row_sep['theta']

    # Initial Guesses
    R_sersic0 = row_query.get('r_gim2d', default=np.nan) 
    if np.isnan(R_sersic0):
        R_sersic0 = row_sep['R50'] * PIX_SCALE
        rec.update(initial_R_sersic0_from='SEP R50')
    else:
        rec.update(initial_R_sersic0_from='r_gim2d')
    n_sersic0 = row_query.get('sersic_n_gim2d', default=np.nan)
    if np.isnan(n_sersic0):
        n_sersic0 = 1.0
        rec.update(initial_n_sersic0_from='default 1.0')
    else:
        rec.update(initial_n_sersic0_from='sersic_n_gim2d')

    x0_init = 0.
    y0_init = 0.
    m = np.asarray(m, int)
    k = len(m)
    a_m0 = np.zeros(k, dtype=float)
    phi_m0 = np.zeros(k, dtype=float)
    bg0 = 0. 

    # pack initial vector
    p0_elliptical_multipole = dict(
                                n_sersic=n_sersic0, R_sersic=R_sersic0, amplitude=1.0,
                                q=q, theta_ell=theta_ell, a_m=a_m0, phi_m=phi_m0,
                                x0=x0_init, y0=y0_init, background=bg0
                            )

    p0_for_profile = p0_elliptical_multipole.copy()
    if 'background' in p0_for_profile:
        del p0_for_profile['background']

    model_unit_ss = Elliptical_Multipole_Profile_2D(
        X_ss, Y_ss, sersic, m=m, **p0_for_profile
    )
    if supersample_factor > 1:
        model_unit = downsample(model_unit_ss, supersample_factor)
    else:
        model_unit = model_unit_ss
        
    I0 = analytic_amplitude(sci_bgsub, model_unit, bg0, mask=mask)
    p0_elliptical_multipole['amplitude'] = I0

    # Initial Plot (using simple comparison plot)
    if plot_initial_contour: # Keeping usage of comparison_plot for initial check
        model0 = I0 * model_unit + bg0
        n_param_model = 5 + 3*len(m) 
        residual_map = residual_map_sigma(sci_bgsub, wht, model0, row_query['EXPTIME_SCI'], mask=mask)
        # ... logic as before for 04-before_fitting ...
        # (Assuming existing logic handles this, or I re-insert it if I replaced it. 
        #  I am replacing lines 382...600, so I must preserve everything I want.)

    # ... RE-INSERT INITIAL PLOT & FIT LOGIC ...
    # ---------------------------
    # Initial Plot Generation
    # ---------------------------
    model0 = I0 * model_unit + bg0
    n_param_model = 5 + 3*len(m) 
    chi2_reduced_0 = reduced_chi_squared(sci_bgsub, wht, model0, n_param_model, row_query['EXPTIME_SCI'], mask=mask)
    residual_map = residual_map_sigma(sci_bgsub, wht, model0, row_query['EXPTIME_SCI'], mask=mask)

    p0_model = copy.copy(p0_elliptical_multipole)
    
    fig1, axs1 = comparison_plot(
        np.ma.masked_array(sci_bgsub, mask=mask), model0, residual_map=residual_map, scale='asinh',
        labels=('Observed', 'Initial model', 'residual (σ)'),
        extent=extent, residual_vmin=-5, residual_vmax=5, extra_text=dict2str_newline(p0_model),
    )
    
    # Overlay if needed
    if plot_initial_contour:
        x1, y1 = Elliptical_Multipole_Profile_1D(
            r0=R_sersic0, q=q, theta_ell=theta_ell, m=m, a_m=a_m0,
            phi_m=phi_m0, x0=x0_init, y0=y0_init, return_type='xy',
            include_end=True, n_points=300
        )
        axs1[1].plot(x1, y1, color='k', lw=1.0)

    out1 = os.path.join(data_dir, f"{seqid_str}-04-before_fitting.pdf")
    fig1.tight_layout()
    fig1.savefig(out1, bbox_inches='tight')
    plt.close(fig1)
    
    # ... Rec Update ...
    rec.update(loss_initial=chi2_reduced_0) # Simplified update

    if not fit_model:
        return rec

    # Limits and Optimization 
    img_half_extent = 0.5 * max(extent[1] - extent[0], extent[3] - extent[2])
    lo, hi = default_bounds(m, img_half_extent)
    
    # Precompute sigma_tot
    sigma_tot = np.sqrt(sigma_total_squared(
        wht, sci_bgsub, row_query['EXPTIME_SCI'],
        poisson_threshold_n=3., mask=mask, verbose=False
    ))

    # Define Loss and Residual using SS
    def loss(vec):
        pp = unpack_params(vec, k)
        # Stability
        if pp['R_sersic'] <= 0 or pp['n_sersic'] <= 0 or pp['q'] <= 0: return np.inf
        
        mod_ss = simulate_model_elliptical_multipole(
            X_ss, Y_ss,
            n_sersic=pp['n_sersic'], R_sersic=pp['R_sersic'], amplitude=pp['amplitude'],
            q=pp['q'], theta_ell=pp['theta_ell'], m=m, a_m=pp['a_m'], phi_m=pp['phi_m'],
            x0=pp['x0'], y0=pp['y0'], background=0.0,
            psf=None
        )
        if supersample_factor > 1:
            mod = downsample(mod_ss, supersample_factor)
        else:
            mod = mod_ss
        mod = mod + pp['background']
        
        res = reduced_chi_squared(sci_bgsub, wht, mod, n_param_model, row_query['EXPTIME_SCI'], mask=mask)
        return res

    def residual_vector(vec):
        # ... logic as above ...
        pp = unpack_params(vec, k)
        mod_ss = simulate_model_elliptical_multipole(
            X_ss, Y_ss,
            n_sersic=pp['n_sersic'], R_sersic=pp['R_sersic'], amplitude=pp['amplitude'],
            q=pp['q'], theta_ell=pp['theta_ell'], m=m, a_m=pp['a_m'], phi_m=pp['phi_m'],
            x0=pp['x0'], y0=pp['y0'], background=0.0,
            psf=None
        )
        if supersample_factor > 1:
            mod = downsample(mod_ss, supersample_factor)
        else:
            mod = mod_ss
        mod = mod + pp['background']
        
        diff = sci_bgsub - mod
        if mask is not None:
            valid = (~mask) & np.isfinite(diff) & np.isfinite(sigma_tot)
        else:
            valid = np.isfinite(diff) & np.isfinite(sigma_tot)
        if not np.any(valid): return np.array([], dtype=float)
        return diff[valid] / sigma_tot[valid]

    # Run Optimization
    v0 = pack_params(p0_elliptical_multipole, k)
    start_time = time.time()
    res = configured_optimizer(loss, v0, lo, hi, opt_method)
    
    # Handle Result
    v_best = res.x
    rec['loss_final'] = res.fun
    rec['opt_attempts_count'] = 1 # We assume 1 attempt here for simplicity, or modify logic if retries used params
    rec['opt_best_attempt'] = 0
    
    # Error Estimation (Jacobian)
    v_err = jacobian_error_estimate(v_best, residual_vector, verbose=verbose)
    
    # Unpack best
    p_best = unpack_params(v_best, k)
    p_best['supersample_factor'] = supersample_factor
    
    # Update Rec
    for k_ in p_best:
        if isinstance(p_best[k_], np.ndarray):
             for i, val in enumerate(p_best[k_]):
                 rec[f"{k_}{i if k_ in ['a_m', 'phi_m'] else ''}_best"] = val
        else:
            rec[f"{k_}_best"] = p_best[k_]
            
    # Unpack Error
    # Need to match order of pack_params
    # n_sersic, R_sersic, amplitude, q, theta_ell, a_m[:], phi_m[:], x0, y0, background
    idx = 0
    rec['n_sersic_err'] = v_err[idx]; idx+=1
    rec['R_sersic_err'] = v_err[idx]; idx+=1
    rec['amplitude_err'] = v_err[idx]; idx+=1
    rec['q_err'] = v_err[idx]; idx+=1
    rec['theta_ell_err'] = v_err[idx]; idx+=1
    for i in range(k):
        rec[f"a_m{m[i]}_err"] = v_err[idx]; idx+=1
    for i in range(k):
        rec[f"phi_m{m[i]}_err"] = v_err[idx]; idx+=1
    rec['x0_err'] = v_err[idx]; idx+=1
    rec['y0_err'] = v_err[idx]; idx+=1
    rec['background_err'] = v_err[idx]; idx+=1

    # Final Plot (Detailed)
    if plot_final_contour:
        # Re-evaluate best model
        mod_ss = simulate_model_elliptical_multipole(
            X_ss, Y_ss,
            n_sersic=p_best['n_sersic'], R_sersic=p_best['R_sersic'], amplitude=p_best['amplitude'],
            q=p_best['q'], theta_ell=p_best['theta_ell'], m=m, a_m=p_best['a_m'], phi_m=p_best['phi_m'],
            x0=p_best['x0'], y0=p_best['y0'], background=0.0,
            psf=None
        )
        if supersample_factor > 1:
            mod_final = downsample(mod_ss, supersample_factor)
        else:
            mod_final = mod_ss
        mod_final = mod_final + p_best['background']
        
        res_map_final = residual_map_sigma(sci_bgsub, wht, mod_final, row_query['EXPTIME_SCI'], mask=mask)
        
        # Prepare Info for Plot
        # Param Dicts
        p_best_flat = {}
        p_unc_flat = {}
        # Flatten array params
        for keys in ['n_sersic', 'R_sersic', 'q', 'theta_ell', 'amplitude', 'background']:
             p_best_flat[keys] = p_best[keys]
             p_unc_flat[keys] = rec.get(f"{keys}_err", np.nan)
        for i, mi in enumerate(m):
             p_best_flat[f"a_m{mi}"] = p_best['a_m'][i]
             p_unc_flat[f"a_m{mi}"] = rec.get(f"a_m{mi}_err", np.nan)
        for i, mi in enumerate(m):
             p_best_flat[f"phi_m{mi}"] = p_best['phi_m'][i]
             p_unc_flat[f"phi_m{mi}"] = rec.get(f"phi_m{mi}_err", np.nan)
             
        # Truth Dict (if provided)
        p_true_flat = truth_row if truth_row else {}
        
        meta = f"Loss Init: {chi2_reduced_0:.2f}\nLoss Final: {res.fun:.2f}\nAttempts: 1\nSS Factor: {supersample_factor}\nTime: TBD"
        
        fig_d, axs_d = detailed_comparison_plot(
            np.ma.masked_array(sci_bgsub, mask=mask), mod_final, res_map_final,
            extent=extent,
            param_best=p_best_flat, param_unc=p_unc_flat, param_true=p_true_flat,
            meta_info_str=meta,
            scale='asinh'
        )
        
        out_final = os.path.join(data_dir, f"{seqid_str}-05-after_fitting.pdf")
        fig_d.savefig(out_final, bbox_inches='tight')
        plt.close(fig_d)
        
    return rec
