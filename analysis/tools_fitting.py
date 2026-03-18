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

def jacobian_error_estimate(v_best, residual_fn, bounds=None, rel_step=1e-6, abs_step=0.0, verbose=False):
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
    bounds : tuple of arrays (lo, hi), optional
        Lower and upper bounds for parameters. If provided, perturbed vectors
        will be clipped to these bounds to avoid invalid parameter values.
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

    # Extract bounds if provided
    if bounds is not None:
        lo, hi = bounds
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
    else:
        lo = np.full(P, -np.inf, dtype=float)
        hi = np.full(P, np.inf, dtype=float)

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
        
        # CRITICAL FIX: Clip to bounds to prevent invalid parameter values
        v_plus = np.clip(v_plus, lo, hi)
        v_minus = np.clip(v_minus, lo, hi)

        try:
            r_plus = np.asarray(residual_fn(v_plus), dtype=float).ravel()
            r_minus = np.asarray(residual_fn(v_minus), dtype=float).ravel()
        except (ValueError, RuntimeError) as e:
            # If evaluation fails even with clipping, use one-sided derivative
            if verbose:
                print(f"[jacobian_error_estimate] Failed to evaluate param {j}: {e}, using zero sensitivity")
            J[:, j] = 0.0
            continue

        # Sanity check on residual length
        if r_plus.size != N or r_minus.size != N:
            if verbose:
                print(f"[jacobian_error_estimate] Inconsistent residual size for param {j}, using zero sensitivity")
            J[:, j] = 0.0
            continue

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


def particle_swarm_optimization(loss_fn, bounds, n_particles, max_iter=200, w=0.5, c1=1.5, c2=1.5):
    """
    Lightweight numpy-based Particle Swarm Optimization.
    """
    lo, hi = bounds
    dim = len(lo)

    # Safe bounds for initialization to prevent OverflowError with np.inf
    lo_safe = np.array([max(x, -1e4) if np.isfinite(x) else -1e4 for x in lo], dtype=np.float64)
    hi_safe = np.array([min(x, 1e4) if np.isfinite(x) else 1e4 for x in hi], dtype=np.float64)

    # Initialize particles
    X = np.random.uniform(lo_safe, hi_safe, size=(n_particles, dim))
    V = np.zeros_like(X)
    
    # Personal bests
    P_best = X.copy()
    P_best_loss = np.array([loss_fn(p) for p in P_best])

    # Global best
    g_best_idx = np.argmin(P_best_loss)
    G_best = P_best[g_best_idx].copy()
    G_best_loss = P_best_loss[g_best_idx]

    # Dummy scipy minimize result class
    class OptimizeResult:
        pass
    res = OptimizeResult()
    res.x = G_best
    res.fun = G_best_loss
    res.success = True
    
    for it in range(max_iter):
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)
        
        # Update velocities
        V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (G_best - X)
        
        # Update positions
        X = X + V
        
        # Apply bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate loss
        # Doing this loop sequentially for single core safety as requested
        current_loss = np.array([loss_fn(p) for p in X])
        
        # Update personal bests
        improved = current_loss < P_best_loss
        P_best[improved] = X[improved]
        P_best_loss[improved] = current_loss[improved]
        
        # Update global best
        min_loss_idx = np.argmin(P_best_loss)
        if P_best_loss[min_loss_idx] < G_best_loss:
            G_best = P_best[min_loss_idx].copy()
            G_best_loss = P_best_loss[min_loss_idx]
            
    res.x = G_best
    res.fun = G_best_loss
    return res

# ---------------------------
# per-target workflow
# ---------------------------
def configured_optimizer(loss, v0, lo, hi, opt_method, n_particles_factor=4):
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
    elif opt_method == 'PSO':
        n_particles = n_particles_factor * len(v0)
        opt = particle_swarm_optimization(loss, (lo, hi), n_particles=n_particles)
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


def process_one_target_optimize(
        row_query, 
        data_dir, 
        row_sep=None,
        sci=None, wht=None, mask=None, segmap=None, psf=None, # New arguments
        m=[3, 4], 
        opt_method='SLSQP', # or 'Newton-CG' (requires jacobian), 'BFGS', 'L-BFGS-B', 'Nelder-Mead'
        PIX_SCALE=0.03,
        plot_initial_contour=False, 
        plot_final_contour=True,
        fit_model=True, # Restored argument
        verbose=True, 
        target_loss=1.5,
        supersample_factor=1,
        truth_row=None,
        plot_name=None,
        initial_guess=None, # Added argument
        enable_PSO=False,
        pso_only=False,
        n_particles_factor=4 # Number of PSO particles = factor * n_params
    ):
    """
    optimize the target galaxy with Sersic + Multipole model.
    """
    
    # If explicit data is passed (cropped), use it.
    if sci is not None:
        # We are using pre-loaded/cropped data
        sci_bgsub = sci
        # wht, mask, segmap are already passed as arguments
    else:
        # Original logic: load from file
        seqid = row['id']
        f_sci = os.path.join(target_dir, f"{seqid}-SCI.fits")
        f_wht = os.path.join(target_dir, f"{seqid}-WHT.fits")
        
        # Load data
        # Note: We use return_orientat=False/center=False to avoid bugs in mocks
        if "mock" in target_dir: # Heuristic
             sci_bgsub, wht = load_fits(f_sci, f_wht, return_orientat=False, return_center=False)
             orientat = 0.0
        else:
             # Regular COSMOS data
             sci_bgsub, wht, orientat, center_xy = load_fits(f_sci, f_wht, return_orientat=True, return_center=True)
        
        # We need mask and segmap if not provided
        if segmap is None or mask is None:
             # For now, default to None or throw error if needed?
             # The usage in run_mock_fitting passes mask/segmap.
             # If called from legacy code, we might need to recreate mask.
             pass # This needs to be handled by the caller or default mask/segmap generation

    if wht is None:
        raise ValueError("Weight map (wht) is required.")

    # Prepare data for fitting
    # In mocks, row['filename'] might be best, but we'll try ID too
    if hasattr(row_query, 'name') and row_query.name is not None:
        seqid_str = str(row_query.name)
    elif row_query is not None and 'sequentialid' in row_query:
        seqid_str = str(int(row_query['sequentialid']))
    elif truth_row is not None:
        # Try to get ID from truth_row
        if 'id' in truth_row:
             seqid_str = str(int(truth_row['id']))
        elif 'seqid' in truth_row:
             seqid_str = str(int(truth_row['seqid']))
        else:
             seqid_str = "mock"
    else:
        # Fallback to the target string from filename arg if passed via sep or args.
        # But here we only have row_sep
        seqid_str = str(int(row_sep['seqid']))

    rec = dict(sequentialid=seqid_str) 
    
    # Load Data (HDF5)
    # The cropped file from preprocess_directory has format '{base}-cropped.hdf5'
    filename_hdf5 = os.path.join(data_dir, f"{seqid_str}-cropped.hdf5")
    if not os.path.exists(filename_hdf5):
        # Trying just .hdf5
        filename_hdf5_alt = os.path.join(data_dir, f"{seqid_str}.hdf5")
        if os.path.exists(filename_hdf5_alt): 
            filename_hdf5 = filename_hdf5_alt
        else:
            raise FileNotFoundError(f"HDF5 file not found: {filename_hdf5}")
            
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
    theta_ell = row_sep['theta']
    
    # Initial Guesses
    if row_query is not None:
        R_sersic0 = row_query.get('r_gim2d', default=np.nan)
    else:
        R_sersic0 = np.nan 
    if np.isnan(R_sersic0):
        R_sersic0 = row_sep['R50'] * PIX_SCALE
        rec.update(initial_R_sersic0_from='SEP R50')
    else:
        rec.update(initial_R_sersic0_from='r_gim2d')

    if row_query is not None:
        n_sersic0 = row_query.get('sersic_n_gim2d', default=np.nan)
    else:
        n_sersic0 = np.nan

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

    # ---------------------------
    # Initial Plot Generation (2x3 Layout)
    # ---------------------------
    model0 = I0 * model_unit + bg0
    n_param_model = 5 + 3*len(m) 
    exptime = None
    if row_query is not None and 'EXPTIME_SCI' in row_query:
        try:
            val = float(row_query['EXPTIME_SCI'])
            if np.isfinite(val):
                exptime = val
        except:
            pass
    elif truth_row is not None and 'EXPTIME_SCI' in truth_row:
        try:
            val = float(truth_row['EXPTIME_SCI'])
            if np.isfinite(val):
                exptime = val
        except:
            pass
    if exptime is None:
        raise ValueError("Could not determine exposure time from row_query or truth_row.")

    chi2_reduced_0 = reduced_chi_squared(sci_bgsub, wht, model0, n_param_model, exptime, mask=mask)
    residual_map_0 = residual_map_sigma(sci_bgsub, wht, model0, exptime, mask=mask)

    # Prepare parameter dictionaries for initial plot
    p0_flat = {}
    for key in ['n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell', 'x0', 'y0', 'background']:
        p0_flat[key] = p0_elliptical_multipole[key]
    for i, mi in enumerate(m):
        p0_flat[f"a_m{mi}"] = p0_elliptical_multipole['a_m'][i]
    for i, mi in enumerate(m):
        p0_flat[f"phi_m{mi}"] = p0_elliptical_multipole['phi_m'][i]
    
    # Truth parameters (if provided)
    p_true_flat_0 = {k.replace('_true', ''): v for k, v in truth_row.items()} if truth_row else {}
    
    # Meta info for initial plot
    meta_0 = f"Loss Init: {chi2_reduced_0:.2f}\nSS Factor: {supersample_factor}"
    
    # Use detailed_comparison_plot for consistency
    fig1, axs1 = detailed_comparison_plot(
        np.ma.masked_array(sci_bgsub, mask=mask), model0, residual_map_0,
        extent=extent,
        param_best=p0_flat, param_unc=None,  # No uncertainties for initial
        param_true=p_true_flat_0,
        meta_info_str=meta_0,
        residual_vmin=-5, residual_vmax=5,
        scale='asinh'
    )
    
    # Overlay contour if requested
    if plot_initial_contour:
        x1, y1 = Elliptical_Multipole_Profile_1D(
            r0=R_sersic0, q=q, theta_ell=theta_ell, m=m, a_m=a_m0,
            phi_m=phi_m0, x0=x0_init, y0=y0_init, return_type='xy',
            include_end=True, n_points=300
        )
        axs1[0,1].plot(x1, y1, color='k', lw=1.0)  # On model panel

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
        wht, sci_bgsub, exptime,
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
        
        res = reduced_chi_squared(sci_bgsub, wht, mod, n_param_model, exptime, mask=mask)
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
        
        return diff[valid] / sigma_tot[valid]
    # Run Optimization Loop
    # Strategy sequence: 
    # 1. opt_method (default SLSQP)
    # 2. L-BFGS-B
    # 3. trust-constr
    strategies = []
    if pso_only:
        strategies.append('PSO')
    else:
        if opt_method not in strategies: strategies.append(opt_method)
        if 'L-BFGS-B' not in strategies: strategies.append('L-BFGS-B')
        if enable_PSO: strategies.append('PSO')
        # if 'trust-constr' not in strategies: strategies.append('trust-constr')
    
    max_strategies = len(strategies)
    best_res = None
    best_loss = np.inf
    final_v_best = None
    final_attempt_count = 0
    best_strategy_name = None
    
    # Define bounds once (lo, hi are already defined)
    idx_phi_start = 5 + k
    phi_lo = lo[idx_phi_start : idx_phi_start+k]
    phi_hi = hi[idx_phi_start : idx_phi_start+k]
    
    # Tolerance for boundary check
    tol = 1e-3

    found_satisfactory_solution = False

    # Independent tracking for non-PSO and PSO (to save both)
    res_non_pso = None
    loss_non_pso = np.inf
    time_non_pso = np.nan

    res_pso = None
    loss_pso = np.inf
    time_pso = np.nan

    v_start = pack_params(p0_elliptical_multipole.copy(),k)
    
    for attempt_idx, current_method in enumerate(strategies):
        if verbose:
            print(f"Optimization Strategy {attempt_idx+1}/{max_strategies}: {current_method}")
        
        # Use v_start from stages as initial guess
        # v_start is modified in place during stages, so it carries over
        # For each strategy, we can restart from the BEST stage result (v_start)
        
        # Sub-loop for boundary retry (max 1 retry per strategy)
        boundary_retry_count = 0
        max_boundary_retries = 1
        
        # Make a local copy for this strategy loop
        v_run = v_start.copy() 
        
        while boundary_retry_count <= max_boundary_retries:
            print(f"fitting with {current_method}... ", end="", flush=True)
            start_time = time.time()
            if current_method == 'PSO':
                res = configured_optimizer(loss, v_run, lo, hi, current_method, n_particles_factor=n_particles_factor)
            else:
                res = configured_optimizer(loss, v_run, lo, hi, current_method)
            
            elapsed_time = time.time() - start_time
            print(f"done in {elapsed_time:.1f}s")
            
            # Check result
            current_loss = res.fun
            
            # Track best global result
            if current_loss < best_loss:
                best_loss = current_loss
                best_res = res
                final_v_best = res.x
                final_attempt_count = attempt_idx + 1 # 1-based index of strategy used
                best_strategy_name = current_method

            # Independent tracking
            if current_method == 'PSO':
                if current_loss < loss_pso:
                    loss_pso = current_loss
                    res_pso = res
                    time_pso = elapsed_time
            else:
                if current_loss < loss_non_pso:
                    loss_non_pso = current_loss
                    res_non_pso = res
                    time_non_pso = elapsed_time

            # 1. Check Target Loss
            if current_loss <= target_loss:
                if verbose:
                    print(f"  Target loss met ({current_loss:.3f} <= {target_loss}). Stopping.")
                found_satisfactory_solution = True
                break # Break boundary loop
            
            # 2. Check Boundaries if loss not met
            # Unpack best params from this run
            p_current = unpack_params(res.x, k)
            phi_current = p_current['phi_m']
            a_current = p_current['a_m']
            
            hit_boundary = False
            new_p = p_current.copy()
            
            for i in range(k):
                # Check lower bound
                if abs(phi_current[i] - phi_lo[i]) < tol:
                    if verbose:
                        print(f"  Mode m={m[i]}: phi hit lower bound {phi_lo[i]:.3f}.")
                    # Flip strategy: a -> -a, phi -> upper bound
                    new_p['a_m'][i] = -a_current[i]
                    new_p['phi_m'][i] = phi_hi[i] - tol*2
                    hit_boundary = True
                    
                # Check upper bound
                elif abs(phi_current[i] - phi_hi[i]) < tol:
                    if verbose:
                        print(f"  Mode m={m[i]}: phi hit upper bound {phi_hi[i]:.3f}.")
                    # Flip strategy: a -> -a, phi -> lower bound
                    new_p['a_m'][i] = -a_current[i]
                    new_p['phi_m'][i] = phi_lo[i] + tol*2
                    hit_boundary = True
            
            if hit_boundary:
                if boundary_retry_count < max_boundary_retries:
                    if verbose:
                        print("  Boundary hit detected. Flipping parameters and retrying (Same Strategy).")
                    # Update v_run for the retry
                    v_run = pack_params(new_p, k)
                    boundary_retry_count += 1
                    continue # Run loop again with new v_run
                else:
                    if verbose:
                        print("  Boundary hit detected, but max boundary retries reached.")
                    break # Break boundary loop, move to next strategy check
                if verbose:
                    print(f"  No boundary hit, but loss {current_loss:.3f} > {target_loss}.")
                break # Break boundary loop, move to next strategy check

            boundary_retry_count += 1

        if found_satisfactory_solution:
            break # Break strategy loop

    # Handle Result (Use best found)
    v_best = final_v_best
    res = best_res # Ensure we have the res object correspond to v_best
    rec['loss_final'] = best_loss
    rec['opt_attempts_count'] = final_attempt_count
    rec['opt_best_attempt'] = final_attempt_count - 1 # 0-indexed best attempt
    rec['opt_best_strategy'] = best_strategy_name
    # Global variables for standard logic
    p_best = unpack_params(v_best, k)
    p_best['supersample_factor'] = supersample_factor
    
    # Error Estimation and Param Unpacking natively mapped specifically for each run outcome!
    if res_non_pso is not None:
        rec['loss_non_pso'] = loss_non_pso
        rec['opt_time_non_pso'] = time_non_pso
        v_err_non_pso = jacobian_error_estimate(res_non_pso.x, residual_vector, bounds=(lo, hi), verbose=verbose)
        p_best_non_pso = unpack_params(res_non_pso.x, k)
        
        # Populate *_best
        for k_ in p_best_non_pso:
            if isinstance(p_best_non_pso[k_], np.ndarray):
                 for i, val in enumerate(p_best_non_pso[k_]):
                     rec[f"{k_}{m[i] if k_ in ['a_m', 'phi_m'] else i}_best"] = val
            else:
                rec[f"{k_}_best"] = p_best_non_pso[k_]
                
        # Unpack Error
        idx = 0
        rec['n_sersic_err'] = v_err_non_pso[idx]; idx+=1
        rec['R_sersic_err'] = v_err_non_pso[idx]; idx+=1
        rec['amplitude_err'] = v_err_non_pso[idx]; idx+=1
        rec['q_err'] = v_err_non_pso[idx]; idx+=1
        rec['theta_ell_err'] = v_err_non_pso[idx]; idx+=1
        for i in range(k): rec[f"a_m{m[i]}_err"] = v_err_non_pso[idx]; idx+=1
        for i in range(k): rec[f"phi_m{m[i]}_err"] = v_err_non_pso[idx]; idx+=1
        rec['x0_err'] = v_err_non_pso[idx]; idx+=1
        rec['y0_err'] = v_err_non_pso[idx]; idx+=1
        rec['background_err'] = v_err_non_pso[idx]; idx+=1

    if res_pso is not None:
        rec['loss_pso'] = loss_pso
        rec['opt_time_pso'] = time_pso
        v_err_pso = jacobian_error_estimate(res_pso.x, residual_vector, bounds=(lo, hi), verbose=verbose)
        p_best_pso = unpack_params(res_pso.x, k)
        
        # Populate *_pso_best
        for k_ in p_best_pso:
            if isinstance(p_best_pso[k_], np.ndarray):
                 for i, val in enumerate(p_best_pso[k_]):
                     rec[f"{k_}{m[i] if k_ in ['a_m', 'phi_m'] else i}_pso_best"] = val
            else:
                rec[f"{k_}_pso_best"] = p_best_pso[k_]
                
        # Unpack Error
        idx = 0
        rec['n_sersic_pso_err'] = v_err_pso[idx]; idx+=1
        rec['R_sersic_pso_err'] = v_err_pso[idx]; idx+=1
        rec['amplitude_pso_err'] = v_err_pso[idx]; idx+=1
        rec['q_pso_err'] = v_err_pso[idx]; idx+=1
        rec['theta_ell_pso_err'] = v_err_pso[idx]; idx+=1
        for i in range(k): rec[f"a_m{m[i]}_pso_err"] = v_err_pso[idx]; idx+=1
        for i in range(k): rec[f"phi_m{m[i]}_pso_err"] = v_err_pso[idx]; idx+=1
        rec['x0_pso_err'] = v_err_pso[idx]; idx+=1
        rec['y0_pso_err'] = v_err_pso[idx]; idx+=1
        rec['background_pso_err'] = v_err_pso[idx]; idx+=1

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
        
        res_map_final = residual_map_sigma(sci_bgsub, wht, mod_final, exptime, mask=mask)
        
        # Prepare Info for Plot
        # Param Dicts
        p_best_flat = {}
        p_unc_flat = {}
        # Flatten array params - ensure ALL parameters are included
        for keys in ['n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell', 'x0', 'y0', 'background']:
             p_best_flat[keys] = p_best[keys]
             p_unc_flat[keys] = rec.get(f"{keys}_err", np.nan)
        for i, mi in enumerate(m):
             p_best_flat[f"a_m{mi}"] = p_best['a_m'][i]
             p_unc_flat[f"a_m{mi}"] = rec.get(f"a_m{mi}_err", np.nan)
        for i, mi in enumerate(m):
             p_best_flat[f"phi_m{mi}"] = p_best['phi_m'][i]
             p_unc_flat[f"phi_m{mi}"] = rec.get(f"phi_m{mi}_err", np.nan)
             
        # Truth Dict (if provided)
        p_true_flat = {k.replace('_true', ''): v for k, v in truth_row.items()} if truth_row else {}
        
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

# ---------------------------
# MCMC Workflow
# ---------------------------
def process_one_target_mcmc(
    row_query,
    data_dir,
    row_sep=None,
    opt_row=None,
    m=[3, 4],
    PIX_SCALE=0.03,
    fit_params=None,
    fix_params=None,
    use_analytic_amplitude=True,
    mcmc_config=None,
    supersample_factor=1,
    continue_mcmc=False,
    restart_and_overwrite_mcmc=False,
    debug=False,
    truth_row=None,
):
    """
    Run MCMC sampling for one target starting from optimization results.
    """
    if mcmc_config is None:
        mcmc_config = {
            "n_walkers": 32,
            "n_steps": 4000,
            "burnin_fraction": 0.3,
            "init_scale": 1e-4,
            "random_seed": 42
        }

    seqid_str = str(int(opt_row['sequentialid']))
    rec = dict(sequentialid=seqid_str)

    # Load Data (HDF5)
    filename_hdf5 = os.path.join(data_dir, f"{seqid_str}-cropped.hdf5")
    with h5py.File(filename_hdf5, "r") as data_file:
        sci_bgsub = np.array(data_file['sci_bgsub_crop'])
        wht       = np.array(data_file['wht_crop'])
        mask      = np.array(data_file['mask_crop'])
        seg       = np.array(data_file['segmap_crop'])

    X, Y, extent = build_arcsec_grid(sci_bgsub.shape, pixscale=PIX_SCALE)
    
    # Get supersample factor if present in opt_row
    ss = supersample_factor
    if ss > 1:
        shape_ss = (sci_bgsub.shape[0] * ss, sci_bgsub.shape[1] * ss)
        pixscale_ss = PIX_SCALE / ss
        X_ss, Y_ss, _ = build_arcsec_grid(shape_ss, pixscale=pixscale_ss)
    else:
        X_ss, Y_ss = X, Y

    val = row_query.get('EXPTIME_SCI', None)
    if val is None or not np.isfinite(float(val)):
        raise ValueError("Could not determine exposure time.")
    exptime = float(val)

    sigma_tot = np.sqrt(sigma_total_squared(
        wht, sci_bgsub, exptime,
        poisson_threshold_n=3., mask=mask, verbose=False
    ))

    # Initialize from best fit
    k = len(m)
    p_opt = {}
    for key in ['n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell', 'x0', 'y0', 'background']:
        p_opt[key] = float(opt_row.get(f"{key}_best", 0.0))
    a_m_opt = []
    phi_m_opt = []
    for mi in m:
        a_m_opt.append(float(opt_row.get(f"a_m{mi}_best", 0.0)))
        phi_m_opt.append(float(opt_row.get(f"phi_m{mi}_best", 0.0)))
    p_opt['a_m'] = np.array(a_m_opt)
    p_opt['phi_m'] = np.array(phi_m_opt)

    img_half_extent = 0.5 * max(extent[1] - extent[0], extent[3] - extent[2])

    v_opt = pack_params(p_opt, k)
    lo, hi = default_bounds(m, img_half_extent)

    # Log Probability Function
    def log_prob(vec):
        # Bound check
        if np.any(vec < lo) or np.any(vec > hi):
            return -np.inf
            
        pp = unpack_params(vec, k)
        if pp['R_sersic'] <= 0 or pp['n_sersic'] <= 0 or pp['q'] <= 0:
            return -np.inf

        mod_ss = simulate_model_elliptical_multipole(
            X_ss, Y_ss,
            n_sersic=pp['n_sersic'], R_sersic=pp['R_sersic'], amplitude=pp['amplitude'],
            q=pp['q'], theta_ell=pp['theta_ell'], m=m, a_m=pp['a_m'], phi_m=pp['phi_m'],
            x0=pp['x0'], y0=pp['y0'], background=0.0
        )
        if ss > 1:
            mod = downsample(mod_ss, ss)
        else:
            mod = mod_ss
        mod = mod + pp['background']

        diff = sci_bgsub - mod
        if mask is not None:
            valid = (~mask) & np.isfinite(diff) & np.isfinite(sigma_tot)
        else:
            valid = np.isfinite(diff) & np.isfinite(sigma_tot)

        res = (diff[valid] / sigma_tot[valid])
        chi2 = np.sum(res**2)
        return -0.5 * chi2

    dim = len(v_opt)
    n_walkers = mcmc_config['n_walkers']
    n_steps = mcmc_config['n_steps']

    np.random.seed(mcmc_config.get('random_seed', 42))
    
    # Init walkers in a tiny ball around the optimization best fit
    init_scale = mcmc_config['init_scale']
    pos = v_opt + init_scale * np.random.randn(n_walkers, dim)
    
    # Ensure they are within bounds
    for i in range(n_walkers):
        pos[i] = np.clip(pos[i], lo + 1e-6, hi - 1e-6)

    import emcee
    backend_file = os.path.join(data_dir, f"{seqid_str}-mcmc_backend.h5")
    backend = emcee.backends.HDFBackend(backend_file)
    
    if restart_and_overwrite_mcmc:
        backend.reset(n_walkers, dim)
    elif not continue_mcmc:
        backend.reset(n_walkers, dim)
    else:
        try:
            if backend.iteration > 0:
                n_walkers_backend, dim_backend = backend.shape
                if dim != dim_backend:
                    raise ValueError(f"incompatible input dimensions ({dim_backend}, {dim})")
                if n_walkers != n_walkers_backend:
                    print(f"  Adjusting n_walkers from {n_walkers} to {n_walkers_backend} to match backend.")
                    n_walkers = n_walkers_backend
        except AttributeError:
            pass # shape might not be there if backend is empty
        
    sampler = emcee.EnsembleSampler(n_walkers, dim, log_prob, backend=backend)
    
    t0 = time.time()
    
    # Auto-convergence loop
    max_steps = mcmc_config.get('n_steps', 5000)
    check_interval = 200
    
    # Track autocorrelation time to test for convergence
    old_tau = np.inf
    converged = False
    
    # Only supply initial pos if starting fresh
    if backend.iteration == 0:
        initial_state = pos
        print(f"  Starting fresh MCMC for {seqid_str} with {n_walkers} walkers...")
    else:
        initial_state = sampler.get_last_sample()
        print(f"  Resuming MCMC for {seqid_str} from step {backend.iteration}...")
        
    try:
        if max_steps > 0:
            for sample in sampler.sample(initial_state, iterations=max_steps, progress=True):
                if sampler.iteration % check_interval == 0:
                    try:
                        # Compute tau with tolerance=0 so it doesn't crash if chain is too short
                        tau = sampler.get_autocorr_time(tol=0)
                        
                        # Check convergence
                        min_independent_samples = 20 # at least 20 independent samples
                        converged = np.all(tau * min_independent_samples < sampler.iteration)
                        stable_fraction = 0.03 # 3 percent of change is okay
                        converged &= np.all(np.abs(old_tau - tau) / tau < stable_fraction)
                        if converged:
                            print(f"  Convergence reached at step {sampler.iteration}!")
                            break
                        old_tau = tau
                    except emcee.autocorr.AutocorrError:
                        # Not enough steps to estimate tau reliably
                        pass
        else:
            print("  Skipping additional sampling steps (max_steps=0). Extracting backend.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  MCMC sampling aborted/failed (Error: {e}). Proceeding to extract from existing backend...")
            
    mcmc_time = time.time() - t0
    if not converged:
        print(f"  MCMC stopped at max {sampler.iteration} steps without fully satisfying convergence criteria.")

    # Determine Burn-in using Auto-correlation time
    burnin = 0
    thin = 1
    try:
        tau = sampler.get_autocorr_time(tol=0)
        burnin = int(2 * np.max(tau))
        thin = int(0.5 * np.min(tau))
        if thin == 0: thin = 1
        if burnin >= sampler.iteration: 
            burnin = int(mcmc_config.get('burnin_fraction', 0.3) * sampler.iteration)
    except Exception as e:
        print(f"  Autocorrelation warning: {e}. Using fixed fraction.")
    try:
        flat_samples = backend.get_chain(discard=burnin, thin=thin, flat=True)
    except AttributeError:
        print("  Error: MCMC backend is empty. Cannot extract parameters.")
        raise RuntimeError("Empty MCMC backend.")
    
    # Get parameter names
    param_names = ['n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell']
    for mi in m: param_names.append(f"a_m{mi}")
    for mi in m: param_names.append(f"phi_m{mi}")
    param_names.extend(['x0', 'y0', 'background'])

    # Debug Plots
    if debug:
        import corner
        # Trace Plot
        # Get chain with unflattened shape (n_steps, n_walkers, dim)
        chain = backend.get_chain()
        fig_trace, axes_trace = plt.subplots(dim, 1, figsize=(10, 1.5 * dim), sharex=True)
        max_tau_str = f"{np.max(tau):.1f}" if 'tau' in locals() and hasattr(tau, '__iter__') else "unknown"
        fig_trace.suptitle(f"MCMC Trace (max tau: {max_tau_str}, burn-in: {burnin}, fully-converged: {converged})", fontsize=14)
        for i in range(dim):
            ax = axes_trace[i]
            ax.plot(chain[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, sampler.iteration)
            ax.set_ylabel(param_names[i])
            ax.axvline(burnin, color="red", linestyle="--", lw=2, label="burn-in")
        axes_trace[-1].set_xlabel("step number")
        axes_trace[0].legend()
        fig_trace.tight_layout()
        trace_file = os.path.join(data_dir, f"{seqid_str}-10-mcmc_trace.pdf")
        fig_trace.savefig(trace_file)
        plt.close(fig_trace)
        
        # Corner Plot
        fig_corner = corner.corner(
            flat_samples, labels=param_names, truths=v_opt,
            quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12}
        )
        corner_file = os.path.join(data_dir, f"{seqid_str}-11-mcmc_corner.pdf")
        fig_corner.savefig(corner_file)
        plt.close(fig_corner)

    # Compute Results
    mcmc_best = np.percentile(flat_samples, 50, axis=0) # Median
    mcmc_16 = np.percentile(flat_samples, 16, axis=0)
    mcmc_84 = np.percentile(flat_samples, 84, axis=0)
    mcmc_err = (mcmc_84 - mcmc_16) / 2.0

    p_mcmc_best = unpack_params(mcmc_best, k)
    
    # Store Best & Err
    rec['mcmc_time'] = mcmc_time
    rec['mcmc_n_steps_used'] = n_steps
    rec['mcmc_burnin'] = burnin
    
    for key in p_mcmc_best:
        if isinstance(p_mcmc_best[key], np.ndarray):
            for i, val in enumerate(p_mcmc_best[key]):
                rec[f"{key}{m[i]}_mcmc_best"] = val
        else:
            rec[f"{key}_mcmc_best"] = p_mcmc_best[key]
            
    # Unpack Error
    idx = 0
    rec['n_sersic_mcmc_err'] = mcmc_err[idx]; idx+=1
    rec['R_sersic_mcmc_err'] = mcmc_err[idx]; idx+=1
    rec['amplitude_mcmc_err'] = mcmc_err[idx]; idx+=1
    rec['q_mcmc_err'] = mcmc_err[idx]; idx+=1
    rec['theta_ell_mcmc_err'] = mcmc_err[idx]; idx+=1
    for i in range(k): rec[f"a_m{m[i]}_mcmc_err"] = mcmc_err[idx]; idx+=1
    for i in range(k): rec[f"phi_m{m[i]}_mcmc_err"] = mcmc_err[idx]; idx+=1
    rec['x0_mcmc_err'] = mcmc_err[idx]; idx+=1
    rec['y0_mcmc_err'] = mcmc_err[idx]; idx+=1
    rec['background_mcmc_err'] = mcmc_err[idx]; idx+=1

    # Optional final detailed plot using MCMC results
    mod_ss = simulate_model_elliptical_multipole(
        X_ss, Y_ss,
        n_sersic=p_mcmc_best['n_sersic'], R_sersic=p_mcmc_best['R_sersic'], amplitude=p_mcmc_best['amplitude'],
        q=p_mcmc_best['q'], theta_ell=p_mcmc_best['theta_ell'], m=m, a_m=p_mcmc_best['a_m'], phi_m=p_mcmc_best['phi_m'],
        x0=p_mcmc_best['x0'], y0=p_mcmc_best['y0'], background=0.0
    )
    if ss > 1:
        mod_final = downsample(mod_ss, ss)
    else:
        mod_final = mod_ss
    mod_final = mod_final + p_mcmc_best['background']
        
    res_map_final = residual_map_sigma(sci_bgsub, wht, mod_final, exptime, mask=mask)
    chi2_final = reduced_chi_squared(sci_bgsub, wht, mod_final, dim, exptime, mask=mask)
    
    rec['loss_mcmc_final'] = chi2_final
    
    # Extract MCMC loss range
    try:
        log_probs = backend.get_log_prob(discard=burnin, thin=thin, flat=True)
        # log_prob = -0.5 * chi2 + const 
        # (Assuming uniform priors where valid, this is mostly the likelihood)
        # Better yet, just compute exact chi2: chi2 = -2 * log_prob
        if mask is not None:
            valid_pixels = np.sum(~mask)
        else:
            valid_pixels = sci_bgsub.size
        dof = valid_pixels - dim
        chi2_samples = -2.0 * log_probs
        reduced_chi2_samples = chi2_samples / dof
        
        loss_mcmc_16 = np.percentile(reduced_chi2_samples, 16)
        loss_mcmc_84 = np.percentile(reduced_chi2_samples, 84)
        
        rec['loss_mcmc_16'] = loss_mcmc_16
        rec['loss_mcmc_84'] = loss_mcmc_84
    except Exception as e:
        print(f"  Warning: could not compute MCMC loss range: {e}")
        rec['loss_mcmc_16'] = np.nan
        rec['loss_mcmc_84'] = np.nan
    
    p_best_flat = {}
    p_unc_flat = {}
    for keys in ['n_sersic', 'R_sersic', 'amplitude', 'q', 'theta_ell', 'x0', 'y0', 'background']:
        p_best_flat[keys] = p_mcmc_best[keys]
        p_unc_flat[keys] = rec[f"{keys}_mcmc_err"]
    for i, mi in enumerate(m):
        p_best_flat[f"a_m{mi}"] = p_mcmc_best['a_m'][i]
        p_unc_flat[f"a_m{mi}"] = rec[f"a_m{mi}_mcmc_err"]
    for i, mi in enumerate(m):
        p_best_flat[f"phi_m{mi}"] = p_mcmc_best['phi_m'][i]
        p_unc_flat[f"phi_m{mi}"] = rec[f"phi_m{mi}_mcmc_err"]
             
    meta = f"Loss MCMC: {chi2_final:.2f}\nSS Factor: {ss}\nMCMC Time: {mcmc_time:.1f}s"
        
    p_true_flat = {k.replace('_true', ''): v for k, v in truth_row.items()} if truth_row else {}
        
    fig_d, axs_d = detailed_comparison_plot(
        np.ma.masked_array(sci_bgsub, mask=mask), mod_final, res_map_final,
        extent=extent,
        param_best=p_best_flat, param_unc=p_unc_flat, param_true=p_true_flat,
        meta_info_str=meta, scale='asinh'
    )
        
    out_final = os.path.join(data_dir, f"{seqid_str}-12-after_mcmc.pdf")
    fig_d.savefig(out_final, bbox_inches='tight')
    plt.close(fig_d)

    return rec
