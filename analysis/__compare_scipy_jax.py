#!/usr/bin/env python3
"""
Benchmark JAX Adam vs SciPy (SLSQP, L-BFGS-B) on simple 2D image fitting.

Creates a mock observation from a chosen distribution (exponential or sersic),
optionally PSF-convolves it, adds background + Gaussian noise + shot noise,
then fits with 3 optimizers across 4 cases and logs loss/time per iteration.
"""

from __future__ import annotations

import os
import time
import json
import math
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import warnings
warnings.simplefilter('error')

try:
    import jax
    import jax.numpy as jnp
except ImportError as e:
    raise ImportError("This script requires jax. Install jax (CPU) first.") from e

# -----------------------------
# Hyperparameters
# -----------------------------
LOSS_SCALE = 'linear'  # 'linear' or 'log'


# -----------------------------
# Utilities: IO / directories
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_csv_header(path: str, header: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

def append_csv_row(path: str, row: List[object]) -> None:
    def fmt(x):
        if isinstance(x, float):
            # high precision for timing/loss
            return f"{x:.12g}"
        return str(x)
    with open(path, "a", encoding="utf-8") as f:
        f.write(",".join(fmt(x) for x in row) + "\n")

def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# -----------------------------
# Sigma clipping (no astropy)
# -----------------------------

def sigma_clip_bg(image: np.ndarray, sigma: float = 3.0, maxiters: int = 10) -> Tuple[float, float]:
    """
    Simple robust sigma clip based on median + MAD -> std.
    Returns (bg_level_est, bg_sigma_est).
    """
    data = image.ravel().astype(np.float32)
    mask = np.isfinite(data)
    data = data[mask]
    if data.size == 0:
        return 0.0, 1.0

    for _ in range(maxiters):
        med = np.median(data)
        mad = np.median(np.abs(data - med))
        # MAD->sigma for normal
        robust_sigma = 1.4826 * mad if mad > 0 else np.std(data)
        if robust_sigma <= 0:
            robust_sigma = max(np.std(data), 1e-6)

        lo = med - sigma * robust_sigma
        hi = med + sigma * robust_sigma
        new = data[(data >= lo) & (data <= hi)]
        if new.size == data.size:
            break
        if new.size < 10:
            break
        data = new

    bg_level = float(np.median(data))
    mad = np.median(np.abs(data - bg_level))
    bg_sigma = float(1.4826 * mad) if mad > 0 else float(np.std(data))
    bg_sigma = max(bg_sigma, 1e-6)
    return bg_level, bg_sigma


# -----------------------------
# PSF builders + convolution
# -----------------------------

def gaussian_psf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    sigma_psf: float,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Build a normalized 2D Gaussian PSF kernel on a cropped grid that extends to +/- n_sigma*sigma.
    Uses the same pixel spacing as x and y.
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    # choose symmetric odd size covering +/- n_sigma*sigma
    half_wx = int(np.ceil((n_sigma * sigma_psf) / dx))
    half_wy = int(np.ceil((n_sigma * sigma_psf) / dy))
    kx = np.arange(-half_wx, half_wx + 1) * dx
    ky = np.arange(-half_wy, half_wy + 1) * dy
    KX, KY = np.meshgrid(kx, ky)
    ker = np.exp(-(KX**2 + KY**2) / (2.0 * sigma_psf**2))
    s = float(np.sum(ker))
    if s <= 0:
        raise ValueError("PSF kernel sum <= 0")
    ker = ker / s
    return ker.astype(np.float32)


def fft_convolve2d_same_numpy(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Linear convolution (no wraparound) using FFT, returning same shape as image.
    Works even if kernel is larger than image.
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    out_shape = (H + Kh - 1, W + Kw - 1)

    # FFT of padded arrays
    F_img = np.fft.fftn(image, out_shape)
    F_ker = np.fft.fftn(kernel, out_shape)
    full = np.fft.ifftn(F_img * F_ker).real

    # Crop central region aligned with kernel center
    start_y = Kh // 2
    start_x = Kw // 2
    return full[start_y:start_y + H, start_x:start_x + W].astype(np.float32)


def fft_convolve2d_same_jax(image: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """
    JAX version of linear FFT convolution returning same shape as image.
    """
    H, W = image.shape
    Kh, Kw = kernel.shape
    out_shape = (H + Kh - 1, W + Kw - 1)

    F_img = jnp.fft.fftn(image, out_shape)
    F_ker = jnp.fft.fftn(kernel, out_shape)
    full = jnp.fft.ifftn(F_img * F_ker).real

    start_y = Kh // 2
    start_x = Kw // 2
    return full[start_y:start_y + H, start_x:start_x + W]


# -----------------------------
# Model image generators
# -----------------------------

def model_exponential_numpy(X: np.ndarray, Y: np.ndarray, amp: float, x0: float, y0: float, width: float) -> np.ndarray:
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    return amp * np.exp(-r / width)

def b_n_approx(n: float) -> float:
    # Good approximation for n ≳ 0.36; for our synthetic test it's fine.
    return 2.0 * n - 1.0/3.0 + 4.0/(405.0*n) + 46.0/(25515.0*n*n)

def model_sersic_numpy(X: np.ndarray, Y: np.ndarray, amp_central: float, x0: float, y0: float, n_sersic: float, R_e: float) -> np.ndarray:
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    bn = b_n_approx(n_sersic)
    # standard form uses I_e at R_e; map central amplitude to I_e:
    I_e = amp_central * np.exp(-bn)
    # I(r) = I_e * exp(-bn * ((r/R_e)^(1/n) - 1))
    return I_e * np.exp(-bn * ((r / R_e)**(1.0 / n_sersic) - 1.0))


# -----------------------------
# Observation generator
# -----------------------------

@dataclass
class Observation:
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    true_image: np.ndarray
    psf_kernel: Optional[np.ndarray]
    observed: np.ndarray
    # gating constants (fixed)
    bg_gate_level: float
    bg_gate_sigma: float

def make_mock_observation(
    distribution: str = "exponential",
    true_params: Optional[dict] = None,
    psf_type: Optional[str] = "gaussian",   # "gaussian" or None
    sigma_psf_true: float = 0.2,
    psf_kernel_custom: Optional[np.ndarray] = None,
    psf_n_sigma: float = 3.0,
    bg_level_true: float = 10.0,
    bg_sigma_true: float = 3.0,
    seed: int = 0,
) -> Observation:
    rng = np.random.default_rng(seed)

    x = np.linspace(-2.0, 2.0, 100)
    y = np.linspace(-2.0, 2.0, 100)
    X, Y = np.meshgrid(x, y)

    if true_params is None:
        true_params = {}

    if distribution == "exponential":
        amp = float(true_params.get("amplitude", 100.0))
        x0 = float(true_params.get("x0", 0.0))
        y0 = float(true_params.get("y0", 0.0))
        width = float(true_params.get("width_dist", 0.6))
        true_image = model_exponential_numpy(X, Y, amp, x0, y0, width)
    elif distribution == "sersic":
        amp = float(true_params.get("amplitude", 100.0))
        x0 = float(true_params.get("x0", 0.0))
        y0 = float(true_params.get("y0", 0.0))
        n = float(true_params.get("n_sersic", 2.5))
        R = float(true_params.get("R_sersic", 0.8))
        true_image = model_sersic_numpy(X, Y, amp, x0, y0, n, R)
    elif distribution == "gaussian":
        raise NotImplementedError("distribution='gaussian' is not implemented (as requested).")
    else:
        raise NotImplementedError(f"distribution='{distribution}' is not implemented.")

    psf_kernel = None
    convolved = true_image.copy()

    if psf_kernel_custom is not None:
        psf_kernel = psf_kernel_custom.astype(np.float32)
        psf_kernel = psf_kernel / np.sum(psf_kernel)
        convolved = fft_convolve2d_same_numpy(convolved, psf_kernel)
    elif psf_type is None:
        psf_kernel = None
    elif psf_type == "gaussian":
        psf_kernel = gaussian_psf_kernel(x, y, sigma_psf_true, n_sigma=psf_n_sigma)
        convolved = fft_convolve2d_same_numpy(convolved, psf_kernel)
    else:
        raise NotImplementedError(f"psf_type='{psf_type}' is not implemented.")

    # Add background Gaussian field
    background = bg_level_true + bg_sigma_true * rng.standard_normal(size=convolved.shape)

    # Shot noise: Poisson with mean = convolved (counts). Ensure non-negative mean.
    lam = np.clip(convolved, 0.0, None)
    shot = rng.poisson(lam=lam)

    observed = shot.astype(np.float32) + background.astype(np.float32)

    # Precompute gating bg estimate (sigma clipping) - used ONLY when bg is "unknown"
    bg_gate_level, bg_gate_sigma = sigma_clip_bg(observed, sigma=3.0, maxiters=10)

    return Observation(
        x=x, y=y, X=X, Y=Y,
        true_image=true_image,
        psf_kernel=psf_kernel,
        observed=observed,
        bg_gate_level=bg_gate_level,
        bg_gate_sigma=bg_gate_sigma
    )


# -----------------------------
# Loss definition (numpy + jax)
# -----------------------------

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def chi2_red_numpy(
    observed: np.ndarray,
    model: np.ndarray,
    bg_sigma: float,
    gate_bg_level: float,
    gate_bg_sigma: float,
    k_sigma: float = 3.0,
    tau_counts: float = 2.0,
    dof: int = 1,
) -> float:
    """
    Reduced chi^2 with variance = bg_sigma^2 + N_eff(observed),
    where N_eff uses softplus gating around T = gate_bg_level + k*gate_bg_sigma.
    """
    T = gate_bg_level + k_sigma * gate_bg_sigma
    # Smooth rectifier in *counts*:
    N_eff = softplus((observed - T) / tau_counts) * tau_counts
    variance = (bg_sigma**2) + N_eff
    # prevent divide-by-zero
    variance = np.clip(variance, 1e-9, None)
    resid2 = (observed - model)**2
    chi2 = float(np.sum(resid2 / variance))
    return chi2 / max(dof, 1)

# JAX: use jnp softplus (stable)
def chi2_red_jax(
    observed: jnp.ndarray,
    model: jnp.ndarray,
    bg_sigma: jnp.ndarray,
    gate_bg_level: float,
    gate_bg_sigma: float,
    k_sigma: float = 3.0,
    tau_counts: float = 2.0,
    dof: int = 1,
) -> jnp.ndarray:
    T = gate_bg_level + k_sigma * gate_bg_sigma
    N_eff = jax.nn.softplus((observed - T) / tau_counts) * tau_counts
    variance = (bg_sigma**2) + N_eff
    variance = jnp.clip(variance, 1e-9, None)
    resid2 = (observed - model)**2
    chi2 = jnp.sum(resid2 / variance)
    return chi2 / max(dof, 1)


def nll_gaussian_numpy(
    observed: np.ndarray,
    model: np.ndarray,
    bg_sigma: float,
    gate_bg_level: float,
    gate_bg_sigma: float,
    k_sigma: float = 3.0,
    tau_counts: float = 2.0,
) -> float:
    """
    Negative Log Likelihood for Gaussian errors:
    NLL = 0.5 * sum( (data - model)^2 / var + log(2 * pi * var) )
    """
    T = gate_bg_level + k_sigma * gate_bg_sigma
    N_eff = softplus((observed - T) / tau_counts) * tau_counts
    variance = (bg_sigma**2) + N_eff
    variance = np.clip(variance, 1e-9, None)
    
    resid2 = (observed - model)**2
    # sum over pixels
    nll = 0.5 * np.sum(resid2 / variance + np.log(2.0 * np.pi * variance))
    return float(nll)


def nll_gaussian_jax(
    observed: jnp.ndarray,
    model: jnp.ndarray,
    bg_sigma: jnp.ndarray,
    gate_bg_level: float,
    gate_bg_sigma: float,
    k_sigma: float = 3.0,
    tau_counts: float = 2.0,
) -> jnp.ndarray:
    T = gate_bg_level + k_sigma * gate_bg_sigma
    N_eff = jax.nn.softplus((observed - T) / tau_counts) * tau_counts
    variance = (bg_sigma**2) + N_eff
    variance = jnp.clip(variance, 1e-9, None)
    
    resid2 = (observed - model)**2
    nll = 0.5 * jnp.sum(resid2 / variance + jnp.log(2.0 * jnp.pi * variance))
    return nll


# -----------------------------
# Parameter handling / bounds
# -----------------------------

PARAM_ORDER_EXP_PSF = ["amplitude", "x0", "y0", "width_dist", "bg_level", "bg_sigma", "sigma_PSF"]
PARAM_ORDER_EXP_NOPSF = ["amplitude", "x0", "y0", "width_dist", "bg_level", "bg_sigma"]

def default_bounds(param: str) -> Tuple[float, float]:
    if param == "amplitude":
        return (1e-3, 1e6)
    if param in ("x0", "y0"):
        return (-2.0, 2.0)
    if param == "width_dist":
        return (1e-3, 5.0)
    if param == "bg_level":
        return (0.0, 200.0)
    if param == "bg_sigma":
        return (1e-3, 10.0)
    if param == "sigma_PSF":
        return (1e-3, 1.0)
    if param == "n_sersic":
        return (0.3, 8.0)
    if param == "R_sersic":
        return (1e-3, 5.0)
    raise KeyError(param)

def make_init_constants(distribution: str, include_psf_param: bool) -> Dict[str, float]:
    # Deterministic initial values
    init = {
        "amplitude": 80.0,
        "x0": 0.10,
        "y0": -0.10,
        "width_dist": 0.8,
        "bg_level": 8.0,
        "bg_sigma": 2.0,
        "sigma_PSF": 0.25,
        # for sersic if needed in future:
        "n_sersic": 2.0,
        "R_sersic": 1.0,
    }
    if distribution == "exponential":
        pass
    elif distribution == "sersic":
        init["amplitude"] = 80.0
        init["n_sersic"] = 2.0
        init["R_sersic"] = 1.0
    else:
        raise NotImplementedError(distribution)
    if not include_psf_param:
        init.pop("sigma_PSF", None)
    return init

def perturb_from_truth(truth: Dict[str, float], seed: int = 0, scale: float = 0.05) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in truth.items():
        # relative perturbation
        out[k] = float(v * (1.0 + scale * rng.standard_normal()))
    return out


# -----------------------------
# Model builder (numpy + jax)
# -----------------------------

def build_model_numpy(
    obs: Observation,
    distribution: str,
    params: Dict[str, float],
    psf_kernel: Optional[np.ndarray],
    psf_type: Optional[str],
    psf_n_sigma: float,
) -> np.ndarray:
    X, Y = obs.X, obs.Y
    bg_level = float(params["bg_level"])
    if distribution == "exponential":
        img = model_exponential_numpy(X, Y,
                                     float(params["amplitude"]),
                                     float(params["x0"]),
                                     float(params["y0"]),
                                     float(params["width_dist"]))
    elif distribution == "sersic":
        img = model_sersic_numpy(X, Y,
                                 float(params["amplitude"]),
                                 float(params["x0"]),
                                 float(params["y0"]),
                                 float(params["n_sersic"]),
                                 float(params["R_sersic"]))
    else:
        raise NotImplementedError(distribution)

    if psf_kernel is not None:
        img = fft_convolve2d_same_numpy(img, psf_kernel)

    img = img + bg_level
    return img.astype(np.float32)

def build_model_jax(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    distribution: str,
    params: Dict[str, jnp.ndarray],
    psf_kernel: Optional[jnp.ndarray],
) -> jnp.ndarray:
    bg_level = params["bg_level"]
    if distribution == "exponential":
        r = jnp.sqrt((X - params["x0"])**2 + (Y - params["y0"])**2)
        img = params["amplitude"] * jnp.exp(-r / params["width_dist"])
    elif distribution == "sersic":
        r = jnp.sqrt((X - params["x0"])**2 + (Y - params["y0"])**2)
        n = params["n_sersic"]
        R = params["R_sersic"]
        bn = (2.0 * n - 1.0/3.0 + 4.0/(405.0*n) + 46.0/(25515.0*n*n))
        I_e = params["amplitude"] * jnp.exp(-bn)
        img = I_e * jnp.exp(-bn * ((r / R)**(1.0 / n) - 1.0))
    else:
        raise NotImplementedError(distribution)

    if psf_kernel is not None:
        img = fft_convolve2d_same_jax(img, psf_kernel)

    return img + bg_level


# -----------------------------
# Packing / unpacking free params
# -----------------------------

def pack_free(params: Dict[str, float], free_names: List[str]) -> np.ndarray:
    return np.array([params[n] for n in free_names], dtype=np.float32)

def unpack_free(base: Dict[str, float], free_names: List[str], x: np.ndarray) -> Dict[str, float]:
    out = dict(base)
    for i, n in enumerate(free_names):
        out[n] = float(x[i])
    return out


# -----------------------------
# JAX constrained parameterization
# -----------------------------

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.exp(-x))

def unconstrained_to_bounded(u: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
    return lo + (hi - lo) * sigmoid(u)

def bounded_to_unconstrained(x: float, lo: float, hi: float) -> float:
    # inverse of lo + (hi-lo)*sigmoid(u)
    # clamp into (lo,hi)
    eps = 1e-12
    x = float(np.clip(x, lo + eps, hi - eps))
    t = (x - lo) / (hi - lo)
    return float(np.log(t / (1.0 - t)))

def make_jax_u0_from_x0(free_names: List[str], x0: np.ndarray) -> np.ndarray:
    u0 = []
    for i, name in enumerate(free_names):
        lo, hi = default_bounds(name)
        u0.append(bounded_to_unconstrained(float(x0[i]), lo, hi))
    return np.array(u0, dtype=np.float32)

def u_to_params_dict_jax(u: jnp.ndarray, free_names: List[str], fixed_params: Dict[str, float]) -> Dict[str, jnp.ndarray]:
    # Start from fixed
    params = {k: jnp.array(v, dtype=jnp.float32) for k, v in fixed_params.items()}
    # Add free mapped into bounds
    for i, name in enumerate(free_names):
        lo, hi = default_bounds(name)
        params[name] = unconstrained_to_bounded(u[i], lo, hi)
    return params


# -----------------------------
# Optimizers: SciPy + JAX Adam
# -----------------------------

class EarlyStop(Exception):
    pass

def run_scipy_optimizer(
    method: str,
    obs: Observation,
    distribution: str,
    free_names: List[str],
    fixed_params: Dict[str, float],
    x0: np.ndarray,
    bounds: List[Tuple[float, float]],
    psf_kernel: Optional[np.ndarray],
    gate_bg_level: float,
    gate_bg_sigma: float,
    tau_counts: float,
    maxiter: int,
    target_loss: float,
    out_dir: str,
    log_mode: str = "iter",  # "iter" or "eval"
) -> Dict[str, object]:
    """
    SciPy minimize with callback (iteration logging) or objective wrapper (eval logging).
    """
    ensure_dir(out_dir)
    log_path = os.path.join(out_dir, "log.csv")
    header = ["index", "loss", "cum_time_s", "step_time_s"] + [f"p_{n}" for n in free_names]
    write_csv_header(log_path, header)

    t0 = time.perf_counter()
    last_t = t0
    it_counter = {"i": 0}

    Npix = obs.observed.size
    dof = int(Npix - len(free_names))

    use_nll = "bg_sigma" in free_names

    def eval_loss_and_obj(x: np.ndarray) -> Tuple[float, float]:
        params = unpack_free(fixed_params, free_names, x)
        model = build_model_numpy(obs, distribution, params, psf_kernel, None, 3.0)
        
        # Always calculate chi2 for logging/stopping
        chi2 = chi2_red_numpy(
            observed=obs.observed,
            model=model,
            bg_sigma=float(params["bg_sigma"]),
            gate_bg_level=gate_bg_level,
            gate_bg_sigma=gate_bg_sigma,
            k_sigma=3.0,
            tau_counts=tau_counts,
            dof=dof,
        )
        
        # Calculate objective
        if use_nll:
             obj = nll_gaussian_numpy(
                observed=obs.observed,
                model=model,
                bg_sigma=float(params["bg_sigma"]),
                gate_bg_level=gate_bg_level,
                gate_bg_sigma=gate_bg_sigma,
                k_sigma=3.0,
                tau_counts=tau_counts,
             )
        else:
             obj = chi2
             
        return chi2, obj

    def log_row(idx: int, loss: float, x: np.ndarray) -> None:
        nonlocal last_t
        now = time.perf_counter()
        cum = now - t0
        step = now - last_t
        last_t = now
        append_csv_row(log_path, [idx, loss, cum, step] + [float(v) for v in x])

    def callback_iter(xk: np.ndarray) -> None:
        i = it_counter["i"]
        loss, _ = eval_loss_and_obj(xk)
        log_row(i, loss, xk)
        it_counter["i"] += 1
        if loss <= target_loss:
            raise EarlyStop()

    def objective_wrapped(x: np.ndarray) -> float:
        loss, obj = eval_loss_and_obj(x)
        if log_mode == "eval":
            i = it_counter["i"]
            log_row(i, loss, x)
            it_counter["i"] += 1
            if loss <= target_loss:
                raise EarlyStop()
        return obj

    # --- Plot Initial State ---
    # We want to plot Obs vs Model(x0)
    # Re-evaluate model at x0
    try:
        params_ini = unpack_free(fixed_params, free_names, x0)
        model_ini = build_model_numpy(obs, distribution, params_ini, psf_kernel, None, 3.0)
        plot_fit_comparison(
            obs=obs,
            model=model_ini,
            params=params_ini,
            path=os.path.join(out_dir, "comparison_01_ini.pdf"),
            title=f"Initial: {method} (Chi2={eval_loss_and_obj(x0)[0]:.3f})"
        )
    except Exception as e:
        print(f"Warning: Failed to plot initial state: {e}")

    options = dict(maxiter=maxiter, disp=False)
    res = None
    stopped_early = False

    try:
        if method.upper() == "L-BFGS-B":
            res = minimize(objective_wrapped, x0, method="L-BFGS-B", bounds=bounds, options=options, tol=0.0, gtol=0.0,
                           callback=None if log_mode == "eval" else callback_iter)
        elif method.upper() == "SLSQP":
            res = minimize(objective_wrapped, x0, method="SLSQP", bounds=bounds, options=options, tol=0.0, ftol=0.0,
                           callback=None if log_mode == "eval" else callback_iter)
        else:
            raise ValueError(f"Unknown SciPy method: {method}")
    except EarlyStop:
        stopped_early = True
        # SciPy doesn't give us a result in this exception path; use last logged x as "best"
        # We'll parse log.csv after the run for best. Here we just continue.
    except Exception as e:
        save_json(os.path.join(out_dir, "error.json"), {"error": repr(e)})
        raise
    
    print("method.upper(): ", method.upper())
    print("stopped early: ", stopped_early)
    print("debug point here")
    
    # Determine best from log
    log_data = np.genfromtxt(log_path, delimiter=",", names=True, dtype=None, encoding=None)
    if log_data.size == 0:
        best_idx = None
        best_loss = float("inf")
        best_x = x0
    else:
        # handle single row case where genfromtxt returns 0-d array
        if log_data.ndim == 0:
            log_data = np.atleast_1d(log_data)
        
        losses = np.array(log_data["loss"], dtype=float)
        best_idx = int(np.argmin(losses))
        best_loss = float(losses[best_idx])
        best_x = np.array([log_data[f"p_{n}"][best_idx] for n in free_names], dtype=float)

    # --- Plot Final State ---
    try:
        params_fin = unpack_free(fixed_params, free_names, best_x)
        model_fin = build_model_numpy(obs, distribution, params_fin, psf_kernel, None, 3.0)
        plot_fit_comparison(
            obs=obs,
            model=model_fin,
            params=params_fin,
            path=os.path.join(out_dir, "comparison_02_fin.pdf"),
            title=f"Final: {method} (Chi2={best_loss:.3f})"
        )
    except Exception as e:
        print(f"Warning: Failed to plot final state: {e}")

    summary = {
        "method": method,
        "stopped_early": stopped_early,
        "best_loss": best_loss,
        "best_params_free": {n: float(best_x[i]) for i, n in enumerate(free_names)},
        "n_logged": int(log_data.size) if hasattr(log_data, "size") else 0,
        "scipy_success": bool(getattr(res, "success", False)) if res is not None else False,
        "scipy_message": str(getattr(res, "message", "")) if res is not None else "",
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary



def adam_update(m, v, g, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1**t)
    vhat = v / (1 - b2**t)
    step = lr * mhat / (jnp.sqrt(vhat) + eps)
    return m, v, step


def run_jax_adam(
    obs: Observation,
    distribution: str,
    free_names: List[str],
    fixed_params: Dict[str, float],
    x0_free: np.ndarray,  # bounded initial guess
    psf_kernel_np: Optional[np.ndarray],
    gate_bg_level: float,
    gate_bg_sigma: float,
    tau_counts: float,
    maxiter: int,
    target_loss: float,
    out_dir: str,
    lr: float = 1e-2,
) -> Dict[str, object]:
    ensure_dir(out_dir)
    log_path = os.path.join(out_dir, "log.csv")
    header = ["index", "loss", "cum_time_s", "step_time_s"] + [f"p_{n}" for n in free_names]
    write_csv_header(log_path, header)

    # JAX arrays
    X = jnp.array(obs.X, dtype=jnp.float32)
    Y = jnp.array(obs.Y, dtype=jnp.float32)
    observed = jnp.array(obs.observed, dtype=jnp.float32)

    psf_kernel = None
    if psf_kernel_np is not None:
        psf_kernel = jnp.array(psf_kernel_np, dtype=jnp.float32)

    Npix = obs.observed.size
    dof = int(Npix - len(free_names))

    # Unconstrained initialization u0 from bounded x0
    u0 = make_jax_u0_from_x0(free_names, x0_free)
    u = jnp.array(u0, dtype=jnp.float32)

    use_nll = "bg_sigma" in free_names

    # Wrapper to compute both objective (for grad) and chi2 (for tracking)
    # We only differentiate 'objective'
    def objective_from_u(u_vec: jnp.ndarray) -> jnp.ndarray:
        params = u_to_params_dict_jax(u_vec, free_names, fixed_params)
        model = build_model_jax(X, Y, distribution, params, psf_kernel)
        
        if use_nll:
            return nll_gaussian_jax(
                observed=observed,
                model=model,
                bg_sigma=params["bg_sigma"],
                gate_bg_level=gate_bg_level,
                gate_bg_sigma=gate_bg_sigma,
                k_sigma=3.0,
                tau_counts=tau_counts,
            )
        else:
            return chi2_red_jax(
                observed=observed,
                model=model,
                bg_sigma=params["bg_sigma"],
                gate_bg_level=gate_bg_level,
                gate_bg_sigma=gate_bg_sigma,
                k_sigma=3.0,
                tau_counts=tau_counts,
                dof=dof,
            )

    def get_chi2_from_u(u_vec: jnp.ndarray) -> jnp.ndarray:
        params = u_to_params_dict_jax(u_vec, free_names, fixed_params)
        model = build_model_jax(X, Y, distribution, params, psf_kernel)
        return chi2_red_jax(
             observed=observed,
             model=model,
             bg_sigma=params["bg_sigma"],
             gate_bg_level=gate_bg_level,
             gate_bg_sigma=gate_bg_sigma,
             k_sigma=3.0,
             tau_counts=tau_counts,
             dof=dof,
        )

    # JIT compile
    # We need value_and_grad of the OBJECTIVE
    obj_and_grad = jax.jit(jax.value_and_grad(objective_from_u))
    calc_chi2 = jax.jit(get_chi2_from_u)
    
    _ = obj_and_grad(u)  # warmup
    _ = calc_chi2(u)

    # --- Plot Initial State ---
    try:
        # Convert u to params dict (numpy for plotting)
        params_j = u_to_params_dict_jax(u, free_names, fixed_params)
        params_ini = {k: float(v) for k,v in params_j.items()}
        # Need numpy model for plotting
        model_ini = build_model_numpy(obs, distribution, params_ini, psf_kernel_np, None, 3.0)
        chi2_ini = float(calc_chi2(u))
        plot_fit_comparison(
            obs=obs,
            model=model_ini,
            params=params_ini,
            path=os.path.join(out_dir, "comparison_01_ini.pdf"),
            title=f"Initial: JAX_ADAM (Chi2={chi2_ini:.3f})"
        )
    except Exception as e:
        print(f"Warning: Failed to plot initial state (JAX): {e}")


    t0 = time.perf_counter()
    last_t = t0

    m = jnp.zeros_like(u)
    v = jnp.zeros_like(u)

    best_loss = float("inf")
    best_params = None
    stopped_early = False
    
    # Run loop
    for i in range(maxiter):
        # 1. Get gradients of OBJECTIVE
        obj_val, grad = obj_and_grad(u)
        
        # 2. Compute Chi2 for logging/stopping
        # If not using NLL, obj_val is chi2. If NLL, we must compute chi2 separately
        if use_nll:
             loss_val = calc_chi2(u)
        else:
             loss_val = obj_val
        
        loss = float(loss_val)

        # Map to bounded params for logging
        params_j = u_to_params_dict_jax(u, free_names, fixed_params)
        free_vals = [float(params_j[n]) for n in free_names]

        now = time.perf_counter()
        cum = now - t0
        step = now - last_t
        last_t = now
        append_csv_row(log_path, [i, loss, cum, step] + free_vals)

        if loss < best_loss:
            best_loss = loss
            best_params = {n: float(params_j[n]) for n in free_names}

        if loss <= target_loss:
            stopped_early = True
            break

        # Adam step in unconstrained space
        m, v, step_u = adam_update(m, v, grad, t=i+1, lr=lr)
        u = u - step_u
    
    # --- Plot Final State ---
    try:
        # Re-calc final model
        params_fin = best_params if best_params else {n: float(u_to_params_dict_jax(u, free_names, fixed_params)[n]) for n in free_names}
        # merge with fixed
        full_params_fin = {**fixed_params, **params_fin}
        
        model_fin = build_model_numpy(obs, distribution, full_params_fin, psf_kernel_np, None, 3.0)
        plot_fit_comparison(
            obs=obs,
            model=model_fin,
            params=full_params_fin,
            path=os.path.join(out_dir, "comparison_02_fin.pdf"),
            title=f"Final: JAX_ADAM (Chi2={best_loss:.3f})"
        )
    except Exception as e:
        print(f"Warning: Failed to plot final state (JAX): {e}")

    summary = {
        "method": "JAX_ADAM",
        "stopped_early": stopped_early,
        "best_loss": float(best_loss),
        "best_params_free": best_params if best_params is not None else {},
        "n_logged": i + 1,
        "lr": lr,
    }
    save_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


# -----------------------------
# Plotting (per-case)
# -----------------------------

def load_log_csv(path: str) -> dict:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    if data.size == 0:
        return {"loss": np.array([]), "cum_time_s": np.array([])}
    return {
        "loss": np.array(data["loss"], dtype=float),
        "cum_time_s": np.array(data["cum_time_s"], dtype=float),
        "index": np.array(data["index"], dtype=int),
    }

def plot_case_comparison(case_dir: str, methods: List[str], out_pdf_prefix: str, target_loss: float) -> None:
    logs = {}
    for m in methods:
        log_path = os.path.join(case_dir, m, "log.csv")
        if os.path.exists(log_path):
            logs[m] = load_log_csv(log_path)

    # Iteration vs loss
    plt.figure()
    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label='loss=1.0')
    plt.axhline(y=target_loss, color="yellow", linestyle="--", linewidth=1, label='target loss')
    for i, (m, d) in enumerate(logs.items()):
        if d["loss"].size == 0:
            continue
        plt.plot(d["index"], d["loss"], label=m, linestyle="-" if i % 2 == 0 else "--", linewidth=1)
    plt.yscale(LOSS_SCALE)
    plt.xlabel("Iteration (optimizer iteration)")
    if LOSS_SCALE == 'log':
        plt.ylabel("Reduced chi^2 (log scale)")
        plt.ylim(bottom=0.5)
    else:
        plt.ylabel("Reduced chi^2 (linear scale)")
        plt.ylim(bottom=0, top=5)
    plt.title("Iteration vs Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f"{out_pdf_prefix}_iter_vs_loss.pdf"))
    plt.close()

    # Time vs loss
    plt.figure()
    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label='loss=1.0')
    plt.axhline(y=target_loss, color="yellow", linestyle="--", linewidth=1, label='target loss')
    for i, (m, d) in enumerate(logs.items()):
        if d["loss"].size == 0:
            continue
        plt.plot(d["cum_time_s"], d["loss"], label=m, linestyle="-" if i % 2 == 0 else "--", linewidth=1)
    plt.yscale(LOSS_SCALE)
    plt.xlabel("Cumulative time (s)")
    if LOSS_SCALE == 'log':
        plt.ylabel("Reduced chi^2 (log scale)")
        plt.ylim(bottom=0.5)
    else:
        plt.ylabel("Reduced chi^2 (linear scale)")
        plt.ylim(bottom=0, top=5)
    plt.title("Time vs Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f"{out_pdf_prefix}_time_vs_loss.pdf"))
    plt.savefig(os.path.join(case_dir, f"{out_pdf_prefix}_time_vs_loss.pdf"))
    plt.close()

def plot_fit_comparison(
    obs: Observation,
    model: np.ndarray,
    params: Dict[str, float],
    path: str,
    title: str = ""
) -> None:
    """
    Plot Observation, Model, and Residuals in a (1,3) subplot.
    Residuals = (Observed - Model) / Sqrt(Variance)
    where Variance ~ bg_sigma^2 + Model (or similar).
    We use the same definition as chi2_red except per pixel.
    Variance = bg_sigma^2 + softplus((Obs-T)/tau)*tau
    """
    # Recalculate variance map for residuals
    T = obs.bg_gate_level + 3.0 * obs.bg_gate_sigma
    N_eff = softplus((obs.observed - T) / 2.0) * 2.0
    bg_sig = params.get("bg_sigma", obs.bg_gate_sigma) # Use fit param if avail, else gate
    variance = (bg_sig**2) + N_eff
    variance = np.clip(variance, 1e-9, None)
    sigma_map = np.sqrt(variance)
    
    resid = (obs.observed - model) / sigma_map
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Observation
    im0 = axes[0].imshow(obs.observed, origin='lower', cmap='viridis')
    axes[0].set_title("Observed")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Model
    im1 = axes[1].imshow(model, origin='lower', cmap='viridis')
    axes[1].set_title("Model")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Residual
    im2 = axes[2].imshow(resid, origin='lower', cmap='coolwarm', vmin=-5, vmax=5)
    axes[2].set_title("Residuals (sigma)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    if title:
        plt.suptitle(title)
        
    plt.tight_layout()
    plt.savefig(path)
    plt.close()



# -----------------------------
# Benchmark runner (4 cases)
# -----------------------------

@dataclass
class CaseSpec:
    name: str
    psf_used: bool
    known: List[str]
    free: List[str]

def run_all_cases(
    out_root: str = "bench_out",
    distribution: str = "exponential",
    seed: int = 0,
    psf_sigma_true: float = 0.2,
    psf_n_sigma: float = 3.0,
    bg_level_true: float = 10.0,
    bg_sigma_true: float = 3.0,
    maxiter: int = 200,
    target_loss: float = 1.3,
    tau_counts: float = 2.0,
    scipy_log_mode: str = "iter",  # "iter" or "eval"
    init_mode: str = "constants",  # "constants" or "truth_plus_random"
    init_perturb_seed: int = 0,
    init_perturb_scale: float = 0.05,
) -> None:
    ensure_dir(out_root)

    # "Truth" used to make observation (exponential defaults)
    true_params = {
        "amplitude": 100.0,
        "x0": 0.0,
        "y0": 0.0,
        "width_dist": 0.6,
        "bg_level": bg_level_true,
        "bg_sigma": bg_sigma_true,
        "sigma_PSF": psf_sigma_true,
    }

    cases = [
        CaseSpec(
            name="Case1_NoPSF_known_x0y0_bg",
            psf_used=False,
            known=["x0", "y0", "bg_level", "bg_sigma"],
            free=["amplitude", "width_dist"],
        ),
        CaseSpec(
            name="Case2_NoPSF_all_free",
            psf_used=False,
            known=[],
            free=["amplitude", "x0", "y0", "width_dist", "bg_level", "bg_sigma"],
        ),
        CaseSpec(
            name="Case3_GaussPSF_known_x0y0_bg",
            psf_used=True,
            known=["x0", "y0", "bg_level", "bg_sigma"],
            free=["amplitude", "width_dist", "sigma_PSF"],
        ),
        CaseSpec(
            name="Case4_GaussPSF_all_free",
            psf_used=True,
            known=[],
            free=["amplitude", "x0", "y0", "width_dist", "bg_level", "bg_sigma", "sigma_PSF"],
        ),
    ]

    for case in cases:
        case_dir = os.path.join(out_root, case.name)
        ensure_dir(case_dir)

        # Make observation for this case
        psf_type = "gaussian" if case.psf_used else None
        obs = make_mock_observation(
            distribution=distribution,
            true_params=true_params,
            psf_type=psf_type,
            sigma_psf_true=psf_sigma_true,
            psf_kernel_custom=None,
            psf_n_sigma=psf_n_sigma,
            bg_level_true=bg_level_true,
            bg_sigma_true=bg_sigma_true,
            seed=seed,
        )

        # Determine PSF kernel to use in model building:
        # - If PSF is used and gaussian, kernel depends on sigma_PSF if sigma_PSF is free
        #   For SciPy, we will rebuild kernel each loss call if sigma_PSF is in free params.
        #   For JAX, same.
        #
        # To keep the script compact and consistent, we handle sigma_PSF only for gaussian PSF
        # and only by regenerating the kernel from sigma_PSF at each evaluation.
        #
        # However: regenerating kernel inside JAX per step is expensive and awkward.
        # So here we choose: when sigma_PSF is a fit parameter, we approximate by
        # building the kernel from sigma_PSF in numpy *outside* JAX? Not possible.
        #
        # Solution: for this benchmark, we handle sigma_PSF as a parameter by:
        #  - building the PSF kernel on the fly in numpy for SciPy (OK)
        #  - for JAX Adam, we also build kernel on the fly in JAX using a fixed support grid.
        #
        # For simplicity, we implement JAX PSF-on-the-fly only for Gaussian PSF with fixed support size.

        # Gate parameters:
        if ("bg_level" in case.known) and ("bg_sigma" in case.known):
            gate_level = bg_level_true
            gate_sigma = bg_sigma_true
        else:
            gate_level = obs.bg_gate_level
            gate_sigma = obs.bg_gate_sigma

        # Initial guess
        include_psf_param = case.psf_used
        init = make_init_constants(distribution=distribution, include_psf_param=include_psf_param)
        if init_mode == "truth_plus_random":
            # Start from truth then perturb
            truth_for_init = {k: true_params[k] for k in case.free}
            pert = perturb_from_truth(truth_for_init, seed=init_perturb_seed, scale=init_perturb_scale)
            for k, v in pert.items():
                init[k] = v

        # Fixed params
        fixed_params = {}
        for name in case.known:
            fixed_params[name] = true_params[name]
        # ensure required params exist in fixed (e.g. bg_sigma used by loss)
        for must in ["bg_level", "bg_sigma", "x0", "y0"]:
            if must not in fixed_params and must not in case.free:
                # shouldn't happen given case specs, but keep safe
                fixed_params[must] = true_params[must]

        # Free init vector + bounds
        x0_free = np.array([init[n] for n in case.free], dtype=np.float32)
        bounds = [default_bounds(n) for n in case.free]
        # clamp init into bounds
        for i, (lo, hi) in enumerate(bounds):
            x0_free[i] = float(np.clip(x0_free[i], lo + 1e-9, hi - 1e-9))

        # Helper: build PSF kernel for a given sigma (numpy)
        def psf_kernel_for_sigma(sig: float) -> Optional[np.ndarray]:
            if not case.psf_used:
                return None
            return gaussian_psf_kernel(obs.x, obs.y, sig, n_sigma=psf_n_sigma)

        # SciPy objective needs to rebuild model; easiest is to embed kernel regeneration via fixed_params
        def run_scipy(method: str, method_dir: str):
            # We'll implement sigma_PSF handling by overriding build_model_numpy via a wrapper
            ensure_dir(method_dir)
            log_path = os.path.join(method_dir, "log.csv")
            header = ["index", "loss", "cum_time_s", "step_time_s"] + [f"p_{n}" for n in case.free]
            write_csv_header(log_path, header)

            t0 = time.perf_counter()
            last_t = t0
            it_counter = {"i": 0}
            Npix = obs.observed.size
            dof = int(Npix - len(case.free))

            use_nll = "bg_sigma" in case.free

            def eval_loss_and_obj(x: np.ndarray) -> Tuple[float, float]:
                params = unpack_free(fixed_params, case.free, x)
                # build PSF kernel if needed
                ker = None
                if case.psf_used:
                    if "sigma_PSF" in case.free:
                        ker = psf_kernel_for_sigma(float(params["sigma_PSF"]))
                    else:
                        ker = psf_kernel_for_sigma(psf_sigma_true)
                model = build_model_numpy(obs, distribution, params, ker, "gaussian" if case.psf_used else None, psf_n_sigma)
                
                chi2 = chi2_red_numpy(
                    observed=obs.observed,
                    model=model,
                    bg_sigma=float(params["bg_sigma"]),
                    gate_bg_level=gate_level,
                    gate_bg_sigma=gate_sigma,
                    k_sigma=3.0,
                    tau_counts=tau_counts,
                    dof=dof,
                )

                if use_nll:
                     obj = nll_gaussian_numpy(
                        observed=obs.observed,
                        model=model,
                        bg_sigma=float(params["bg_sigma"]),
                        gate_bg_level=gate_level,
                        gate_bg_sigma=gate_sigma,
                        k_sigma=3.0,
                        tau_counts=tau_counts,
                     )
                else:
                     obj = chi2
                
                return chi2, obj

            def log_row(idx: int, loss: float, x: np.ndarray) -> None:
                nonlocal last_t
                now = time.perf_counter()
                cum = now - t0
                step = now - last_t
                last_t = now
                append_csv_row(log_path, [idx, loss, cum, step] + [float(v) for v in x])

            def callback_iter(xk: np.ndarray) -> None:
                i = it_counter["i"]
                loss, _ = eval_loss_and_obj(xk)
                log_row(i, loss, xk)
                it_counter["i"] += 1
                # if loss <= target_loss:
                #     raise EarlyStop()

            def objective_wrapped(x: np.ndarray) -> float:
                loss, obj = eval_loss_and_obj(x)
                if scipy_log_mode == "eval":
                    i = it_counter["i"]
                    log_row(i, loss, x)
                    it_counter["i"] += 1
                    # if loss <= target_loss:
                    #     raise EarlyStop()
                return obj

            # --- Plot Initial State ---
            try:
                params_ini = unpack_free(fixed_params, case.free, x0_free)
                # build kernel for init
                ker_ini = None
                if case.psf_used:
                    if "sigma_PSF" in case.free:
                        ker_ini = psf_kernel_for_sigma(float(params_ini["sigma_PSF"]))
                    else:
                        ker_ini = psf_kernel_for_sigma(psf_sigma_true)
                
                model_ini = build_model_numpy(obs, distribution, params_ini, ker_ini, "gaussian" if case.psf_used else None, psf_n_sigma)
                chi2_ini, _ = eval_loss_and_obj(x0_free)
                
                plot_fit_comparison(
                    obs=obs,
                    model=model_ini,
                    params=params_ini,
                    path=os.path.join(method_dir, "comparison_01_ini.pdf"),
                    title=f"Initial: {method} (Chi2={chi2_ini:.3f})"
                )
            except Exception as e:
                print(f"Warning: Failed to plot initial state ({method}): {e}")

            options = dict(maxiter=maxiter, disp=False)
            
            # Stricter tolerances to force maxiter
            # float32 requires larger eps for finite difference
            options["eps"] = 1e-4

            if method == "SLSQP":
                options["ftol"] = 1e-30
            elif method == "L-BFGS-B":
                options["ftol"] = 1e-30
                options["gtol"] = 1e-30
            
            res = None
            stopped_early = False
            try:
                options_wo_disp = copy.deepcopy(options)
                options_wo_disp.pop("disp", None) # to avoid the following warning message: warnings.warn("scipy.optimize: The `disp` and `iprint` options of the "DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
                
                # 'eps' is not a standard option for all methods in 'options' dict, 
                # but 'minimize' takes 'options' which passes them down.
                # For L-BFGS-B and SLSQP, 'eps' (or 'epsilon') inside options works in some versions, 
                # but 'minimize' has a specific 'options' key for it.
                # Actually, for L-BFGS-B/SLSQP in scipy, the finite difference step is controlled by 
                # 'eps' in options (SLSQP) or 'eps' in options (L-BFGS-B)? 
                # Let's check docs or be safe. 
                # Default approx_grad=True uses forward diff.
                # 'jac' parameter is missing, so it uses 2-point.
                
                res = minimize(
                    objective_wrapped,
                    x0_free,
                    method=method,
                    bounds=bounds,
                    options=options_wo_disp,
                    tol=0.0,
                    callback=None if scipy_log_mode == "eval" else callback_iter
                )
            except EarlyStop:
                stopped_early = True

            # Determine best from log
            log_data = np.genfromtxt(log_path, delimiter=",", names=True, dtype=None, encoding=None, ndmin=1)
            if log_data.size == 0:
                best_loss = float("inf")
                best_x = x0_free
            else:
                losses = np.array(log_data["loss"], dtype=float, ndmin=1)
                best_idx = int(np.argmin(losses))
                best_loss = float(losses[best_idx])
                best_x = np.array([log_data[f"p_{n}"][best_idx] for n in case.free], dtype=float)

            # --- Plot Final State ---
            try:
                params_fin = unpack_free(fixed_params, case.free, best_x)
                ker_fin = None
                if case.psf_used:
                    if "sigma_PSF" in case.free:
                        ker_fin = psf_kernel_for_sigma(float(params_fin["sigma_PSF"]))
                    else:
                        ker_fin = psf_kernel_for_sigma(psf_sigma_true)

                model_fin = build_model_numpy(obs, distribution, params_fin, ker_fin, "gaussian" if case.psf_used else None, psf_n_sigma)
                
                plot_fit_comparison(
                    obs=obs,
                    model=model_fin,
                    params=params_fin,
                    path=os.path.join(method_dir, "comparison_02_fin.pdf"),
                    title=f"Final: {method} (Chi2={best_loss:.3f})"
                )
            except Exception as e:
                print(f"Warning: Failed to plot final state ({method}): {e}")

            summary = {
                "case": case.name,
                "method": method,
                "distribution": distribution,
                "psf_used": case.psf_used,
                "known_params": case.known,
                "free_params": case.free,
                "gate_bg_level": gate_level,
                "gate_bg_sigma": gate_sigma,
                "tau_counts": tau_counts,
                "target_loss": target_loss,
                "maxiter": maxiter,
                "stopped_early": stopped_early,
                "best_loss": best_loss,
                "best_params_free": {n: float(best_x[i]) for i, n in enumerate(case.free)},
                "scipy_success": bool(getattr(res, "success", False)) if res is not None else False,
                "scipy_message": str(getattr(res, "message", "")) if res is not None else "",
            }
            save_json(os.path.join(method_dir, "summary.json"), summary)

        # JAX Adam: needs PSF handling. If sigma_PSF is free, create Gaussian kernel in JAX with fixed support.
        # We do this by precomputing a kernel coordinate grid at the maximum support size implied by bounds.
        def run_jax(method_dir: str):
            ensure_dir(method_dir)

            # Build fixed kernel coordinate grid for JAX Gaussian PSF if needed
            psf_kernel_np_fixed = None

            if not case.psf_used:
                psf_kernel_np_fixed = None
            else:
                if "sigma_PSF" in case.free:
                    # Build a kernel grid based on the *upper bound* sigma to ensure support covers n_sigma
                    sig_hi = default_bounds("sigma_PSF")[1]
                    dx = float(obs.x[1] - obs.x[0])
                    dy = float(obs.y[1] - obs.y[0])
                    half_wx = int(np.ceil((psf_n_sigma * sig_hi) / dx))
                    half_wy = int(np.ceil((psf_n_sigma * sig_hi) / dy))
                    kx = np.arange(-half_wx, half_wx + 1) * dx
                    ky = np.arange(-half_wy, half_wy + 1) * dy
                    KX, KY = np.meshgrid(kx, ky)
                    # store coords for JAX kernel build
                    KXj = jnp.array(KX, dtype=jnp.float32)
                    KYj = jnp.array(KY, dtype=jnp.float32)

                    Xj = jnp.array(obs.X, dtype=jnp.float32)
                    Yj = jnp.array(obs.Y, dtype=jnp.float32)
                    observed_j = jnp.array(obs.observed, dtype=jnp.float32)
                    Npix = obs.observed.size
                    dof = int(Npix - len(case.free))

                    # logging
                    log_path = os.path.join(method_dir, "log.csv")
                    header = ["index", "loss", "cum_time_s", "step_time_s"] + [f"p_{n}" for n in case.free]
                    write_csv_header(log_path, header)

                    # init u
                    u0 = jnp.array(make_jax_u0_from_x0(case.free, x0_free), dtype=jnp.float32)
                    u = u0
                    m = jnp.zeros_like(u)
                    v = jnp.zeros_like(u)

                    use_nll = True  # Case 4 has free inputs including bg_sigma

                    def objective_from_u(u_vec: jnp.ndarray) -> jnp.ndarray:
                        params = u_to_params_dict_jax(u_vec, case.free, fixed_params)
                        sig = params["sigma_PSF"]
                        ker = jnp.exp(-(KXj**2 + KYj**2) / (2.0 * sig**2))
                        ker = ker / jnp.sum(ker)
                        model = build_model_jax(Xj, Yj, distribution, params, ker)
                        
                        if use_nll:
                            return nll_gaussian_jax(
                                observed=observed_j,
                                model=model,
                                bg_sigma=params["bg_sigma"],
                                gate_bg_level=gate_level,
                                gate_bg_sigma=gate_sigma,
                                k_sigma=3.0,
                                tau_counts=tau_counts,
                            )
                        else:
                            return chi2_red_jax(
                                observed=observed_j,
                                model=model,
                                bg_sigma=params["bg_sigma"],
                                gate_bg_level=gate_level,
                                gate_bg_sigma=gate_sigma,
                                k_sigma=3.0,
                                tau_counts=tau_counts,
                                dof=dof,
                            )

                    def get_chi2_from_u(u_vec: jnp.ndarray) -> jnp.ndarray:
                        params = u_to_params_dict_jax(u_vec, case.free, fixed_params)
                        sig = params["sigma_PSF"]
                        # Rebuild kernel for metrics... wasteful but OK for checking
                        ker = jnp.exp(-(KXj**2 + KYj**2) / (2.0 * sig**2))
                        ker = ker / jnp.sum(ker)
                        model = build_model_jax(Xj, Yj, distribution, params, ker)
                        
                        return chi2_red_jax(
                            observed=observed_j,
                            model=model,
                            bg_sigma=params["bg_sigma"],
                            gate_bg_level=gate_level,
                            gate_bg_sigma=gate_sigma,
                            k_sigma=3.0,
                            tau_counts=tau_counts,
                            dof=dof,
                        )

                    obj_and_grad = jax.jit(jax.value_and_grad(objective_from_u))
                    calc_chi2 = jax.jit(get_chi2_from_u)
                    
                    _ = obj_and_grad(u)  # warm-up (excluded from timing)
                    _ = calc_chi2(u)

                    # --- Plot Initial State ---
                    try:
                        params_j = u_to_params_dict_jax(u, case.free, fixed_params)
                        params_ini = {k: float(v) for k,v in params_j.items()}
                        # Build numpy kernel for initial plotting
                        sig_psf = params_ini["sigma_PSF"]
                        ker_ini = gaussian_psf_kernel(obs.x, obs.y, sig_psf, n_sigma=psf_n_sigma)
                        model_ini = build_model_numpy(obs, distribution, params_ini, ker_ini, "gaussian", psf_n_sigma)
                        chi2_ini = float(calc_chi2(u))
                        
                        plot_fit_comparison(
                            obs=obs,
                            model=model_ini,
                            params=params_ini,
                            path=os.path.join(method_dir, "comparison_01_ini.pdf"),
                            title=f"Initial: JAX_ADAM (Chi2={chi2_ini:.3f})"
                        )
                    except Exception as e:
                        print(f"Warning: Failed to plot initial state (JAX-PSF): {e}")

                    t0 = time.perf_counter()
                    last_t = t0

                    best_loss = float("inf")
                    best_params = None
                    stopped_early = False

                    for i in range(maxiter):
                        obj_val, grad = obj_and_grad(u)
                        
                        if use_nll:
                             loss_val = calc_chi2(u)
                        else:
                             loss_val = obj_val
                        
                        loss = float(loss_val)

                        params_j = u_to_params_dict_jax(u, case.free, fixed_params)
                        free_vals = [float(params_j[n]) for n in case.free]

                        now = time.perf_counter()
                        cum = now - t0
                        step = now - last_t
                        last_t = now
                        append_csv_row(log_path, [i, loss, cum, step] + free_vals)

                        if loss < best_loss:
                            best_loss = loss
                            best_params = {n: float(params_j[n]) for n in case.free}

                        if loss <= target_loss:
                            stopped_early = True
                            break

                        m, v, step_u = adam_update(m, v, grad, t=i+1, lr=1e-2)
                        u = u - step_u
                    
                    # --- Plot Final State ---
                    try:
                        params_fin = best_params if best_params else {n: float(u_to_params_dict_jax(u, case.free, fixed_params)[n]) for n in case.free}
                        full_params_fin = {**fixed_params, **params_fin}
                        
                        sig_psf = full_params_fin["sigma_PSF"]
                        ker_fin = gaussian_psf_kernel(obs.x, obs.y, sig_psf, n_sigma=psf_n_sigma)
                        model_fin = build_model_numpy(obs, distribution, full_params_fin, ker_fin, "gaussian", psf_n_sigma)
                        
                        plot_fit_comparison(
                            obs=obs,
                            model=model_fin,
                            params=full_params_fin,
                            path=os.path.join(method_dir, "comparison_02_fin.pdf"),
                            title=f"Final: JAX_ADAM (Chi2={best_loss:.3f})"
                        )
                    except Exception as e:
                        print(f"Warning: Failed to plot final state (JAX-PSF): {e}")



                    summary = {
                        "case": case.name,
                        "method": "JAX_ADAM",
                        "distribution": distribution,
                        "psf_used": True,
                        "sigma_PSF_free": True,
                        "known_params": case.known,
                        "free_params": case.free,
                        "gate_bg_level": gate_level,
                        "gate_bg_sigma": gate_sigma,
                        "tau_counts": tau_counts,
                        "target_loss": target_loss,
                        "maxiter": maxiter,
                        "stopped_early": stopped_early,
                        "best_loss": float(best_loss),
                        "best_params_free": best_params if best_params is not None else {},
                        "lr": 1e-2,
                    }
                    save_json(os.path.join(method_dir, "summary.json"), summary)
                    return

                else:
                    # sigma_PSF fixed at truth for PSF convolution in model
                    psf_kernel_np_fixed = psf_kernel_for_sigma(psf_sigma_true)

            # If we got here, PSF kernel is fixed (or no PSF).
            run_jax_adam(
                obs=obs,
                distribution=distribution,
                free_names=case.free,
                fixed_params=fixed_params,
                x0_free=x0_free,
                psf_kernel_np=psf_kernel_np_fixed,
                gate_bg_level=gate_level,
                gate_bg_sigma=gate_sigma,
                tau_counts=tau_counts,
                maxiter=maxiter,
                target_loss=target_loss,
                out_dir=method_dir,
                lr=1e-2,
            )

        # Run methods
        methods = ["SLSQP", "L-BFGS-B", "JAX_ADAM"]
        ensure_dir(os.path.join(case_dir, "SLSQP"))
        ensure_dir(os.path.join(case_dir, "L-BFGS-B"))
        ensure_dir(os.path.join(case_dir, "JAX_ADAM"))

        run_scipy("SLSQP", os.path.join(case_dir, "SLSQP"))
        run_scipy("L-BFGS-B", os.path.join(case_dir, "L-BFGS-B"))
        run_jax(os.path.join(case_dir, "JAX_ADAM"))

        # Summarize convergence
        summary_rows = []
        bad = []
        for m in methods:
            s_path = os.path.join(case_dir, m, "summary.json")
            if os.path.exists(s_path):
                with open(s_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                loss = float(s.get("best_loss", float("inf")))
                summary_rows.append((m, loss))
                if loss > 1.5:
                    bad.append(m)

        case_summary = {
            "case": case.name,
            "distribution": distribution,
            "psf_used": case.psf_used,
            "known_params": case.known,
            "free_params": case.free,
            "gate_bg_level": gate_level,
            "gate_bg_sigma": gate_sigma,
            "tau_counts": tau_counts,
            "target_loss": target_loss,
            "maxiter": maxiter,
            "results_best_loss": {m: loss for m, loss in summary_rows},
            "bad_convergence_methods_loss_gt_1p5": bad,
            "notes": {
                "padding_rule": "Linear FFT conv uses full size (H+Kh-1,W+Kw-1) then crops starting at (Kh//2,Kw//2). This is equivalent to padding ~Kh//2, Kw//2 on each side for 'same' output when kernel is odd-sized and centered."
            }
        }
        save_json(os.path.join(case_dir, "case_summary.json"), case_summary)

        # Plots
        plot_case_comparison(case_dir, methods, out_pdf_prefix="compare", target_loss=target_loss)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=str, default="bench_out")
    p.add_argument("--distribution", type=str, default="exponential", choices=["exponential", "sersic"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--target-loss", type=float, default=1.0)
    p.add_argument("--tau-counts", type=float, default=2.0)
    p.add_argument("--scipy-log-mode", type=str, default="eval", choices=["iter", "eval"])
    p.add_argument("--init-mode", type=str, default="constants", choices=["constants", "truth_plus_random"])
    p.add_argument("--init-perturb-seed", type=int, default=0)
    p.add_argument("--init-perturb-scale", type=float, default=0.05)
    args = p.parse_args()

    run_all_cases(
        out_root=args.out_root,
        distribution=args.distribution,
        seed=args.seed,
        maxiter=args.maxiter,
        target_loss=args.target_loss,
        tau_counts=args.tau_counts,
        scipy_log_mode=args.scipy_log_mode,
        init_mode=args.init_mode,
        init_perturb_seed=args.init_perturb_seed,
        init_perturb_scale=args.init_perturb_scale,
    )

if __name__ == "__main__":
    main()