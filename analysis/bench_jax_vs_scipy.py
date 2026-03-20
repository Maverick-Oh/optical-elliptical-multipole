#!/usr/bin/env python3
"""
Benchmark: JAX (CPU) vs non-JAX model evaluation and SciPy optimization.

Compares wall-clock times for:
  1) Single model evaluation (forward pass only)
  2) Full SciPy SLSQP optimization loop
  3) Full SciPy L-BFGS-B optimization loop (with JAX grad if available)

Usage:
  python bench_jax_vs_scipy.py                      # uses built-in mock
  python bench_jax_vs_scipy.py --hdf5 path/to.hdf5  # uses a real cutout
"""
import os, sys, time, argparse, warnings
import numpy as np

# ── project root on PYTHONPATH ──
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── non-JAX imports (always available) ──
from optical_elliptical_multipole.nonjax.intensity_functions import sersic as sersic_np
from optical_elliptical_multipole.nonjax.profiles2D import Elliptical_Multipole_Profile_2D as EMP2D_np

# ── JAX imports (guarded) ──
HAS_JAX = False
try:
    import jax
    jax.config.update("jax_platform_name", "cpu")  # force CPU
    import jax.numpy as jnp
    from jax import jit, grad, value_and_grad
    from optical_elliptical_multipole.jax.intensity_functions import sersic as sersic_jax
    from optical_elliptical_multipole.jax.profiles2D import Elliptical_Multipole_Profile_2D as EMP2D_jax
    HAS_JAX = True
    print(f"JAX {jax.__version__} found (CPU mode)")
except ImportError:
    print("JAX not found – only non-JAX benchmarks will run.")

from scipy.optimize import minimize, Bounds
from tools_fitting import build_arcsec_grid, pack_params, unpack_params, downsample

PIX_SCALE = 0.03

# ═══════════════════════════════════════════════════════════
#  Model evaluation helpers
# ═══════════════════════════════════════════════════════════

def make_model_np(X, Y, params, m, background):
    """Non-JAX model evaluation."""
    I = EMP2D_np(
        X, Y, sersic_np,
        q=params['q'], theta_ell=params['theta_ell'],
        m=np.asarray(m), a_m=np.asarray(params['a_m']),
        phi_m=np.asarray(params['phi_m']),
        x0=params['x0'], y0=params['y0'],
        amplitude=params['amplitude'],
        R_sersic=params['R_sersic'], n_sersic=params['n_sersic']
    )
    return I + background


def make_model_jax(X, Y, params, m, background):
    """JAX model evaluation (un-jitted)."""
    I = EMP2D_jax(
        X, Y, sersic_jax,
        q=params['q'], theta_ell=params['theta_ell'],
        m=jnp.asarray(m), a_m=jnp.asarray(params['a_m']),
        phi_m=jnp.asarray(params['phi_m']),
        x0=params['x0'], y0=params['y0'],
        amplitude=params['amplitude'],
        R_sersic=params['R_sersic'], n_sersic=params['n_sersic']
    )
    return I + background


# ═══════════════════════════════════════════════════════════
#  Timing utility
# ═══════════════════════════════════════════════════════════

def time_fn(fn, *args, n_warmup=2, n_repeat=10, label=""):
    """Time *fn* with warmup.  Returns (mean_s, std_s, last_result)."""
    for _ in range(n_warmup):
        out = fn(*args)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        if HAS_JAX:
            # block until JAX async dispatch finishes
            try:
                out.block_until_ready()
            except AttributeError:
                pass
        times.append(time.perf_counter() - t0)
    mean_t = np.mean(times)
    std_t = np.std(times)
    print(f"  {label:40s}  {mean_t*1e3:8.2f} ± {std_t*1e3:5.2f} ms  (n={n_repeat})")
    return mean_t, std_t, out


# ═══════════════════════════════════════════════════════════
#  Build a synthetic observation (or load real HDF5)
# ═══════════════════════════════════════════════════════════

def make_mock_data(shape=(101, 101), ss_factor=1):
    """Generate a small mock for benchmarking."""
    ny, nx = shape
    if ss_factor > 1:
        X, Y, ext = build_arcsec_grid((ny*ss_factor, nx*ss_factor), pixscale=PIX_SCALE/ss_factor)
    else:
        X, Y, ext = build_arcsec_grid(shape, pixscale=PIX_SCALE)

    true_params = dict(
        n_sersic=3.0, R_sersic=1.2, amplitude=0.03,
        q=0.8, theta_ell=0.3,
        a_m=np.array([0.03, 0.02]),
        phi_m=np.array([0.1, -0.05]),
        x0=0.0, y0=0.0,
        background=0.001,
    )
    m = [3, 4]
    bg = true_params['background']

    model = make_model_np(X, Y, true_params, m, bg)
    if ss_factor > 1:
        model = downsample(model, ss_factor)
        X_det, Y_det, ext = build_arcsec_grid(shape, pixscale=PIX_SCALE)
    else:
        X_det = X; Y_det = Y

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.005, model.shape)
    sci = model + noise
    wht = np.full_like(sci, 1.0 / 0.005**2)

    return X, Y, X_det, Y_det, sci, wht, true_params, m, bg, ss_factor


def load_real_data(hdf5_path, ss_factor=1):
    """Load a real preprocessed cutout for benchmarking."""
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        sci = f['sci_bgsub_crop'][()]
        wht = f['wht_crop'][()]
    ny, nx = sci.shape
    X_det, Y_det, ext = build_arcsec_grid((ny, nx), pixscale=PIX_SCALE)

    if ss_factor > 1:
        X, Y, _ = build_arcsec_grid((ny*ss_factor, nx*ss_factor), pixscale=PIX_SCALE/ss_factor)
    else:
        X, Y = X_det, Y_det

    # Rough initial guess for real data
    true_params = dict(
        n_sersic=2.5, R_sersic=0.8, amplitude=0.03,
        q=0.7, theta_ell=0.0,
        a_m=np.array([0.01, 0.01]),
        phi_m=np.array([0.0, 0.0]),
        x0=0.0, y0=0.0,
        background=0.001,
    )
    m = [3, 4]
    bg = true_params['background']
    return X, Y, X_det, Y_det, sci, wht, true_params, m, bg, ss_factor


# ═══════════════════════════════════════════════════════════
#  Loss functions for SciPy minimize
# ═══════════════════════════════════════════════════════════

def build_loss_np(X, Y, sci, wht, m, ss_factor):
    """Non-JAX reduced-chi2 loss as a function of parameter vector."""
    k = len(m)
    ny, nx = sci.shape
    mask = np.isfinite(sci) & np.isfinite(wht) & (wht > 0)
    n_dof = np.sum(mask) - (5 + 2*k + 2 + 1)  # P params

    def loss(vec):
        p = unpack_params(vec, k)
        model_ss = make_model_np(X, Y, p, m, p.get('background', 0.001))
        if ss_factor > 1:
            model = downsample(model_ss, ss_factor)
        else:
            model = model_ss
        resid = (sci - model)
        chi2 = np.sum(wht[mask] * resid[mask]**2) / max(n_dof, 1)
        return float(chi2)

    return loss


def build_loss_jax(X_jax, Y_jax, sci_jax, wht_jax, m, ss_factor):
    """JAX reduced-chi2 loss, JIT compiled; returns (loss_fn, grad_fn)."""
    k = len(m)
    mask_np = np.isfinite(np.asarray(sci_jax)) & np.isfinite(np.asarray(wht_jax)) & (np.asarray(wht_jax) > 0)
    mask_jax = jnp.asarray(mask_np)
    n_dof = float(np.sum(mask_np) - (5 + 2*k + 2 + 1))

    m_jax = jnp.asarray(m)

    # JIT-safe sersic: no Python-level if-guards on traced values
    def _sersic_jit(R, amplitude=1.0, R_sersic=1.0, n_sersic=4.0):
        R = jnp.maximum(0.0001, jnp.asarray(R))
        R_sersic = jnp.maximum(1e-8, R_sersic)
        n_sersic = jnp.maximum(1e-8, n_sersic)
        bn = 1.999 * n_sersic - 0.327
        bn = jnp.maximum(bn, 1e-5)
        x = R / R_sersic
        logx = jnp.where(x > 0, jnp.log(x), -jnp.inf)
        pow_term = jnp.exp((1.0 / n_sersic) * logx)
        out = amplitude * jnp.exp(-bn * (pow_term - 1.0))
        return jnp.where(x >= 0, out, 0.0)

    @jit
    def _loss(vec):
        # Unpack inline (must use jnp for tracing)
        n_sersic_v = vec[0]
        R_sersic_v = vec[1]
        amplitude_v = vec[2]
        q_v = vec[3]
        theta_ell_v = vec[4]
        a_m_v = vec[5:5+k]
        phi_m_v = vec[5+k:5+2*k]
        x0_v = vec[5+2*k]
        y0_v = vec[5+2*k+1]
        bg_v = vec[5+2*k+2]

        I = EMP2D_jax(
            X_jax, Y_jax, _sersic_jit,
            q=q_v, theta_ell=theta_ell_v,
            m=m_jax, a_m=a_m_v, phi_m=phi_m_v,
            x0=x0_v, y0=y0_v,
            amplitude=amplitude_v, R_sersic=R_sersic_v, n_sersic=n_sersic_v
        )
        model = I + bg_v
        # Downsample if needed --- JIT-compatible block average 
        # (only works if ss_factor is static, which it is)
        if ss_factor > 1:
            ny_out = model.shape[0] // ss_factor
            nx_out = model.shape[1] // ss_factor
            model = model[:ny_out*ss_factor, :nx_out*ss_factor]
            model = model.reshape(ny_out, ss_factor, nx_out, ss_factor).mean(axis=(1, 3))

        resid = (sci_jax - model)
        chi2 = jnp.sum(jnp.where(mask_jax, wht_jax * resid**2, 0.0)) / n_dof
        return chi2

    _grad = jit(grad(_loss))
    _vg = jit(value_and_grad(_loss))

    # Wrappers that convert JAX arrays to float/numpy for SciPy
    def loss_fn(vec):
        return float(_loss(jnp.asarray(vec)))

    def grad_fn(vec):
        return np.asarray(_grad(jnp.asarray(vec)))

    def vg_fn(vec):
        v, g = _vg(jnp.asarray(vec))
        return float(v), np.asarray(g)

    return loss_fn, grad_fn, vg_fn


# ═══════════════════════════════════════════════════════════
#  Main benchmark
# ═══════════════════════════════════════════════════════════

def run_benchmark(args):
    ss = args.supersample
    image_sizes = [61, 101, 151]  # representative detector-pixel sizes

    for sz in image_sizes:
        print(f"\n{'='*60}")
        print(f"  Image size: {sz}×{sz}  |  Supersample: {ss}×")
        print(f"  Model grid: {sz*ss}×{sz*ss} = {(sz*ss)**2:,} pixels")
        print(f"{'='*60}")

        if args.hdf5:
            X, Y, X_det, Y_det, sci, wht, p0, m, bg, _ = load_real_data(args.hdf5, ss)
            sci = sci[:sz, :sz]
            wht = wht[:sz, :sz]
            X_det, Y_det, _ = build_arcsec_grid((sz, sz), pixscale=PIX_SCALE)
            if ss > 1:
                X, Y, _ = build_arcsec_grid((sz*ss, sz*ss), pixscale=PIX_SCALE/ss)
            else:
                X, Y = X_det, Y_det
        else:
            X, Y, X_det, Y_det, sci, wht, p0, m, bg, _ = make_mock_data((sz, sz), ss)

        k = len(m)
        v0 = pack_params(p0.copy(), k)

        # ── 1) Forward pass timing ──
        print("\n  [Forward model evaluation]")
        time_fn(lambda: make_model_np(X, Y, p0, m, bg), label="non-JAX  numpy", n_repeat=20)

        if HAS_JAX:
            X_j = jnp.asarray(X)
            Y_j = jnp.asarray(Y)
            # Un-JIT'd
            time_fn(lambda: make_model_jax(X_j, Y_j, p0, m, bg), label="JAX (not JIT)", n_repeat=20)
            # JIT'd
            _jit_model = jit(lambda Xj, Yj: make_model_jax(Xj, Yj, p0, m, bg))
            time_fn(lambda: _jit_model(X_j, Y_j), label="JAX (JIT)", n_repeat=20)

        # ── 2) Loss evaluation timing ──
        print("\n  [Loss evaluation (reduced χ²)]")
        loss_np = build_loss_np(X, Y, sci, wht, m, ss)
        time_fn(lambda: loss_np(v0), label="non-JAX loss", n_repeat=20)

        if HAS_JAX:
            sci_j = jnp.asarray(sci)
            wht_j = jnp.asarray(wht)
            loss_jax_fn, grad_jax_fn, vg_jax_fn = build_loss_jax(X_j, Y_j, sci_j, wht_j, m, ss)
            time_fn(lambda: loss_jax_fn(v0), label="JAX loss (JIT)", n_repeat=20)
            time_fn(lambda: grad_jax_fn(v0), label="JAX grad (JIT)", n_repeat=20)
            time_fn(lambda: vg_jax_fn(v0), label="JAX val+grad (JIT)", n_repeat=20)

        # ── 3) Full optimization timing ──
        print("\n  [Full optimization (max 200 iter)]")
        lo, hi = _get_bounds(v0, k)
        bounds_scipy = Bounds(lo, hi)

        # SLSQP (non-JAX)
        t0 = time.perf_counter()
        res_np = minimize(loss_np, v0, method='SLSQP', bounds=bounds_scipy,
                          options={'maxiter': 200, 'ftol': 1e-12})
        t_slsqp_np = time.perf_counter() - t0
        print(f"  {'SLSQP (non-JAX)':40s}  {t_slsqp_np:8.2f} s  loss={res_np.fun:.4f}  nfev={res_np.nfev}")

        # L-BFGS-B (non-JAX, finite diff)
        t0 = time.perf_counter()
        res_lbfgs_np = minimize(loss_np, v0, method='L-BFGS-B', bounds=bounds_scipy,
                                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        t_lbfgs_np = time.perf_counter() - t0
        print(f"  {'L-BFGS-B (non-JAX, finite-diff)':40s}  {t_lbfgs_np:8.2f} s  loss={res_lbfgs_np.fun:.4f}  nfev={res_lbfgs_np.nfev}")

        if HAS_JAX:
            # SLSQP (JAX loss, finite-diff grad)
            t0 = time.perf_counter()
            res_jax_slsqp = minimize(loss_jax_fn, v0, method='SLSQP', bounds=bounds_scipy,
                                     options={'maxiter': 200, 'ftol': 1e-12})
            t_slsqp_jax = time.perf_counter() - t0
            print(f"  {'SLSQP (JAX loss)':40s}  {t_slsqp_jax:8.2f} s  loss={res_jax_slsqp.fun:.4f}  nfev={res_jax_slsqp.nfev}")

            # L-BFGS-B (JAX loss + JAX grad)
            t0 = time.perf_counter()
            res_jax_lbfgs = minimize(loss_jax_fn, v0, method='L-BFGS-B', jac=grad_jax_fn,
                                     bounds=bounds_scipy,
                                     options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
            t_lbfgs_jax = time.perf_counter() - t0
            print(f"  {'L-BFGS-B (JAX loss + JAX grad)':40s}  {t_lbfgs_jax:8.2f} s  loss={res_jax_lbfgs.fun:.4f}  nfev={res_jax_lbfgs.nfev}")

    print(f"\n{'='*60}")
    print("  Benchmark complete")
    print(f"{'='*60}")


def _get_bounds(v0, k):
    """Simple bounds matching tools_fitting.default_bounds."""
    n = len(v0)
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    # n_sersic ∈ [0.5, 8]
    lo[0] = 0.5; hi[0] = 8.0
    # R_sersic ∈ [0.01, 10]
    lo[1] = 0.01; hi[1] = 10.0
    # amplitude ∈ [1e-6, 1]
    lo[2] = 1e-6; hi[2] = 1.0
    # q ∈ [0.1, 1]
    lo[3] = 0.1; hi[3] = 1.0
    # theta_ell ∈ [-π, π]
    lo[4] = -np.pi; hi[4] = np.pi
    # a_m ∈ [-0.3, 0.3]
    for i in range(k):
        lo[5+i] = -0.3; hi[5+i] = 0.3
    # phi_m ∈ [-π/(2*m), π/(2*m)] — use conservative ±π/4
    for i in range(k):
        lo[5+k+i] = -np.pi/4; hi[5+k+i] = np.pi/4
    # x0, y0 ∈ [-1, 1]
    lo[5+2*k] = -1.0; hi[5+2*k] = 1.0
    lo[5+2*k+1] = -1.0; hi[5+2*k+1] = 1.0
    # background ∈ [-0.1, 0.1]
    lo[5+2*k+2] = -0.1; hi[5+2*k+2] = 0.1
    return lo, hi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark JAX vs non-JAX fitting speed")
    parser.add_argument("--hdf5", type=str, default=None,
                        help="Path to a real *-cropped.hdf5 cutout (optional)")
    parser.add_argument("--supersample", type=int, default=3,
                        help="Supersampling factor (default: 3)")
    args = parser.parse_args()
    run_benchmark(args)
