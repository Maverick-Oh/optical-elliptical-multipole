
"""
Core geometric/multipole utilities shared by 1D/2D profiles.
Vectorized where possible; numerical kernels annotated with Numba @njit
when available (falls back to no-op decorator). Public wrappers coerce inputs,
so lists/tuples work too.
"""
import numpy as np

try:
    from numba import njit
except Exception:  # Numba not available; define a no-op decorator
    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap

# -------------------- Multipole kernels (vectorized sums) --------------------
@njit(cache=True)
def _cos_arg(m, ang, phi_m):
    k = m.size
    n = ang.size
    out = np.empty((k, n))
    for i in range(k):
        mi = m[i]
        phi_i = phi_m[i]
        for j in range(n):
            out[i, j] = mi * (ang[j] - phi_i)
    return out

def _broadcast_sum_cos(m, a_m, ang, phi_m):
    """
    Sum_k a_m[k] * cos(m[k] * (ang - phi_m[k])) with broadcasting.
    - m, a_m, phi_m: 1-D (or scalars)
    - ang: arbitrary shape (...), e.g. (101, 101)
    Returns an array with the same shape as `ang`.
    """
    m     = np.atleast_1d(np.asarray(m, dtype=float))
    a_m   = np.atleast_1d(np.asarray(a_m, dtype=float))
    phi_m = np.atleast_1d(np.asarray(phi_m, dtype=float))
    ang   = np.asarray(ang, dtype=float)

    # Broadcast mode arrays to shape (K, 1, 1, ..., 1) to match ang.ndim
    expand = (slice(None),) + (None,) * ang.ndim
    m_b     = m[expand]
    a_b     = a_m[expand]
    phi_b   = phi_m[expand]

    # angles has shape (K, *ang.shape); sum over K (axis=0) -> ang.shape
    angles = m_b * (ang - phi_b)
    return (a_b * np.cos(angles)).sum(axis=0)

# -------------------- Public multipole utilities --------------------
def _single_multipole_factor(r, phi, m, a_m, phi_m):
    """Fractional perturbation a_m * cos(m*(phi - phi_m)) for a single mode."""
    if callable(a_m) or callable(phi_m):
        raise NotImplementedError("Callable amplitudes/phases are not implemented yet.")
    return a_m * np.cos(m * (phi - phi_m))

def add_multipole(r, phi, m, a_m, phi_m):
    """
    Apply elliptical multipole(s) in eccentric anomaly phi to radius r:
    r * (1 + sum_k a_m[k] * cos(m[k]*(phi - phi_m[k]))).
    """
    if callable(a_m) or callable(phi_m):
        raise NotImplementedError("Callable amplitudes/phases are not implemented yet.")
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)  # keep original shape
    factor = _broadcast_sum_cos(m, a_m, phi, phi_m)
    return r * (1.0 + factor)

# -------------------- Ellipse geometry (core + Python wrappers) --------------------
@njit(cache=True)
def _ellipticize_factor_simple_core(q, theta):
    return 1.0 / np.sqrt(q * np.cos(theta)**2 + (np.sin(theta)**2) / q)

def ellipticize_factor_simple(q, theta):
    """
    Geometric factor mapping circle -> ellipse:
    f(theta) = 1 / sqrt(q*cos^2(theta) + sin^2(theta)/q).

    Accepts scalars, lists, tuples, or ndarrays.
    """
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    theta_arr = np.asarray(theta, dtype=float)
    return _ellipticize_factor_simple_core(float(q), theta_arr)

@njit(cache=True)
def _phi2theta_core(phi, q):
    return np.arctan2(q * np.sin(phi), np.cos(phi))

def phi2theta(phi, q):
    """
    Eccentric anomaly (phi) -> parametric/central angle (theta).
    Accepts scalars, lists, tuples, or ndarrays.
    """
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    phi_arr = np.asarray(phi, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    return _phi2theta_core(phi_arr, q_arr)

@njit(cache=True)
def _theta2phi_core(theta, q):
    return np.arctan2(np.sin(theta) / q, np.cos(theta))

def theta2phi(theta, q):
    """
    Parametric/central angle (theta) -> eccentric anomaly (phi).
    Accepts scalars, lists, tuples, or ndarrays.
    """
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    theta_arr = np.asarray(theta, dtype=float)
    return _theta2phi_core(theta_arr, float(q))

def ellipticize(r, phi, q):
    """Map circular (r, phi) to elliptical (r_new, theta)."""
    r = np.asarray(r, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    theta = _phi2theta_core(phi_arr, float(q))
    return r * _ellipticize_factor_simple_core(float(q), theta), theta

# -------------------- Circular multipoles (in theta) --------------------
def _delta_unit_single_circular_multipole(theta, m, a_m, theta_m):
    if callable(a_m) or callable(theta_m):
        raise NotImplementedError("Callable amplitudes/phases are not implemented yet.")
    return a_m * np.cos(m * (theta - theta_m))

def delta_unit_circular_multipole(theta, m, a_m, theta_m):
    if callable(a_m) or callable(theta_m):
        raise NotImplementedError("Callable amplitudes/phases are not implemented yet.")
    theta = np.asarray(theta, dtype=float)  # preserve 2D grid shape
    return _broadcast_sum_cos(m, a_m, theta, theta_m)

def delta_circular_multipole(r0, theta, m, a_m, theta_m):
    r0 = float(r0)
    return r0 * delta_unit_circular_multipole(theta, m, a_m, theta_m)

def circularize(r, theta, q):
    """Map ellipse (r, theta) to circle (r_new, phi): inverse of ellipticize."""
    r = np.asarray(r, dtype=float)
    theta_arr = np.asarray(theta, dtype=float)
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    phi = _theta2phi_core(theta_arr, float(q))
    return r / _ellipticize_factor_simple_core(float(q), theta_arr), phi

def remove_multipole(r, phi, m, a_m, phi_m):
    """Remove elliptical multipole(s) from r expressed in phi."""
    if callable(a_m) or callable(phi_m):
        raise NotImplementedError("Callable amplitudes/phases are not implemented yet.")
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)  # keep original shape
    denom = 1.0 + _broadcast_sum_cos(m, a_m, phi, phi_m)
    return r / denom

def remove_circular_multipole_and_circularize(r, theta, q, m, a_m, theta_m):
    """
    Invert the circular-multipole construction and elliptic factor simultaneously.

    r0 = r / ( f_ell(theta) + (1/sqrt(q)) * delta_unit(theta) )
    """
    if callable(a_m) or callable(theta_m) or callable(q):
        raise NotImplementedError("Callable amplitudes/phases/q are not implemented yet.")
    r = np.asarray(r, dtype=float)
    theta_arr = phi = np.asarray(theta, dtype=float)  # keep original shape
    f_ell = _ellipticize_factor_simple_core(float(q), theta_arr)
    delta = delta_unit_circular_multipole(theta_arr, m, a_m, theta_m)
    denom = f_ell + (1.0 / np.sqrt(float(q))) * delta
    return r / denom

# -------------------- Cartesian <-> Polar --------------------
def XY2RTHETA(X, Y):
    """Cartesian -> polar (R, theta)."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    R = np.hypot(X, Y)
    THETA = np.arctan2(Y, X)
    return R, THETA

def RTHETA2XY(R, THETA):
    """Polar -> Cartesian (X, Y)."""
    R = np.asarray(R, dtype=float)
    THETA = np.asarray(THETA, dtype=float)
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    return X, Y

# -------------------- Angle grid helpers --------------------
def angle_like_r(r, include_end=True, n_points=100):
    """
    Return an angle grid theta of length len(r) over [0, 2π].
    If r is scalar (or 0-d array), expand it to length n_points and
    return the expanded r alongside theta.

    Returns
    -------
    theta : ndarray
    r_vec : ndarray
        Possibly-expanded r vector matching theta length.
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 0:
        r = np.full((n_points,), r.item(), dtype=float)
    n = r.size
    if n < 2:
        raise ValueError("Need at least 2 samples to build an angle grid.")
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=include_end)
    return theta, r

def amplitude_angle_wrapper(amp, phi, m):
    # Adjusting angles that go over ( -pi/(2*m), pi/(2*m) )
    m = int(m)
    phi_ = phi + np.pi/2./m
    factor, phi_ = np.divmod(phi_, np.pi/m)
    phi_ +=  - np.pi/2./m
    amp_ = amp.copy()
    amp_[factor%2==1] *= -1
    return amp, phi_
