
"""
Core geometric/multipole utilities shared by 1D/2D profiles.
Vectorized where possible; compatible with JAX.
"""
import jax.numpy as jnp
from jax import jit

# -------------------- Multipole kernels (vectorized sums) --------------------
# JAX handles vectorization via vmap or broadcasting. 
# The original code used manual loops in _cos_arg for numba.
# We can use broadcasting here which is more efficient in JAX/NumPy.

def _broadcast_sum_cos(m, a_m, ang, phi_m):
    """
    Sum_k a_m[k] * cos(m[k] * (ang - phi_m[k])) with broadcasting.
    - m, a_m, phi_m: 1-D (or scalars)
    - ang: arbitrary shape (...), e.g. (101, 101)
    Returns an array with the same shape as `ang`.
    """
    m     = jnp.atleast_1d(jnp.asarray(m, dtype=float))
    a_m   = jnp.atleast_1d(jnp.asarray(a_m, dtype=float))
    phi_m = jnp.atleast_1d(jnp.asarray(phi_m, dtype=float))
    ang   = jnp.asarray(ang, dtype=float)

    # Broadcast mode arrays to shape (K, 1, 1, ..., 1) to match ang.ndim
    # JAX doesn't support tuple concatenation with + like (slice(None),) + (None,)*ndim inside indexing directly cleanly sometimes?
    # Actually Python tuple addition works fine.
    expand = (slice(None),) + (None,) * ang.ndim
    m_b     = m[expand]
    a_b     = a_m[expand]
    phi_b   = phi_m[expand]

    # angles has shape (K, *ang.shape); sum over K (axis=0) -> ang.shape
    angles = m_b * (ang - phi_b)
    return (a_b * jnp.cos(angles)).sum(axis=0)

# -------------------- Public multipole utilities --------------------
def _single_multipole_factor(r, phi, m, a_m, phi_m):
    """Fractional perturbation a_m * cos(m*(phi - phi_m)) for a single mode."""
    # JAX supports callables via jit/grad, but for data structures we assume arrays usually.
    return a_m * jnp.cos(m * (phi - phi_m))

def add_multipole(r, phi, m, a_m, phi_m):
    """
    Apply elliptical multipole(s) in eccentric anomaly phi to radius r:
    r * (1 + sum_k a_m[k] * cos(m[k]*(phi - phi_m[k]))).
    """
    r = jnp.asarray(r, dtype=float)
    phi = jnp.asarray(phi, dtype=float)  # keep original shape
    factor = _broadcast_sum_cos(m, a_m, phi, phi_m)
    return r * (1.0 + factor)

# -------------------- Ellipse geometry (core + Python wrappers) --------------------
def _ellipticize_factor_simple_core(q, theta):
    return 1.0 / jnp.sqrt(q * jnp.cos(theta)**2 + (jnp.sin(theta)**2) / q)

def ellipticize_factor_simple(q, theta):
    """
    Geometric factor mapping circle -> ellipse:
    f(theta) = 1 / sqrt(q*cos^2(theta) + sin^2(theta)/q).

    Accepts scalars, lists, tuples, or ndarrays.
    """
    theta_arr = jnp.asarray(theta, dtype=float)
    return _ellipticize_factor_simple_core(q, theta_arr)

def _phi2theta_core(phi, q):
    return jnp.arctan2(q * jnp.sin(phi), jnp.cos(phi))

def phi2theta(phi, q):
    """
    Eccentric anomaly (phi) -> parametric/central angle (theta).
    Accepts scalars, lists, tuples, or ndarrays.
    """
    phi_arr = jnp.asarray(phi, dtype=float)
    q_arr = jnp.asarray(q, dtype=float)
    return _phi2theta_core(phi_arr, q_arr)

def _theta2phi_core(theta, q):
    return jnp.arctan2(jnp.sin(theta) / q, jnp.cos(theta))

def theta2phi(theta, q):
    """
    Parametric/central angle (theta) -> eccentric anomaly (phi).
    Accepts scalars, lists, tuples, or ndarrays.
    """
    theta_arr = jnp.asarray(theta, dtype=float)
    return _theta2phi_core(theta_arr, q)

def ellipticize(r, phi, q):
    """Map circular (r, phi) to elliptical (r_new, theta)."""
    r = jnp.asarray(r, dtype=float)
    phi_arr = jnp.asarray(phi, dtype=float)
    theta = _phi2theta_core(phi_arr, q)
    return r * _ellipticize_factor_simple_core(q, theta), theta

# -------------------- Circular multipoles (in theta) --------------------
def _delta_unit_single_circular_multipole(theta, m, a_m, theta_m):
    return a_m * jnp.cos(m * (theta - theta_m))

def delta_unit_circular_multipole(theta, m, a_m, theta_m):
    theta = jnp.asarray(theta, dtype=float)  # preserve 2D grid shape
    return _broadcast_sum_cos(m, a_m, theta, theta_m)

def delta_circular_multipole(r0, theta, m, a_m, theta_m):
    # r0 = float(r0) # Remov float cast for JAX
    return r0 * delta_unit_circular_multipole(theta, m, a_m, theta_m)

def circularize(r, theta, q):
    """Map ellipse (r, theta) to circle (r_new, phi): inverse of ellipticize."""
    r = jnp.asarray(r, dtype=float)
    theta_arr = jnp.asarray(theta, dtype=float)
    phi = _theta2phi_core(theta_arr, q)
    
    # Check for zero factor skipped for JAX compatibility
    # if (_ellipticize_factor_simple_core(q, theta_arr)==0.).any():
    #     raise ValueError("Ellipticize factor cannot be computed.")
    
    my_return = r / _ellipticize_factor_simple_core(q, theta_arr), phi
    return my_return

def remove_multipole(r, phi, m, a_m, phi_m):
    """Remove elliptical multipole(s) from r expressed in phi."""
    r = jnp.asarray(r, dtype=float)
    phi = jnp.asarray(phi, dtype=float)  # keep original shape
    denom = 1.0 + _broadcast_sum_cos(m, a_m, phi, phi_m)
    return r / denom

def remove_circular_multipole_and_circularize(r, theta, q, m, a_m, theta_m):
    """
    Invert the circular-multipole construction and elliptic factor simultaneously.

    r0 = r / ( f_ell(theta) + (1/sqrt(q)) * delta_unit(theta) )
    """
    r = jnp.asarray(r, dtype=float)
    theta_arr = np = jnp.asarray(theta, dtype=float)  # keep original shape
    f_ell = _ellipticize_factor_simple_core(q, theta_arr)
    delta = delta_unit_circular_multipole(theta_arr, m, a_m, theta_m)
    denom = f_ell + (1.0 / jnp.sqrt(q)) * delta
    return r / denom

# -------------------- Cartesian <-> Polar --------------------
def XY2RTHETA(X, Y):
    """Cartesian -> polar (R, theta)."""
    X = jnp.asarray(X, dtype=float)
    Y = jnp.asarray(Y, dtype=float)
    R = jnp.hypot(X, Y)
    THETA = jnp.arctan2(Y, X)
    return R, THETA

def RTHETA2XY(R, THETA):
    """Polar -> Cartesian (X, Y)."""
    R = jnp.asarray(R, dtype=float)
    THETA = jnp.asarray(THETA, dtype=float)
    X = R * jnp.cos(THETA)
    Y = R * jnp.sin(THETA)
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
    r = jnp.asarray(r, dtype=float)
    if r.ndim == 0:
        r = jnp.full((n_points,), r, dtype=float)
    n = r.size
    # JAX array size is static usually, but here we can check if it's concrete
    # If using JIT, n must be static or we use other ways.
    # Assuming this is called outside JIT mainly for setup, or with static shapes.
    
    # if n < 2:
    #     raise ValueError("Need at least 2 samples to build an angle grid.")
    
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=include_end)
    return theta, r

def amplitude_angle_wrapper(amp, phi, m):
    # Adjusting angles that go over ( -pi/(2*m), pi/(2*m) )
    # This function uses logic that assumes modifiable arrays or control flow based on values.
    # We implement it in JAX style.
    
    # m = int(m) # Assume m is scalar
    
    # phi_ = phi + np.pi/2./m
    phi_ = phi + jnp.pi/2.0/m
    
    # factor, phi_ = np.divmod(phi_, np.pi/m)
    factor = jnp.floor(phi_ / (jnp.pi/m))
    phi_ = phi_ - factor * (jnp.pi/m)
    
    phi_ +=  - jnp.pi/2.0/m
    
    # amp_[factor%2==1] *= -1
    # usage of where
    amp_ = jnp.where(factor % 2 == 1, -amp, amp)
    
    return amp_, phi_
