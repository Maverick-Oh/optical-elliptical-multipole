
"""
1D profiles: generate contour samples (x,y) or (r,theta) for circular/elliptical
multipole constructions. These functions accept scalar r0 and expand to vectors.
"""
import numpy as np
from optical_elliptical_multipole.nonjax.tools import (
    add_multipole, XY2RTHETA,
    angle_like_r, RTHETA2XY,
    delta_unit_circular_multipole, ellipticize_factor_simple, phi2theta, theta2phi
)

def _resolve_phi_m(phi_m, theta_m, q):
    """Return phi_m, converting from theta_m if needed; enforce exactly one provided."""
    if (phi_m is None) == (theta_m is None):
        raise ValueError("Provide exactly one of phi_m or theta_m.")
    if phi_m is not None:
        return np.asarray(phi_m, dtype=float)
    else:
        return np.asarray(theta2phi(theta_m, q), dtype=float)

def _format_output(r, theta, return_type='xy', x0=0., y0=0.):
    """Return (x,y) for return_type='xy' else (r, theta) after translation."""
    x, y = RTHETA2XY(r, theta)
    x += x0
    y += y0
    if return_type == 'xy':
        return x, y
    elif return_type == 'theta':
        r, theta = XY2RTHETA(x, y)
        return r, theta
    else:
        raise ValueError("return_type must be 'xy' or 'polar'.")

def Circular_Profile_1D(r0=1.0, return_type='xy', x0=0.0, y0=0.0, include_end=True, n_points=100):
    """
    Parametric circle contour with center translation.
    If r0 is scalar, expand to length n_points.
    """
    theta, r_vec = angle_like_r(r0, include_end=include_end, n_points=n_points)
    # Pure circle: r(theta) = r0
    x, y = RTHETA2XY(r_vec, theta)
    return (x + x0, y + y0) if return_type == 'xy' else (r_vec, theta)


def Elliptical_Profile_1D(
    r0=None, q=None, theta_ell=None,
    x0=0.0, y0=0.0, return_type='xy', include_end=True, n_points=100
):
    """
    1D contour with **elliptical multipole** added in eccentric anomaly (phi),
    then mapped circle -> ellipse and rotated by theta_ell.

    Construction (matches original behavior):
      1) Start on circle with radius r0(θ) (scalar expands to length n_points).
      2) Add multipole(s) in phi (where on the circle phi == theta).
      3) Convert phi -> theta_ellipse via q and apply geometric scale
         f(theta_ellipse) = 1 / sqrt(q*cos^2 + sin^2/q).
      4) Rotate by theta_ell and translate (x0, y0).
    """
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    # Angle grid + (possibly expanded) radius
    phi, r_vec = angle_like_r(r0, include_end=include_end, n_points=n_points)

    theta_ellipse = phi2theta(phi, q)
    r_ell = r_vec * ellipticize_factor_simple(q, theta_ellipse)

    # 4) rotate + format output
    theta_out = theta_ellipse + theta_ell
    out = _format_output(r_ell, theta_out, return_type=return_type, x0=x0, y0=y0)
    return out

def Elliptical_Multipole_Profile_1D(
    r0=None, q=None, theta_ell=None, m=None, a_m=None, phi_m=None,
    x0=0.0, y0=0.0, return_type='xy', include_end=True, n_points=100
):
    """
    1D contour with **elliptical multipole** added in eccentric anomaly (phi),
    then mapped circle -> ellipse and rotated by theta_ell.

    Construction (matches original behavior):
      1) Start on circle with radius r0(θ) (scalar expands to length n_points).
      2) Add multipole(s) in phi (where on the circle phi == theta).
      3) Convert phi -> theta_ellipse via q and apply geometric scale
         f(theta_ellipse) = 1 / sqrt(q*cos^2 + sin^2/q).
      4) Rotate by theta_ell and translate (x0, y0).
    """
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")

    # Angle grid + (possibly expanded) radius
    phi, r_vec = angle_like_r(r0, include_end=include_end, n_points=n_points)

    # 1) + 2) add elliptical multipole in phi (on the circle)
    r_circ = add_multipole(r_vec, phi, m, a_m, phi_m)

    # 3) map circle -> ellipse: convert angle and apply geometric factor
    theta_ellipse = phi2theta(phi, q)
    r_ell = r_circ * ellipticize_factor_simple(q, theta_ellipse)

    # 4) rotate + format output
    theta_out = theta_ellipse + theta_ell
    out = _format_output(r_ell, theta_out, return_type=return_type, x0=x0, y0=y0)
    return out


def Circular_Multipole_Profile_1D(
    r0=None, q=None, theta_ell=None, m=None, a_m=None, theta_m=None,
    x0=0.0, y0=0.0, return_type='xy', include_end=True, n_points=100
):
    """
    1D contour with **circular multipole** added in central angle theta,
    then geometric ellipse mapping/rotation.
    Construction:
        r(theta) = r0 * ( f_ell(theta) + (1/sqrt(q)) * delta_unit(theta) )
    Here we build the contour directly in the rotated frame.
    """
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    theta, r_vec = angle_like_r(r0, include_end=include_end, n_points=n_points)
    # Build the geometric ellipse factor and delta on the unrotated theta
    f_ell = ellipticize_factor_simple(q, theta)
    delta = delta_unit_circular_multipole(theta, m, a_m, theta_m)
    r = r_vec * (f_ell + (1.0 / np.sqrt(q)) * delta)
    # Rotate by theta_ell and translate
    theta_rot = theta + theta_ell
    out = _format_output(r, theta_rot, return_type=return_type)
    if return_type == 'xy':
        x, y = out
        return x + x0, y + y0
    return out
