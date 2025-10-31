
"""
2D profiles: evaluate an intensity function on a grid (X,Y) after
undoing multipole constructions to recover the base circular radius.
"""
import numpy as np

from optical_elliptical_multipole.nonjax.tools import (
    XY2RTHETA, circularize, remove_multipole,
    remove_circular_multipole_and_circularize, theta2phi, ellipticize_factor_simple
)

def Circular_Profile_2D(X, Y, intensity_fun,
                        x0=0.0, y0=0.0, **intensity_fun_kwargs):
    """
    Evaluate intensity_fun at circular radius centered at (x0,y0).
    No ellipticity or multipoles.
    """
    R, _TH = XY2RTHETA(X - x0, Y - y0)
    return intensity_fun(R, **intensity_fun_kwargs)

def Elliptical_Profile_2D(X, Y, intensity_fun, q, theta_ell,
                          x0=0.0, y0=0.0, **intensity_fun_kwargs):
    """
    Evaluate intensity_fun at circular radius centered at (x0,y0).
    No ellipticity or multipoles.
    """
    R, THETA = XY2RTHETA(X - x0, Y - y0)
    THETA = THETA - theta_ell
    R_circ, PHI = circularize(R, THETA, q)
    return intensity_fun(R_circ, **intensity_fun_kwargs)

def Elliptical_Multipole_Profile_2D(
    X, Y, intensity_fun, q, theta_ell, m, a_m, phi_m,
        x0=0.0, y0=0.0, **intensity_fun_kwargs
):
    """
    Evaluate intensity with an **elliptical multipole** removed.
    Steps (inverse of 1D construction):
      1) Translate/rotate pixels by (-x0,-y0) and -theta_ell.
      2) Circularize (R, theta) -> (R_circ, phi) via q.
      3) Remove elliptical multipole(s) in phi to recover base circular R_core.
      4) Evaluate intensity_fun(R_core).
    """
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    R, THETA = XY2RTHETA(X - x0, Y - y0)
    THETA = THETA - theta_ell
    R_circ, PHI = circularize(R, THETA, q) # Circle with multipoles
    R_core = remove_multipole(R_circ, PHI, m, a_m, phi_m)
    return intensity_fun(R_core, **intensity_fun_kwargs)

def Circular_Multipole_Profile_2D(
    X, Y, intensity_fun, q, theta_ell, m, a_m, theta_m,
        x0=0.0, y0=0.0, **intensity_fun_kwargs
):
    """
    Evaluate intensity with a **circular multipole** removed along with geometric factor.
    Steps (inverse of 1D circular multipole construction):
      1) Translate/rotate pixels by (-x0,-y0) and -theta_ell.
      2) Compute r0 = r / ( f_ell(theta) + (1/sqrt(q)) * delta_unit(theta) ).
      3) Evaluate intensity_fun(r0).
    """
    if callable(q):
        raise NotImplementedError("Callable q is not implemented yet.")
    R, THETA = XY2RTHETA(X - x0, Y - y0)
    THETA = THETA - theta_ell
    R_core = remove_circular_multipole_and_circularize(R, THETA, q, m, a_m, theta_m)
    return intensity_fun(R_core, **intensity_fun_kwargs)
