import numpy as np
from optical_elliptical_multipole.nonjax.tools import (RTHETA2XY, theta2phi, ellipticize,
                                                       angle_like_r, add_multipole, delta_circular_multipole)

def Circular_Profile_1D(r0, return_type='xy', x0=0., y0=0., include_end=True):
    # circular profile without any elliptical or multipole
    theta = angle_like_r(r0, include_end=include_end)
    x, y = RTHETA2XY(r0, theta)
    x += x0
    y += y0
    if return_type == 'xy':
        return (x,y)
    elif return_type == 'polar':
        r_translated = np.sqrt(x**2 + y**2)
        theta_translated = np.arctan2(y, x)
        return r_translated, theta_translated
    else:
        raise ValueError(f"return_type must be 'xy' or 'polar'! Currently it is: {return_type}")

def Elliptical_Multipole_Profile_1D(r0, q, theta_ell, m, a_m, phi_m=None, theta_m=None, x0=0., y0=0., return_type='xy', include_end=True):
    # theta_ell: the angle of the ellipse itself
    if phi_m is None and theta_m is None:
        raise ValueError("Either phi_m or theta_m must be given!")
    elif phi_m is None and theta_m is not None:
        print("theta_m given, not phi_m; converting theta_m to phi_m...")
        phi_m = theta2phi(theta_m, q)
    elif phi_m is not None and theta_m is None:
        print("phi_m given!")
    else:
        raise ValueError("One of phi_m and theta_m should be given; currently both are given.")
    phi = angle_like_r(r0, include_end=include_end)
    r_multipole = add_multipole(r0, phi, m=m, a_m=a_m, phi_m=phi_m)
    r_EM_multiple, theta_EM_multiple = ellipticize(r_multipole, phi, q)
    theta_EM_multiple_rotated = theta_EM_multiple + theta_ell
    x, y = RTHETA2XY(r_EM_multiple, theta_EM_multiple_rotated)
    x += x0
    y += y0
    if return_type == 'xy':
        return x, y
    elif return_type == 'polar':
        r_translated = np.sqrt(x**2 + y**2)
        theta_translated = np.arctan2(y, x)
        return r_translated, theta_translated
    else:
        raise ValueError(f"return_type must be 'xy' or 'polar'! Currently it is: {return_type}")

def Circular_Multipole_Profile_1D(r0, q, theta_ell, m, a_m, theta_m=None, x0=0., y0=0., return_type='xy', include_end=True):
    if theta_m is None:
        raise ValueError("theta_m must be given!")
    phi = angle_like_r(r0, include_end=include_end)
    r_ell, theta = ellipticize(r0, phi, q)
    r_CM = r_ell + 1 / np.sqrt(q) * delta_circular_multipole(r0, theta, m=m, a_m=a_m, theta_m=theta_m)
    theta_CM_rotated = theta + theta_ell
    x, y = RTHETA2XY(r_CM, theta_CM_rotated)
    x += x0
    y += y0
    if return_type == 'xy':
        return x, y
    elif return_type == 'polar':
        r_translated = np.sqrt(x**2 + y**2)
        theta_translated = np.arctan2(y, x)
        return r_translated, theta_translated
    else:
        raise ValueError(f"return_type must be 'xy' or 'polar'! Currently it is: {return_type}")
