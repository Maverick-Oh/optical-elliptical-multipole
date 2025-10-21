from optical_elliptical_multipole.nonjax.tools import XY2RTHETA, theta2phi, circularize, remove_multipole, remove_circular_multipole_and_circularize

def Circular_Profile_2D(X, Y, intensity_fun, x0=0., y0=0., **intensity_fun_kwargs):
    # circular profile without any elliptical or multipole
    X = X.copy() - x0
    Y = Y.copy() - y0
    R, THETA = XY2RTHETA(X, Y)
    return intensity_fun(R, **intensity_fun_kwargs)

def Elliptical_Multipole_Profile_2D(X, Y, intensity_fun, q, theta_ell, m, a_m, phi_m=None, theta_m=None, x0=0., y0=0.,
                                   **intensity_fun_kwargs):
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
    X = X.copy() - x0
    Y = Y.copy() - y0
    R, THETA = XY2RTHETA(X, Y)
    THETA -= theta_ell
    R_, PHI = circularize(R, THETA, q)
    R_C = remove_multipole(R_, PHI, m, a_m, phi_m)
    return intensity_fun(R_C, **intensity_fun_kwargs)

def Circular_Multipole_Profile_2D(X, Y, intensity_fun, q, theta_ell, m, a_m, theta_m=None, x0=0., y0=0.,
                                  verbose=True, **intensity_fun_kwargs):
    if theta_m is None:
        raise ValueError("theta_m must be given!")
    X = X.copy() - x0
    Y = Y.copy() - y0
    R, THETA = XY2RTHETA(X, Y)
    THETA -= theta_ell
    R_C_ = remove_circular_multipole_and_circularize(R, THETA, q, m, a_m, theta_m)
    return intensity_fun(R_C_, **intensity_fun_kwargs)