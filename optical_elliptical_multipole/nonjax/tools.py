import numpy as np
import warnings

def _single_multipole_factor(r, phi, m, a_m, phi_m):
    if callable(a_m) or callable(phi_m):
        if callable(a_m) and callable(phi_m):
            # in case a_m and phi_m are functions
            # r_new = r * (1 + a_m(r) * np.cos(m * (phi - phi_m(r) ) ) )
            raise NotImplementedError()
        else:
            raise ValueError("(a_m, phi_m) cannot be (function, number) or (number, function). They should be both functions or both numbers.")
    else:
        factor = a_m * np.cos(m * (phi - phi_m))
    return factor

def add_multipole(r, phi, m, a_m, phi_m):
    m = np.array([m]).reshape(-1)
    a_m = np.array([a_m]).reshape(-1)
    phi_m = np.array([phi_m]).reshape(-1)
    assert len(a_m) == len(m) == len(phi_m)
    if m.size == 1:
        r_new = r.copy() * ( 1 + _single_multipole_factor(r, phi, m[0], a_m[0], phi_m[0]) )
    else:
        factor = np.zeros_like(r)
        for i in range(len(m)):
            factor += _single_multipole_factor(r, phi, m[i], a_m[i], phi_m[i])
        r_new = r.copy() * (1 + factor)
    return r_new

def ellipticize_factor_simple(q, theta):
    return 1/np.sqrt( q * np.cos(theta)**2 + np.sin(theta)**2 / q)

def phi2theta(phi, q):
    if type(phi)==list:
        warnings.warn("phi is given as list and being converted to numpy array!")
        phi = np.asarray(phi).reshape(-1)
    theta = np.arctan2(q * np.sin(phi), np.cos(phi))
    return theta

def theta2phi(theta, q):
    if type(theta)==list:
        warnings.warn("theta is given as list and being converted to numpy array!")
        theta = np.asarray(theta).reshape(-1)
    phi = np.arctan2(np.sin(theta)/q, np.cos(theta))
    return phi

def ellipticize(r, phi, q):
    theta = phi2theta(phi, q)
    r_new = r * ellipticize_factor_simple(q,theta)
    return r_new, theta

def _delta_unit_single_circular_multipole(theta, m, a_m, theta_m):
    # for a unit circle; needs to be scaled with r0/sqrt(q)
    if callable(a_m) or callable(theta_m):
        if callable(a_m) and callable(theta_m):
            # in case a_m and theta_m are functions
            # delta_r = r0 * a_m(r0) * np.cos(m * (theta(r) - theta_m))
            raise NotImplementedError("Not Implemented Yet")
        else:
            raise ValueError("(a_m, phi_m) must both be callables OR both be scalars; mixed (callable, scalar) is not supported.")
    else:
        delta_r = a_m * np.cos(m * (theta - theta_m))
    return delta_r

def delta_unit_circular_multipole(theta, m, a_m, theta_m):
    # for a unit circle; needs to be scaled with r0
    m = np.array([m]).reshape(-1)
    a_m = np.array([a_m]).reshape(-1)
    theta_m = np.array([theta_m]).reshape(-1)
    assert len(a_m) == len(m) == len(theta_m)
    delta_r = np.zeros_like(theta)
    for i in range(len(m)):
        delta_r += _delta_unit_single_circular_multipole(theta, m[i], a_m[i], theta_m[i])
    return delta_r

def delta_circular_multipole(r0, theta, m, a_m, theta_m):
    # scaled with r0 only; needs to be scaled with 1/sqrt(q)
    delta_r = delta_unit_circular_multipole(theta, m, a_m, theta_m)
    return delta_r * r0

def circularize(r, theta, q):
    phi = theta2phi(theta, q)
    r_new = r / ellipticize_factor_simple(q,theta)
    return r_new, phi

def remove_multipole(r, phi, m, a_m, phi_m):
    m = np.asarray(m).reshape(-1)
    a_m = np.asarray(a_m).reshape(-1)
    phi_m = np.asarray(phi_m).reshape(-1)
    assert len(a_m) == len(m) == len(phi_m)
    factor = np.zeros_like(r)
    for i in range(len(m)):
        factor += _single_multipole_factor(r, phi, m[i], a_m[i], phi_m[i])
    r_new = r.copy() / (1 + factor)
    return r_new

def remove_circular_multipole_and_circularize(r, theta, q, m, a_m, theta_m):
    m = np.array([m]).reshape(-1)
    a_m = np.array([a_m]).reshape(-1)
    theta_m = np.array([theta_m]).reshape(-1)
    assert len(a_m) == len(m) == len(theta_m)
    # if m.size == 1:
    factor_ell = ellipticize_factor_simple(q,theta)
    factor_CM = 1/np.sqrt(q) * delta_unit_circular_multipole(theta, m, a_m, theta_m)
    r_new = r.copy() / (factor_ell + factor_CM)
    return r_new

def XY2RTHETA(X, Y):
    # X, Y: 1D or 2D Array
    R = np.sqrt(X * X + Y * Y)
    THETA = np.arctan2(Y, X)
    return R, THETA

def RTHETA2XY(R, THETA):
    # R, THETA: 1D or 2D array
    X = R * np.cos(THETA)
    Y = R * np.sin(THETA)
    return X, Y

def angle_like_r(r, include_end=True):
    assert len(r > 1)
    # makes the same dimention theta from r, 0~2 pi
    if include_end:
        theta = np.linspace(0, 2 * np.pi, num=len(r))
    else:
        theta = np.linspace(0, 2 * np.pi, num=len(r)+1)[:-1]
    return theta