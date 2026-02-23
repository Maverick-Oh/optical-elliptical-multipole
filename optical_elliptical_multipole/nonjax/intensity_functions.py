
"""
Intensity functions (circular profiles).

Notes
-----
These functions are kept NumPy-only (no Numba) because they are typically
lightweight wrappers and are convenient to JIT later with JAX. Numerical
heavy-lifting happens in tools/profiles layers.
"""
import numpy as np

def exp_intensity(R, R0=1.0):
    """
    Simple exponential intensity profile: I(R) = exp(-R/R0) for testing purposes.

    Parameters
    ----------
    R : array_like
        Radius (>= 0).
    R0 : float
        Scale radius R0 (> 0).

    Returns
    -------
    I : ndarray
        Intensity values with the same shape as R.
    """
    R = np.asarray(R, dtype=float)
    return np.exp(-R/R0)

def _b(n):
    """
    Approximation to the Sersic b_n parameter.
    b(n) ~ 1.999n - 0.327  (valid for common n).
    """
    n = float(n)
    if not (n > 0):
        raise ValueError("Sersic index n must be positive.")
    b = 1.999 * n - 0.327
    # Guard against pathological inputs
    return max(b, 1e-5)

def _R_stable(R, R_min=0.0001):
    """
    For numerical stability, limit minimum R value as R_min.
    """
    return np.maximum(R_min, R)

def sersic(R, amplitude=1.0, R_sersic=1.0, n_sersic=4.0):
    """
    Sersic surface brightness profile.

    I(R) = A * exp{ -b_n * [ (R/R_s)^(1/n) - 1 ] }

    Parameters
    ----------
    R : array_like
        Radius (>= 0).
    amplitude : float
        Central normalization A.
    R_sersic : float
        Scale radius R_s (> 0).
    n_sersic : float
        Sersic index n (> 0).

    Returns
    -------
    I : ndarray
        Intensity values with the same shape as R.
    """
    R = _R_stable(np.asarray(R, dtype=float))
    if not (R_sersic > 0):
        raise ValueError("R_sersic must be > 0.")
    if not (n_sersic > 0):
        raise ValueError("n_sersic must be > 0.")
    bn = _b(n_sersic)
    # Use exp(log) form for numeric stability at extreme ranges
    # with np.errstate(divide='ignore', invalid='ignore'):
    x = R / R_sersic
    # x**(1/n) as exp((1/n)*log(x)), guard x<=0 separately
    logx = np.where(x > 0, np.log(x), -np.inf)
    pow_term = np.exp((1.0 / n_sersic) * logx)
    out = amplitude * np.exp(-bn * (pow_term - 1.0))
    # For R=0 -> x=0 -> logx=-inf -> pow_term=0 -> exp(bn) finite
    out = np.where(x >= 0, out, 0.0)
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print('debug of intensity_functions.py')
    range_arcsec = 2
    ss_factor = 4 # supersample factor
    r = 0.03*np.arange(int(range_arcsec/0.03))
    r_ss = 0.03 / ss_factor * np.arange(int(range_arcsec/0.03*ss_factor))
    #
    def mean_of_every_n(arr, n):
        # Ensure the input is a numpy array
        arr = np.array(arr)
        
        # Check if the array length is a multiple of n
        if len(arr) % n != 0:
            # Handle cases where the array length is not evenly divisible by n
            print(f"Warning: Array length ({len(arr)}) is not a multiple of {n}. Truncating array.")
        arr = arr[:len(arr) - (len(arr) % n)]

        # Reshape the 1D array into a 2D array with n columns
        # The -1 lets numpy automatically calculate the number of rows
        reshaped_arr = arr.reshape(-1, n)
        
        # Calculate the mean across the second axis (axis=1, which corresponds to the columns)
        # This computes the average of each row (each group of n elements)
        means = np.mean(reshaped_arr, axis=1)
        
        return means
    #
    y = sersic(r, amplitude=1.0, R_sersic=0.5, n_sersic=4.0)
    y_ss = sersic(r_ss, amplitude=1.0, R_sersic=0.5, n_sersic=4.0)
    y_ss_mean = mean_of_every_n(y_ss, ss_factor)
    y_lg = sersic(r, amplitude=1.0, R_sersic=2.0, n_sersic=4.0)
    #
    plt.plot(r, y, 'k-', label='original')
    plt.plot(r, y_ss_mean, 'r-', label='supersampled')
    plt.plot(r, y_lg, 'g-', label='large R_sersic')
    plt.legend()
    plt.show()
    print('debug')