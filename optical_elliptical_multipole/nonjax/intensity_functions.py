import numpy as np

def exp_intensity(R):
    # for developing code and debugging purpose
    return np.exp(-R)

def _b(n):
    b = 1.999 * n - 0.327
    b = np.maximum(
            b, 0.00001
        )
    return b

def sersic(R, amplitude=None, R_sersic=None, n_sersic=None):
    assert R_sersic is not None
    assert n_sersic is not None
    assert amplitude is not None

    return amplitude * np.exp(-_b(n_sersic) * ((R/R_sersic)**(1./n_sersic)-1.) )
