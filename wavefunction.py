import numpy as _np


def norm(x, psi):
    """
    Calculates the norm of the wavefunction psi(x).
    :param x: array of x coordinates.
    :param psi: array of corresponding values of psi(x). 
    :return: the norm of the wavefunction
    """
    psi_sqr = _np.abs(psi) ** 2
    dx = x[1:] - x[:-1]
    return _np.sum(0.5 * (psi_sqr[1:] + psi_sqr[:-1]) * dx)