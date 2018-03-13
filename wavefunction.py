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


def momentum_representation(x, psi):
    """
    Converts wavefunction psi(x) to the momentum representation
    :param x:
    :param psi:
    :return: a tuple of (p, psi_p)
    """
    from scipy.fftpack import fft, fftfreq, fftshift, ifftshift
    if len(x) != len(psi):
        raise ValueError(f"Lengths of x ({len(x)}) and psi ({len(psi)}) are different")
    dx = x[1] - x[0]
    p = 2 * _np.pi * fftshift(fftfreq(len(psi))) / dx
    psi_p = fftshift(fft(ifftshift(psi))) * dx / _np.sqrt(2 * _np.pi)
    return p, psi_p
