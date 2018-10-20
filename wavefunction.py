import numpy as _np


def _integrate(x, f):
    """
    Calculates an integral of `f` over `x`.
    If the dimension `N` of `f` is more than 1, than `x` should contain a tuple of `N` arrays of the same shape as `f`.
    :param x: array or tuple of arrays of coordinates.
    :param f: array of function values
    :return: the value of the integral
    """
    if isinstance(x, _np.ndarray):
        if x.shape != f.shape:
            raise ValueError(f"Shapes of x {x.shape} and integrand {f.shape} are different")
        return _np.trapz(f, x)
    else:
        dim = len(x)
        if dim != len(f.shape):
            raise ValueError(f"Dimensions of x {len(x)} and integrand {len(f.shape)} are different")
        tmp = f
        for i in range(dim):
            # constructing a slice for indexing, e.g. x[1][0, :, 0] should give a 1D array of the 2nd coordinate values.
            index = [0] * dim
            index[i] = slice(None)
            coord = x[i][tuple(index)]
            tmp = _np.trapz(tmp, coord, axis=0)
        return tmp


def norm(x, psi):
    """
    Calculates the norm of the wavefunction psi(x).
    Accepts an arbitrary dimension of the coordinate space.
    If the dimension `N` of `psi1` and `psi2` is more than 1, than `x` should contain a tuple of `N` arrays of the same
    shape as `psi1`.
    :param x: array or tuple of arrays of coordinates.
    The arrays are assumed to be constructed via `np.meshgrid` with `ij` indexing.
    :param psi: array of corresponding values of psi(x). 
    :return: the norm of the wavefunction
    """
    return _integrate(x, _np.abs(psi) ** 2)


def correlation(x, psi1, psi2):
    """
    Calculates a correlation between two functions defined as int(psi1* psi2 dx)
    Accepts an arbitrary dimension of the coordinate space.
    If the dimension `N` of `psi1` and `psi2` is more than 1, than `x` should contain a tuple of `N` arrays of the same
    shape as `psi1`.
    :param x: array or tuple of arrays of coordinates.
    The arrays are assumed to be constructed via `np.meshgrid` with `ij` indexing.
    :param psi1: array of corresponding values of psi1(x)
    :param psi2: same
    :return: the correlation between the two wavefunctions
    """
    if psi1.shape != psi2.shape:
        raise ValueError(f"Shapes of psi1 {psi1.shape} and psi2 {psi2.shape} are different")

    corr_arr = _np.conj(psi1) * psi2
    return _integrate(x, corr_arr)


def momentum_representation(x, psi):
    """
    Converts a wavefunction psi(x) to the momentum representation
    :param x:
    :param psi:
    :return: a tuple of (p, psi_p)
    """
    from scipy.fftpack import fft, fftfreq, fftshift, ifftshift
    if len(x) != len(psi):
        raise ValueError(f"Lengths of x ({len(x)}) and psi ({len(psi)}) are different")
    if len(x) % 2 == 0:
        raise ValueError(f"Incorrect length {len(x)}, only odd lengths are allowed")
    dx = x[1] - x[0]
    p = 2 * _np.pi * fftshift(fftfreq(len(psi))) / dx
    psi_p = fftshift(fft(ifftshift(psi))) * dx / _np.sqrt(2 * _np.pi)
    return p, psi_p


def coordinate_representation(p, psi_p):
    """
    Converts a wavefunction psi_p(p) to the coordinate representation
    :param p:
    :param psi_p:
    :return: a tuple of (x, psi).
    """
    from scipy.fftpack import ifft, fftfreq, fftshift, ifftshift
    if len(p) != len(psi_p):
        raise ValueError(f"Lengths of p ({len(p)}) and psi_p ({len(psi_p)}) are different")
    if len(p) % 2 == 0:
        raise ValueError(f"Incorrect length {len(p)}, only odd lengths are allowed")
    dp = p[1] - p[0]
    x = 2 * _np.pi * fftshift(fftfreq(len(psi_p))) / dp
    psi = fftshift(ifft(ifftshift(psi_p))) * len(p) * dp / _np.sqrt(2 * _np.pi)
    return x, psi