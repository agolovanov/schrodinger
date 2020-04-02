import numpy as _np

def wigner_distribution(x, psi):
    """
    Calculates the Wigner distribution of the wavefunction psi(x)

    :param x: array of coordinates
    :param psi: array of wavefunction values
    :return: x, p, w - three 2d arrays with x, p and W (Wigner fanction) values.
    """
    from scipy.fft import fft, fftshift, fftfreq

    length = len(x)
    dx = x[1] - x[0]
    x_new = _np.linspace(_np.min(x), _np.min(x) + (length - 0.5) * dx, 2 * length)

    psi_new = _np.zeros(x_new.shape, dtype=_np.complex)
    psi_new[::2] = psi
    psi_new[1::2] = 0.5 * (psi + _np.roll(psi, -1))

    p_new = 2 * _np.pi * fftshift(fftfreq(2 * length, d=dx))

    wigner_core = _np.zeros((2 * length, 2 * length), dtype=_np.complex)
    for i in range(2 * length):
        wigner_core[:, i] = _np.roll(_np.conj(psi_new), i) * _np.roll(psi_new, -i)

    wigner = _np.real(fftshift(fft(wigner_core, axis=1), axes=1) * dx) / 2 / _np.pi

    xx, pp = _np.meshgrid(x_new, p_new)

    return xx, pp, wigner.T


def calculate_x_density(x, p, w):
    """
    Calculates the density |psi(x)|^2 in the coordinate space from the Wigner distribution w
    :param x:
    :param p:
    :param w:
    :return: x (1d array), |psi(x)|^2
    """
    return x[0], _np.sum(w, axis=0) * __calculate_dp(p)


def calculate_p_density(x, p, w):
    """
    Calculates the density |psi(p)|^2 in the momentum space from the Wigner distribution w
    :param x:
    :param p:
    :param w:
    :return: p (1d array), |psi(p)|^2
    """
    integral = _np.sum(w, axis=1) * __calculate_dx(x)
    integral = 0.5 * (integral + _np.roll(integral, -1))
    return p[1::2, 0], integral[1::2]


def __calculate_dx(x):
    return x[0, 1] - x[0, 0]


def __calculate_dp(p):
    return p[1, 0] - p[0, 0]