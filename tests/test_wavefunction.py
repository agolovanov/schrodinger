import numpy as np
import wavefunction


def test_norm():
    phases = np.linspace(0, 2*np.pi, 20)
    x = np.linspace(-100, 100, 10000)
    for phase in phases:
        psi = np.exp(1j * phase) * np.power(2 / np.pi, 0.25) * np.exp(- x ** 2)
        norm = wavefunction.norm(x, psi)
        np.testing.assert_almost_equal(norm, 1.0)

        psi *= 4.0
        norm = wavefunction.norm(x, psi)
        np.testing.assert_almost_equal(norm, 16.0)