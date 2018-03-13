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


def test_momentum_representation_planewave():
    x = np.linspace(-100, 100, 10000)
    l = (x[-1] - x[0] + x[1] - x[0])
    p0s = [10 * np.pi / l, 100 * np.pi / l, 1000 * np.pi / l]
    for p0 in p0s:
        psi = np.exp(1.0j * p0 * x) / np.sqrt(l)
        np.testing.assert_array_almost_equal(wavefunction.norm(x, psi), 1.0, decimal=4)

        p, psi_p = wavefunction.momentum_representation(x, psi)
        np.testing.assert_equal(np.sort(p), p, err_msg="Array of p is not sorted")

        np.testing.assert_array_almost_equal(wavefunction.norm(p, psi_p), 1.0, decimal=4,
                                             err_msg="The norm is not conserved in momentum representation")

        p_max = p[np.argmax(np.abs(psi_p))]
        np.testing.assert_allclose(p0, p_max, err_msg=f"Spectral maximum {p_max} is not equal to expected {p0}")


def __gauss(x, x0, sigma):
    return (2 / np.pi / sigma ** 2) ** 0.25 * np.exp(- (x - x0) ** 2 / sigma ** 2) + 0.0j


def test_momentum_representation_gauss():
    x = np.linspace(-100, 100, 10000)
    x0s = [0.0, -10.5, 5.0]
    sigmas = [1.0, 5.0, 0.1]
    for sigma in sigmas:
        for x0 in x0s:
            psi = __gauss(x, x0, sigma)
            np.testing.assert_array_almost_equal(wavefunction.norm(x, psi), 1.0)

            p, psi_p = wavefunction.momentum_representation(x, psi)
            np.testing.assert_equal(np.sort(p), p, err_msg="Array of p is not sorted")

            np.testing.assert_array_almost_equal(wavefunction.norm(p, psi_p), 1.0,
                                                 err_msg="The norm is not conserved in momentum representation")

            psi_p_expected = __gauss(p, 0.0, 2 / sigma) * np.exp(-1.0j * p * x0)
            np.testing.assert_allclose(psi_p, psi_p_expected, atol=0.02)