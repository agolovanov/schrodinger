import numpy as np
import wavefunction


def test_norm():
    phases = np.linspace(0, 2*np.pi, 20)
    x = np.linspace(-100, 100, 10001)
    for phase in phases:
        psi = np.exp(1j * phase) * np.power(2 / np.pi, 0.25) * np.exp(- x ** 2)
        norm = wavefunction.norm(x, psi)
        np.testing.assert_almost_equal(norm, 1.0)

        psi *= 4.0
        norm = wavefunction.norm(x, psi)
        np.testing.assert_almost_equal(norm, 16.0)


def test_momentum_representation_planewave():
    x = np.linspace(-100, 100, 10001)
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


def test_coordinate_representation_planewave():
    p = np.linspace(-100, 100, 10001)
    l = (p[-1] - p[0] + p[1] - p[0])
    x0s = [10 * np.pi / l, 100 * np.pi / l, 1000 * np.pi / l]
    for x0 in x0s:
        psi_p = np.exp(-1.0j * p * x0) / np.sqrt(l)
        np.testing.assert_array_almost_equal(wavefunction.norm(p, psi_p), 1.0, decimal=4)

        x, psi = wavefunction.coordinate_representation(p, psi_p)
        np.testing.assert_equal(np.sort(x), x, err_msg="Array of x is not sorted")

        np.testing.assert_array_almost_equal(wavefunction.norm(x, psi), 1.0, decimal=4,
                                             err_msg="The norm is not conserved in coordinate representation")

        x_max = x[np.argmax(np.abs(psi))]
        np.testing.assert_allclose(x0, x_max, err_msg=f"Spectral maximum {x_max} is not equal to expected {x0}")


def __gauss(x, x0, sigma):
    return (2 / np.pi / sigma ** 2) ** 0.25 * np.exp(- (x - x0) ** 2 / sigma ** 2) + 0.0j


def test_momentum_representation_gauss():
    x = np.linspace(-100, 100, 10001)
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


def test_coordinate_representation_gauss():
    p = np.linspace(-100, 100, 10001)
    p0s = [0.0, -10.5, 5.0]
    sigmas = [1.0, 5.0, 0.1]
    for sigma in sigmas:
        for p0 in p0s:
            psi_p = __gauss(p, p0, 2 / sigma)
            np.testing.assert_array_almost_equal(wavefunction.norm(p, psi_p), 1.0)

            x, psi = wavefunction.coordinate_representation(p, psi_p)
            np.testing.assert_equal(np.sort(x), x, err_msg="Array of x is not sorted")

            np.testing.assert_array_almost_equal(wavefunction.norm(x, psi), 1.0,
                                                 err_msg="The norm is not conserved in coordinate representation")

            psi_expected = __gauss(x, 0.0, sigma) * np.exp(1.0j * p0 * x)
            np.testing.assert_allclose(psi, psi_expected, atol=0.02)


def test_two_representations_gauss():
    x = np.linspace(-100, 100, 10001)
    x0s = [0.0, -10.5, 5.0]
    sigmas = [1.0, 5.0, 0.1]
    for sigma in sigmas:
        for x0 in x0s:
            psi = __gauss(x, x0, sigma)
            x1, psi1 = wavefunction.coordinate_representation(*wavefunction.momentum_representation(x, psi))

            np.testing.assert_allclose(x, x1, err_msg=f"The array of x changed after two transforms for x0={x0}, "
                                                      f"sigma={sigma}")
            np.testing.assert_allclose(psi, psi1, err_msg=f"Wavefunction changed after two transforms for x0={x0}, "
                                                          f"sigma={sigma}", atol=1e-7)


def test_correlation():
    x = np.linspace(-100, 100, 10001)
    psi1 = __gauss(x, 0.0, 2.0)
    phase_mult = np.exp(0.25j * np.pi)
    psi2 = psi1 * phase_mult
    np.testing.assert_allclose(wavefunction.norm(x, psi1), wavefunction.correlation(x, psi1, psi1))
    np.testing.assert_allclose(wavefunction.norm(x, psi2), wavefunction.correlation(x, psi2, psi2))

    np.testing.assert_allclose(wavefunction.norm(x, psi1) * phase_mult, wavefunction.correlation(x, psi1, psi2))
    np.testing.assert_allclose(wavefunction.norm(x, psi1) * np.conj(phase_mult),
                               wavefunction.correlation(x, psi2, psi1))


def test_two_representations_correlation():
    x = np.linspace(-100, 100, 10001)
    x1 = 0.0
    x2 = 3.0
    sigma1 = 1.0
    sigma2 = 5.0
    psi1 = __gauss(x, x1, sigma1)
    psi2 = __gauss(x, x2, sigma2)
    p, psi1_p = wavefunction.momentum_representation(x, psi1)
    _, psi2_p = wavefunction.momentum_representation(x, psi2)

    corr1 = wavefunction.correlation(x, psi1, psi2)
    corr2 = wavefunction.correlation(p, psi1_p, psi2_p)
    np.testing.assert_almost_equal(corr1, corr2)
