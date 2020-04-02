import wigner
import wavefunction
import numpy as _np


def test_gaussian():
    x = _np.linspace(-10, 10, 201)

    ps = [0, -2, 1.5]
    x0s = [0, -1, 2.3]
    widths = [1, 2.5]

    for p in ps:
        for x0 in x0s:
            for sigma in widths:
                psi = _np.exp(- (x - x0) ** 2 / sigma ** 2) * _np.exp(1j * p * x)
                xx, pp, w = wigner.wigner_distribution(x, psi)
                x_new, psi_density = wigner.calculate_x_density(xx, pp, w)
                x_new = x_new[::2]
                psi_density = psi_density[::2]

                _np.testing.assert_almost_equal(x, x_new)
                _np.testing.assert_almost_equal(_np.abs(psi) ** 2, psi_density)

                p, psi_p = wavefunction.momentum_representation(x, psi)
                p_new, psi_density_p = wigner.calculate_p_density(xx, pp, w)
                _np.testing.assert_allclose(_np.abs(psi_p) ** 2, psi_density_p, atol=5e-2)