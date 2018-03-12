import potential
import wavefunction
import numpy as np
import pytest
import random


def test_delta_potential():
    x = np.linspace(-50, 50, 40000)
    depths = np.linspace(0.1, 10, 10)
    for d in depths:
        v = potential.DeltaPotential(d)
        assert(v.get_depth() == d)

        with pytest.raises(ValueError):
            v.get_eigenenergy(random.randint(1, 100))

        with pytest.raises(ValueError):
            v.get_eigenfunction(random.randint(1, 100))

        psi = v.get_eigenfunction()(x)
        np.testing.assert_almost_equal(wavefunction.norm(x, psi), 1.0, decimal=4,
                                       err_msg=f"Norm is not 1 for depth {d}")


def test_quadratic_potential():
    frequencies = [0.1, 1.0, 7.5]
    x =  np.linspace(-50, 50, 40000)
    levels = range(20)

    for f in frequencies:
        v = potential.QuadraticPotential(f)
        assert(v.get_frequency() == f)

        for l in levels:
            e = v.get_eigenenergy(l)
            np.testing.assert_almost_equal(e, (2 * l + 1) * f * 0.5)

            psi = v.get_eigenfunction(l)
            psi_value = psi(x)

            np.testing.assert_almost_equal(wavefunction.norm(x, psi_value), 1.0,
                                           err_msg=f'Norm is not 1 for frequency {f}, level {l}')

        psi0_value = v.get_eigenfunction()(x)
        psi0_expected = (f / np.pi) ** 0.25 * np.exp(- 0.5 * f * x ** 2)

        np.testing.assert_allclose(psi0_value, psi0_expected)