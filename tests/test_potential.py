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
        assert(v.get_delta_depth() == d)

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


def test_quadratic_orthogonality():
    frequencies = [0.1, 1.0, 7.5]
    x = np.linspace(-50, 50, 40000)
    levels = range(10)

    for f in frequencies:
        v = potential.QuadraticPotential(f)
        for l1 in levels:
            for l2 in levels[l1+1:]:
                psi1 = v.get_eigenfunction(l1)(x)
                psi2 = v.get_eigenfunction(l2)(x)
                np.testing.assert_almost_equal(wavefunction.correlation(x, psi1, psi2), 0.0,
                                               err_msg=f'Functions for levels {l1} and {l2} are not orthogonal '
                                                       f'for frequency {f}')


def test_uniform_field():
    amps = [1.0, -2.0]
    x = np.linspace(-10, 10, 1000)
    for amp in amps:
        v = potential.UniformField(amp)
        with pytest.raises(ValueError):
            v.get_eigenfunction()
        with pytest.raises(ValueError):
            v.get_eigenenergy()
        np.testing.assert_allclose(-amp * x, v.get_potential()(x))

        assert(v.get_delta_depth() == 0.0)

        v = potential.UniformField(amp, potential=potential.QuadraticPotential(1.0))
        value1 = - amp * x + 0.5 * x ** 2
        value2 = v.get_potential()(x)
        np.testing.assert_allclose(value1, value2)

        v = potential.UniformField(amp, potential=potential.DeltaPotential(1.0))
        np.testing.assert_allclose(-amp * x, v.get_potential()(x))
        assert(v.get_delta_depth() == 1.0)