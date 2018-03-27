import potential
import wavefunction
import numpy as np
import pytest
import random


def test_delta_potential():
    x = np.linspace(-50, 50, 40000)
    depths = np.linspace(0.1, 10, 10)
    for d in depths:
        v = potential.DeltaPotential1D(d)
        assert(v.get_delta_depth() == d)

        assert(v.get_number_of_levels() == 1)

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
        v = potential.QuadraticPotential1D(f)
        assert(v.get_frequency() == f)

        with pytest.raises(Exception):
            v.get_number_of_levels()

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
        v = potential.QuadraticPotential1D(f)
        for l1 in levels:
            for l2 in levels[l1+1:]:
                psi1 = v.get_eigenfunction(l1)(x)
                psi2 = v.get_eigenfunction(l2)(x)
                np.testing.assert_almost_equal(wavefunction.correlation(x, psi1, psi2), 0.0,
                                               err_msg=f'Functions for levels {l1} and {l2} are not orthogonal '
                                                       f'for frequency {f}')


def test_uniform_field():
    amps = [1.0, -2.0, lambda t: 0.5 * t]
    t = 3.0
    x = np.linspace(-10, 10, 1000)

    for amp in amps:
        v = potential.UniformField1D(amp)

        assert(v.get_number_of_levels() == 0)

        with pytest.raises(ValueError):
            v.get_eigenfunction()
        with pytest.raises(ValueError):
            v.get_eigenenergy()
        if callable(amp):
            np.testing.assert_allclose(-amp(t) * x, v.get_potential()(t, x))
        else:
            np.testing.assert_allclose(-amp * x, v.get_potential()(x))

        assert(v.get_delta_depth() == 0.0)

        v = potential.UniformField1D(amp, potential=potential.QuadraticPotential1D(1.0))
        if callable(amp):
            value1 = - amp(t) * x + 0.5 * x ** 2
            value2 = v.get_potential()(t, x)
        else:
            value1 = - amp * x + 0.5 * x ** 2
            value2 = v.get_potential()(x)
        np.testing.assert_allclose(value1, value2)

        v = potential.UniformField1D(amp, potential=potential.DeltaPotential1D(1.0))
        if callable(amp):
            np.testing.assert_allclose(-amp(t) * x, v.get_potential()(t, x))
        else:
            np.testing.assert_allclose(-amp * x, v.get_potential()(x))
        assert(v.get_delta_depth() == 1.0)


def test_square_potential():
    widths = [1.0, 0.5, 2.0]
    depths = [1.0, 5.0, 10.0]
    x = np.linspace(-150, 150, 3000)

    expected_levels = {
        (1.0, 1.0): 1,
        (0.5, 1.0): 1,
        (2.0, 1.0): 2,
        (1.0, 5.0): 3,
        (0.5, 5.0): 2,
        (2.0, 5.0): 5,
        (1.0, 10.0): 3,
        (0.5, 10.0): 2,
        (2.0, 10.0): 6
    }

    from itertools import product
    for V0, a in product(depths, widths):
        v = potential.SquarePotential1D(V0, a)

        assert v.get_depth() == V0, f"Depth {v.get_depth()} is not {V0}"
        assert v.get_width() == a, f"Width {v.get_width()} is not {a}"
        assert v.get_delta_depth() == 0.0, f"Delta depth {v.get_delta_depth()} is non-zero"

        max_levels = v.get_number_of_levels()
        assert max_levels == expected_levels[a, V0], f"Max levels {max_levels}, expected {expected_levels[a, V0]}"
        with pytest.raises(ValueError):
            v.get_eigenenergy(max_levels)
        with pytest.raises(ValueError):
            v.get_eigenfunction(max_levels)

        assert v.get_potential()(0.0) == -V0
        assert v.get_potential()(2 * a) == 0.0
        assert v.get_potential()(-2 * a) == 0.0

        energies = np.array([v.get_eigenenergy(i) for i in range(max_levels)])
        np.testing.assert_equal(energies, np.sort(energies), err_msg=f"Energies aren't sorted for V0={V0}, a={a}")
        assert np.all(energies < 0.0), f"Positive energies for V0={V0}, a={a}"
        assert np.all(energies > -V0), f"Too low energies for V0={V0}, a={a}"

        for i in range(max_levels):
            psi = v.get_eigenfunction(i)(x)
            np.testing.assert_almost_equal(wavefunction.norm(x, psi), 1.0, decimal=4,
                                           err_msg=f"Eigenfunction {i} norm is incorrect for V0={V0}, a={a}")


def test_square_potential_orthogonality():
    from itertools import combinations
    widths = [1.0, 2.0]
    depths = [10.0, 5.0]
    x = np.linspace(-15, 15, 3000)

    for V0, a in zip(depths, widths):
        v = potential.SquarePotential1D(V0, a)
        assert v.get_depth() == V0, f"Depth {v.get_depth()} is not {V0}"
        assert v.get_width() == a, f"Width {v.get_width()} is not {a}"

        psis = [v.get_eigenfunction(n)(x) for n in range(v.get_number_of_levels())]
        for psi1, psi2 in combinations(psis, 2):
            np.testing.assert_almost_equal(wavefunction.correlation(x, psi1, psi2), 0.0,
                                           err_msg="Non-orthogonal eigenfunctions for V0={V0}, a={a}")


def test_coulomb_potential():
    x = np.linspace(-30, 30, 201)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    r = (xx, yy, zz)
    levels = [
        (1, 0, 0),
        (2, 0, 0),
        (2, 1, 1),
        (2, 1, -1),
        (3, 2, -1)
    ]
    v = potential.CoulombPotential()
    psis = []

    for l in levels:
        e = v.get_eigenenergy(*l)
        np.testing.assert_allclose(e, - 0.5 / l[0] ** 2)

        psi = v.get_eigenfunction(*l)(*r)
        psis.append(psi)

        np.testing.assert_array_almost_equal(wavefunction.norm(r, psi), 1.0, decimal=3,
                                             err_msg=f"Wavefunction norm for level {l} is not unity")

    from itertools import combinations
    for psi1, psi2 in combinations(psis, 2):
        np.testing.assert_allclose(wavefunction.correlation(r, psi1, psi2), 0.0, atol=0.001,
                                   err_msg=f"Non-orthogonal wavefunctions")