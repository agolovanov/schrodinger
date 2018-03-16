import potential
import wavefunction
import eigen
import numpy as np


def test_eigen_quadratic():
    freqs = [0.5, 1.0, 5.0]
    x = np.linspace(-200, 200, 10000)
    levels = range(10)

    for f in freqs:
        v = potential.QuadraticPotential(f)

        es, psis = eigen.calculate_eigenstates(x, v.get_potential(), 10, 0.0)
        for i, psi in enumerate(psis):
            np.testing.assert_almost_equal(wavefunction.norm(x, psi), 1.0, err_msg=f'Eigenfunction {i} norm is not 1')

        e0s = np.array([v.get_eigenenergy(l) for l in levels])
        psi0s = [v.get_eigenfunction(l)(x) for l in levels]

        np.testing.assert_allclose(e0s, es, rtol=0.02, err_msg=f"Energy spectra are different for frequency {f}")

        assert len(psis) == len(psi0s), f"Incorrect number of eigenfunctions {len(psis)}, expected {len(psi0s)}"

        for i, (psi0, psi) in enumerate(zip(psi0s, psis)):
            corr = np.abs(wavefunction.correlation(x, psi0, psi))
            np.testing.assert_almost_equal(corr, 1.0, decimal=3, err_msg=f'Function {i} is incorrect')


def test_eigen_square():
    widths = [4.0, 6.0]
    depths = [2., 5.]
    x = np.linspace(-100, 100, 10000)

    for a, V0 in zip(widths, depths):
        v = potential.SquarePotential(V0, a)

        levels = v.get_number_of_levels()
        es, psis = eigen.calculate_eigenstates(x, v.get_potential(), levels, -V0)
        for i, psi in enumerate(psis):
            np.testing.assert_almost_equal(wavefunction.norm(x, psi), 1.0, err_msg=f'Eigenfunction {i} norm is not 1')

        e0s = np.array([v.get_eigenenergy(l) for l in range(levels)])
        psi0s = [v.get_eigenfunction(l)(x) for l in range(levels)]

        np.testing.assert_allclose(e0s, es, atol=0.02, rtol=0.05,
                                   err_msg=f"Energy spectra are different for a={a}, V0={V0}")

        assert len(psis) == len(psi0s), f"Incorrect number of eigenfunctions {len(psis)}, expected {len(psi0s)}"

        for i, (psi0, psi) in enumerate(zip(psi0s, psis)):
            corr = np.abs(wavefunction.correlation(x, psi0, psi))
            np.testing.assert_almost_equal(corr, 1.0, decimal=3, err_msg=f'Function {i} is incorrect')