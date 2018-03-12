import potential
import wavefunction
import solver
import numpy as np


def test_coordinate():
    s = solver.EulerSolver(1, 0.5, 0.1)
    np.testing.assert_array_almost_equal(s.x, np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))

    s = solver.EulerSolver(2, 0.6, 0.1)
    np.testing.assert_array_almost_equal(s.x, np.linspace(-2.4, 2.4, 9))


def test_euler_delta():
    depths = [0.2, 1.0, 5.0]

    for d in depths:
        v = potential.DeltaPotential(d)

        e0 = v.get_eigenenergy()
        tmax = - 2 * np.pi / e0
        dt = tmax / 500000

        s = solver.EulerSolver(20 / d, 0.1 / d, dt, v)
        psi0 = v.get_eigenfunction()
        psi0_value = psi0(s.x)
        np.testing.assert_almost_equal(wavefunction.norm(s.x, psi0_value), 1, decimal=2)

        psi = s.execute(tmax, psi0=psi0)
        np.testing.assert_almost_equal(wavefunction.norm(s.x, psi), wavefunction.norm(s.x, psi0_value), decimal=3,
                                       err_msg=f"Norm not conserved for depth {d}")

        np.testing.assert_allclose(np.abs(psi), np.abs(psi0_value), atol=0.06)


def test_cn_delta():
    depths = [0.2, 1.0, 5.0]

    for d in depths:
        v = potential.DeltaPotential(d)

        e0 = v.get_eigenenergy()
        tmax = - 2 * np.pi / e0
        dt = tmax / 1000

        s = solver.CrankNicolsonSolver(20 / d, 0.1 / d, dt, v)
        psi0 = v.get_eigenfunction()
        psi0_value = psi0(s.x)
        np.testing.assert_almost_equal(wavefunction.norm(s.x, psi0_value), 1, decimal=2)

        times = [tmax, tmax / 2, tmax / 4]
        psi_expected = [psi0_value, -psi0_value, psi0_value * np.exp(0.5j * np.pi)]

        for t, psi1 in zip(times, psi_expected):
            psi = s.execute(t, psi0=psi0)
            np.testing.assert_almost_equal(wavefunction.norm(s.x, psi), wavefunction.norm(s.x, psi0_value), decimal=7,
                                           err_msg=f"Norm not conserved for depth {d}")

            np.testing.assert_allclose(np.abs(psi), np.abs(psi0_value), atol=0.02)

            np.testing.assert_allclose(psi, psi1, atol=0.05)