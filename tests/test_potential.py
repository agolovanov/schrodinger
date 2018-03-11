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