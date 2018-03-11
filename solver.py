import numpy as _np
import numba as _numba
import potential as _potential


class Solver():
    _psi = None
    x = None
    dx = None
    n_points = None
    potential = None
    stationary = False

    def __init__(self, x_max, dx, potential=None, stationary=False):
        """
        Creates a solver for the interval [-x_max, x_max] with the spatial step `dx`.
        :param x_max:
        :param dx:
        :param potential:
        :param stationary: if True, the potential is considered constant in time
        """
        self.dx = dx
        n1 = int(x_max / dx)
        if n1 * dx < x_max - 0.01 * dx:
            n1 += 1

        self.n_points = 2 * n1 + 1
        self.x = _np.linspace(- n1 * dx, n1 * dx, self.n_points)
        self.stationary = stationary
        if potential is None:
            self.potential = _numba.vectorize(nopython=True)(lambda x: 0.0j)
            self.stationary = True
        elif isinstance(potential, _potential.Potential):
            self.potential = potential.get_potential()
            self.stationary = True
            if isinstance(potential, _potential.DeltaPotential):
                self.delta_depth = potential.get_depth()
        else:
            self.potential = potential
        self._psi = _np.zeros(self.x.shape, dtype=_np.complex)

    def set_psi(self, psi):
        """
        Sets the psi-function. If `psi` is a tuple of functions, they define the real and imaginary parts, respectively.
        If `psi` is a single function, than imaginary part is assumed to be zero.
        :param psi:
        """
        if isinstance(psi, tuple):
            if len(psi) != 2:
                raise ValueError(f"Tuple of length 2 expected, got {len(psi0)}")
            self._psi = psi[0](self.x) + 1j * psi[1](self.x)
        else:
            self._psi = psi(self.x) + 1j * 0.0

    def iterate(self, dt, t):
        """
        Iterates the system over a single time step `dt` from the starting time `t`.
        :param dt:
        :param t:
        :return:
        """
        raise NotImplementedError()

    def execute(self, t_max, dt, output_dt=None, psi0=None):
        """
        Calculates the system up to `t_max` with timestep `dt` for the initial function `psi0`.
        Returns a list of times and solutions with the interval `output_dt`
        :param t_max:
        :param dt:
        :param output_dt:
        :param psi0: if None, current psi will be used.
        :return: A tuple of times and psi-functions.
        """
        if psi0 is not None:
            self.set_psi(psi0)
        if output_dt is not None and output_dt < dt:
            raise ValueError(f"output_dt = {output_dt} is smaller than dt = {dt}")
        iterations = int(t_max / dt)
        output_i = 0
        ts = []
        psis = []
        for i in range(iterations):
            if (output_dt is not None) and (i * dt >= output_dt * output_i):
                ts.append(i * dt)
                psis.append(self._psi.copy())
                output_i += 1
            self.iterate(dt, i * dt)
        return (_np.array(ts), psis) if output_dt is not None else self._psi.copy()


class EulerSolver(Solver):
    delta_depth = 0.0
    __dpsi = None
    __potential_x = None

    def __init__(self, x_max, dx, potential=None, stationary=False):
        Solver.__init__(self, x_max, dx, potential, stationary)
        if self.stationary:
            self.__potential_x = self.potential(self.x)
        self.__dpsi = _np.zeros(self.x.shape, dtype=_np.complex)

    @staticmethod
    @_numba.jit(nopython=True)
    def __iterate(psi, dpsi, dt, dx, n_points, potential, delta_depth):
        dpsi[0] = 0.5 * 1j * (psi[1] + psi[-1] - 2 * psi[0]) / dx ** 2
        dpsi[-1] = 0.5 * 1j * (psi[0] + psi[-2] - 2 * psi[-1]) / dx ** 2
        center = n_points // 2
        for i in range(1, n_points - 1):
            dpsi[i] = 0.5 * 1j * (psi[i + 1] + psi[i - 1] - 2 * psi[i]) / dx ** 2 - 1j * potential[i] * psi[i]
        for i in range(n_points):
            psi[i] += dpsi[i] * dt
        if delta_depth != 0.0:
            psi[center] = (psi[center - 1] + psi[center + 1]) / (2 - 2 * delta_depth * dx)

    def iterate(self, dt, t):
        potential = self.__potential_x if self.stationary else self.potential(t, self.x)
        EulerSolver.__iterate(self._psi, self.__dpsi, dt, self.dx, self.n_points, potential, self.delta_depth)