import numpy as _np
import numba as _numba


class Solver():
    psi = None
    x = None
    dx = None
    n_points = None
    potential = None

    def __init__(self, x_max, dx, potential=None):
        """
        Creates a solver for the interval [-x_max, x_max] with the spatial step `dx`.
        :param x_max:
        :param dx:
        :param potential:
        """
        self.dx = dx
        self.n_points = int((2 * x_max) / dx) + 1
        if self.n_points % 2 == 0:
            self.n_points += 1
        self.x = _np.linspace(- (self.n_points // 2) * dx, (self.n_points // 2) * dx, self.n_points)
        if potential is not None:
            self.potential = potential
        else:
            self.potential = _numba.vectorize(nopython=True)(lambda t, x: 0.0)
        self.psi = _np.zeros(self.x.shape, dtype=_np.complex)

    def set_psi(self, psi):
        """
        Sets the psi-function. If `psi` is a tuple of functions, they define the real and imaginary parts, respectively.
        If `psi` is a single function, than imaginary part is assumed to be zero.
        :param psi:
        """
        if isinstance(psi, tuple):
            if len(psi) != 2:
                raise ValueError(f"Tuple of length 2 expected, got {len(psi0)}")
            self.psi = psi[0](self.x) + 1j * psi[1](self.x)
        else:
            self.psi = psi(self.x) + 1j * 0.0

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
        if output_dt < dt:
            raise ValueError(f"output_dt = {output_dt} is smaller than dt = {dt}")
        iterations = int(t_max / dt)
        output_i = 0
        ts = []
        psis = []
        for i in range(iterations):
            if (output_dt is not None) and (i * dt >= output_dt * output_i):
                ts.append(i * dt)
                psis.append(self.psi.copy())
                output_i += 1
            self.iterate(dt, i * dt)
        return (_np.array(ts), psis)


class EulerSolver(Solver):
    delta_depth = 0.0
    dpsi = None

    def __init__(self, x_max, dx, potential=None, delta_depth=0.0):
        """
        Eulerian solver with delta potential of depth `delta_depth`.
        :param x_max:
        :param dx:
        :param potential:
        :param delta_depth:
        """
        Solver.__init__(self, x_max, dx, potential)
        self.delta_depth = delta_depth
        self.dpsi = _np.zeros(self.psi.shape, dtype=_np.complex)

    @staticmethod
    @_numba.jit(nopython=True)
    def __iterate(psi, dpsi, dt, dx, n_points, potential, delta_depth):
        dpsi[0] = 0.5 * 1j * (psi[1] + psi[-1] - 2 * psi[0]) / dx ** 2
        dpsi[-1] = 0.5 * 1j * (psi[0] + psi[-2] - 2 * psi[-1]) / dx ** 2
        center = n_points // 2
        for i in range(1, n_points - 1):
            dpsi[i] = 0.5 * 1j * (psi[i + 1] + psi[i - 1] - 2 * psi[i]) / dx ** 2 - 1j * potential[i]
        for i in range(n_points):
            psi[i] += dpsi[i] * dt
        if delta_depth != 0.0:
            psi[center] = (psi[center - 1] + psi[center + 1]) / (2 - 2 * delta_depth * dx)

    def iterate(self, dt, t):
        EulerSolver.__iterate(self.psi, self.dpsi, dt, self.dx, self.n_points, self.potential(t, self.x),
                              self.delta_depth)