import numpy as _np
import numba as _numba
import potential as _potential
import scipy.sparse as _sparse
from scipy.sparse.linalg import spsolve as _spsolve


class Solver():
    _psi = None
    x = None
    dx = None
    dt = None
    n_points = None
    potential = None
    stationary = False
    delta_depth = 0.0

    def __init__(self, x_max, dx, dt, potential=None, stationary=False):
        """
        Creates a solver for the interval [-x_max, x_max] with a spatial step `dx` and a time step `dt`.
        :param x_max:
        :param dx:
        :param dt:
        :param potential:
        :param stationary: if True, the potential is considered constant in time
        """
        self.dx = dx
        self.dt = dt
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
            self.stationary = potential.is_stationary()
            self.delta_depth = potential.get_delta_depth()
        else:
            self.potential = potential
        self._psi = _np.zeros(self.x.shape, dtype=_np.complex)

    def set_psi(self, psi):
        """
        Sets the psi-function.
        :param psi: an array, a function, or a tuple of functions.
        If `psi` is a tuple of functions, they define the real and imaginary parts, respectively.
        """
        if isinstance(psi, _np.ndarray):
            if psi.shape != self.x.shape:
                raise ValueError(f"Shape of array of psi {psi.shape} is not the same as shape of x {x.shape}")
            self._psi = psi
        elif isinstance(psi, tuple):
            if len(psi) != 2:
                raise ValueError(f"Tuple of length 2 expected, got {len(psi0)}")
            self._psi = psi[0](self.x) + 1j * psi[1](self.x)
        else:
            self._psi = psi(self.x) + 1j * 0.0

    def iterate(self, t):
        """
        Iterates the system over a single time step from the starting time `t`.
        :param t:
        :return:
        """
        raise NotImplementedError()

    def execute(self, t_max, output_dt=None, psi0=None):
        """
        Calculates the system up to `t_max` for the initial function `psi0`.
        Returns a list of times and solutions with the interval `output_dt`
        :param t_max:
        :param output_dt:
        :param psi0: if None, current psi will be used.
        :return: A tuple of times and psi-functions.
        """
        dt = self.dt
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
            self.iterate(i * dt)
        return (_np.array(ts), psis) if output_dt is not None else self._psi.copy()


class EulerSolver(Solver):
    __dpsi = None
    __potential_x = None

    def __init__(self, x_max, dx, dt, potential=None, stationary=False):
        Solver.__init__(self, x_max, dx, dt, potential, stationary)
        if self.stationary:
            self.__potential_x = self.potential(self.x)
            self.__potential_x[self.n_points // 2] -= self.delta_depth / self.dx
        self.__dpsi = _np.zeros(self.x.shape, dtype=_np.complex)

    @staticmethod
    @_numba.jit(nopython=True)
    def __iterate(psi, dpsi, dt, dx, n_points, potential, delta_depth):
        dpsi[0] = 0.5 * 1j * (psi[1] + psi[-1] - 2 * psi[0]) / dx ** 2
        dpsi[-1] = 0.5 * 1j * (psi[0] + psi[-2] - 2 * psi[-1]) / dx ** 2
        for i in range(1, n_points - 1):
            dpsi[i] = 0.5 * 1j * (psi[i + 1] + psi[i - 1] - 2 * psi[i]) / dx ** 2 - 1j * potential[i] * psi[i]
        for i in range(n_points):
            psi[i] += dpsi[i] * dt

    def iterate(self, t):
        potential = self.__potential_x if self.stationary else self.potential(t, self.x)
        EulerSolver.__iterate(self._psi, self.__dpsi, self.dt, self.dx, self.n_points, potential, self.delta_depth)


class CrankNicolsonSolver(Solver):
    __potential_x = None
    __matrix = None
    __diagonal = 0.0j

    def __init__(self, x_max, dx, dt, potential=None, stationary=False):
        Solver.__init__(self, x_max, dx, dt, potential, stationary)
        dim = len(self.x)
        A = _sparse.dok_matrix((dim, dim), dtype=_np.complex)
        dt = self.dt
        dx = self.dx
        self.__diagonal = 1.0 / dt + 0.5j / dx ** 2
        for i in range(dim):
            A[i, i] = self.__diagonal
            A[i, i - 1] = -0.25j / dx ** 2
        for i in range(dim - 1):
            A[i, i + 1] = -0.25j / dx ** 2
        A[dim - 1, 0] = -0.25j / dx ** 2
        self.__matrix = _sparse.csc_matrix(A)

        if self.stationary:
            self.__potential_x = self.potential(self.x)
            self.__potential_x[self.n_points // 2] -= self.delta_depth / self.dx
            self.__matrix[_np.arange(dim), _np.arange(dim)] = self.__diagonal + 0.5j * self.__potential_x

    def iterate(self, t):
        dt = self.dt
        dx = self.dx
        psi = self._psi
        if self.stationary:
            v = self.__potential_x
        else:
            v = self.potential(t, self.x)
            v[self.n_points // 2] -= self.delta_depth / self.dx
            v2 = self.potential(t + self.dt, self.x)
            v2[self.n_points // 2] -= self.delta_depth / self.dx
            dim = len(self.x)
            self.__matrix[_np.arange(dim), _np.arange(dim)] = self.__diagonal + 0.5j * v2

        b = psi / dt + 0.25j * (_np.roll(psi, 1) + _np.roll(psi, -1) - 2 * psi) / dx ** 2 \
            - 0.5j * v * psi
        self._psi = _spsolve(self.__matrix, b)


class SplitOperatorHalfSpectralSolver(Solver):
    __potential_exponent = None
    __momentum_exponent = None

    def __init__(self, x_max, dx, dt, potential=None, stationary=False):
        from scipy.fftpack import fftfreq

        Solver.__init__(self, x_max, dx, dt, potential, stationary)
        if self.stationary:
            v = self.potential(self.x)
            v[self.n_points // 2] -= self.delta_depth / self.dx
            self.__potential_exponent = _np.exp(-0.5j * v * dt)
        p = 2.0 * _np.pi * fftfreq(len(self.x)) / self.dx
        self.__momentum_exponent = _np.exp(-0.5j * p ** 2 * dt)

    def iterate(self, t):
        from scipy.fftpack import fft, ifft
        if self.stationary:
            pot_exp = self.__potential_exponent
        else:
            v = self.potential(t + 0.5 * self.dt, self.x)
            v[self.n_points // 2] -= self.delta_depth / self.dx
            pot_exp = _np.exp(-0.5j * v * self.dt)

        psi = pot_exp * self._psi
        psi = ifft(self.__momentum_exponent * fft(psi))
        self._psi = pot_exp * psi