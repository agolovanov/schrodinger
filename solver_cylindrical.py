import numpy as _np
import numba as _numba
import potential as _potential


class Solver():
    r = None
    z = None
    dr = None
    dz = None
    dt = None
    nr_points = None
    nz_points = None
    potential = None
    stationary = False

    def __init__(self, r_max, dr, z_max, dz, dt, potential=None, stationary=False):
        """
        Creates a solver in cylindrical space for the intervals `[0; r_max]` and `[-z_max, z_max]`
        with spatial steps `dr` and `dz` and a time step `dt`.
        :param r_max:
        :param dr:
        :param z_max
        :param dz
        :param dt:
        :param potential:
        :param stationary: if True, the potential is considered constant in time
        """
        self.dr = dr
        self.dt = dt
        nr_tmp = int((r_max + 0.99 * dr) / dr) + 1
        nz_tmp = int(z_max / dz)
        if nz_tmp * dz < z_max - 0.01 * dz:
            nz_tmp += 1

        self.nr_points = nr_tmp
        self.nz_points = nz_tmp
        r_tmp = _np.linspace(0, (self.nr_points - 1) * dr, self.nr_points)
        z_tmp = _np.linspace(- nz_tmp * dz, nz_tmp * dz, self.n_points)
        self.r, self.z = _np.meshgrid(r_tmp, z_tmp)
        self.stationary = stationary
        if potential is None:
            self.potential = _numba.vectorize(nopython=True)(lambda r, z: 0.0j)
            self.stationary = True
        elif isinstance(potential, _potential.Potential):
            if not isinstance(potential, _potential.Potential3D):
                raise ValueError("Only 3D potentials are allowed")
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