import numpy as _np
import numba as _numba


class Potential:
    def is_stationary(self):
        """
        :return: True if the potential does not depend on time, False otherwise
        """
        return True


class Potential1D(Potential):
    delta_depth = 0.0

    def get_potential(self):
        """
        Returns the function V(x) determining the potential
        :return:
        """
        raise NotImplementedError

    def get_eigenenergy(self, number=0):
        """
        Get the N=`number` energy level
        :param number: 
        :return: 
        """
        raise NotImplementedError

    def get_eigenfunction(self, number=0):
        """
        Get the Nth=`number` eigenfunction
        :param number: 
        :return: 
        """
        raise NotImplementedError

    def get_delta_depth(self):
        """
        :return: the depth of the delta-potential at x=0 if it exists or 0.0 if it doesn't
        """
        return self.delta_depth

    def get_number_of_levels(self):
        """
        :return: the maximum number of levels in the potential or None if there are no levels
        """
        raise NotImplementedError


class DeltaPotential1D(Potential1D):
    def __init__(self, depth):
        """
        Describes a potential V(x) = - depth * delta(x)
        """
        self.delta_depth = depth

    def get_potential(self):
        return _numba.vectorize(lambda x: 0.0 + 0.0j)

    def get_eigenenergy(self, number=0):
        if number != 0:
            raise ValueError(f"Illegal level number {number}, delta potential has only 0th level")
        return - self.delta_depth ** 2 / 2

    def get_eigenfunction(self, number=0):
        if number != 0:
            raise ValueError(f"Illegal level number {number}, delta potential has only 0th level")

        depth = self.delta_depth

        @_numba.vectorize(nopython=True)
        def tmp_function(x):
            if x < 0:
                return _np.sqrt(depth) * _np.exp(+ depth * x) + 0j
            else:
                return _np.sqrt(depth) * _np.exp(- depth * x) + 0j
        return tmp_function

    def get_number_of_levels(self):
        return 1


class QuadraticPotential1D(Potential1D):
    frequency = 0.0

    def __init__(self, frequency=1.0):
        """
        Describes a harmonic oscillator with V(x) = omega ** 2 * x ** 2 / 2 with omega equal to `frequency`
        :param frequency:
        """
        self.frequency = frequency

    def get_potential(self):
        return lambda x: 0.5 * self.frequency ** 2 * x ** 2

    def get_eigenenergy(self, number=0):
        return (number + 0.5) * self.frequency

    def get_eigenfunction(self, number=0):
        from scipy.special import hermite
        from math import factorial, sqrt, pi
        n = number
        w = self.frequency
        return lambda x: 0j + hermite(n)(sqrt(w) * x) * _np.exp(- 0.5 * w * x ** 2) * \
                         ((w / pi) ** 0.25) / sqrt(2 ** n * factorial(n))

    def get_frequency(self):
        return self.frequency

    def get_number_of_levels(self):
        raise Exception("Quadratic potential has an infinite number of levels")


class UniformField1D(Potential1D):
    amplitude = 0.0
    potential = None
    __stationary = True

    def __init__(self, amplitude, potential: Potential1D=None):
        """
        Potenial in uniform electric field V(x) = - E(t) * x + V0(x), where E(t) is determined by `amplitude`, and V0(x) is
        the initial potential
        :param amplitude: either constant value or E(t)
        """
        self.__stationary = not callable(amplitude)
        self.amplitude = amplitude
        if potential is not None:
            self.potential = potential
            self.delta_depth = potential.get_delta_depth()

    def get_potential(self):
        if self.__stationary:
            if self.potential is None:
                return lambda x: - self.amplitude * x
            else:
                return lambda x: - self.amplitude * x + self.potential.get_potential()(x)
        else:
            if self.potential is None:
                return lambda t, x: - self.amplitude(t) * x
            else:
                return lambda t, x: - self.amplitude(t) * x + self.potential.get_potential()(x)

    def get_eigenfunction(self, number=0):
        raise ValueError("No eigenstates in a uniform field")

    def get_eigenenergy(self, number=0):
        raise ValueError("No eigenstates in a uniform field")

    def get_number_of_levels(self):
        return 0

    def is_stationary(self):
        return self.__stationary


class SquarePotential1D(Potential1D):
    depth = 0.0
    width = 0.0

    def __init__(self, depth, width):
        """
        Potential V(x) = -V_0 for |x| < a, and 0.0 elsewhere.
        "V_0" and "a" are determined by `depth` and `width`, respectively
        :param depth:
        :param width:
        """
        self.depth = depth
        self.width = width

    def get_potential(self):
        V0 = self.depth
        a = self.width
        return _numba.vectorize(nopython=True)(lambda x: -V0 if _np.abs(x) < a else 0.0)

    def get_number_of_levels(self):
        return int(_np.floor(1 + 2 * _np.sqrt(2 * self.depth) * self.width / _np.pi))

    def __calculate_k(self, number):
        from scipy.optimize import brentq
        func = lambda k: k * self.width + _np.arcsin(k / _np.sqrt(2 * self.depth)) - _np.pi * (number + 1) / 2
        return brentq(func, 0, _np.sqrt(2 * self.depth))

    def get_eigenenergy(self, number=0):
        if number >= self.get_number_of_levels():
            raise ValueError(f"Level number {number} does not exist. Only {self.get_number_of_levels()} levels.")
        k = self.__calculate_k(number)
        return -self.depth + 0.5 * k ** 2

    def get_eigenfunction(self, number=0):
        if number >= self.get_number_of_levels():
            raise ValueError(f"Level number {number} does not exist. Only {self.get_number_of_levels()} levels.")
        k = self.__calculate_k(number)
        V0 = self.depth
        a = self.width
        alpha = _np.sqrt(2 * V0 - k ** 2)
        if number % 2 == 0:
            A = 1 / _np.sqrt(a + 0.5 * _np.sin(2 * k * a) / k + _np.cos(k * a) ** 2 / alpha)

            @_numba.vectorize(nopython=True)
            def psi(x):
                if x < - a:
                    return A * _np.cos(k * a) * _np.exp(alpha * (x + a))
                elif x < a:
                    return A * _np.cos(k * x)
                else:
                    return A * _np.cos(k * a) * _np.exp(-alpha * (x - a))

            return psi
        else:
            A = 1 / _np.sqrt(a - 0.5 * _np.sin(2 * k * a) / k + _np.sin(k * a) ** 2 / alpha)

            @_numba.vectorize(nopython=True)
            def psi(x):
                if x < -a:
                    return -A * _np.sin(k * a) * _np.exp(alpha * (x + a))
                elif x < a:
                    return A * _np.sin(k * x)
                else:
                    return A * _np.sin(k * a) * _np.exp(-alpha * (x - a))

            return psi

    def get_depth(self):
        return self.depth

    def get_width(self):
        return self.width


class Potential3D(Potential):
    def get_potential(self):
        """
        Returns the function V(r) determining the potential
        :return:
        """
        raise NotImplementedError

    def get_eigenenergy(self, **numbers):
        """
        Get the N=`number` energy level
        :param number:
        :return:
        """
        raise NotImplementedError

    def get_eigenfunction(self, **numbers):
        """
        Get the Nth=`number` eigenfunction
        :param number:
        :return:
        """
        raise NotImplementedError


def _r(x, y, z):
    return _np.sqrt(x * x + y * y + z * z)


def _theta(x, y, z):
    return _np.arctan2(_np.sqrt(x * x + y * y), z)


def _phi(x, y):
    return _np.arctan2(y, x)


class CoulombPotential(Potential3D):
    @staticmethod
    def r_func(n, l, r):
        import math
        from scipy.special import eval_genlaguerre
        return 2 / n ** 2 * math.sqrt(math.factorial(n - l - 1) / math.factorial(n + l)) * \
               _np.exp(-r / n) * (2 * r / n) ** l * eval_genlaguerre(n - l - 1, 2 * l + 1, 2 * r / n)

    def get_potential(self):
        return lambda x, y, z: - 1 / _r(x, y, z)

    @staticmethod
    def _check_numbers(n, l, m):
        for x in (n, l, m):
            if not isinstance(x, int):
                raise ValueError(f"Number {x} is not integer")
        if n < 1:
            raise ValueError(f"Number n={n} must be positive")
        if l < 0 or l >= n:
            raise ValueError(f"Number l={l} should be from 0 to n={n}")
        if abs(m) > l:
            raise ValueError(f"Number m={m} should be from -{l} to {l} for l={l}")

    def get_eigenenergy(self, n, l, m):
        self._check_numbers(n, l, m)
        return - 0.5 / n ** 2

    def get_eigenfunction(self, n, l, m):
        self._check_numbers(n, l, m)
        from scipy.special import sph_harm
        return lambda x, y, z: self.r_func(n, l, _r(x, y, z)) * sph_harm(m, l, _phi(x, y), _theta(x, y, z))