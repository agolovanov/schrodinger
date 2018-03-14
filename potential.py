import numpy as _np
import numba as _numba


class Potential():
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


class DeltaPotential(Potential):
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


class QuadraticPotential(Potential):
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


class UniformField(Potential):
    amplitude = 0.0
    potential = None

    def __init__(self, amplitude, potential: Potential=None):
        """
        Potenial in uniform electric field V(x) = - E * x + V0(x), where E is determined by `amplitude`, and V0(x) is
        the initial potential
        """
        self.amplitude = amplitude
        if potential is not None:
            self.potential = potential
            self.delta_depth = potential.get_delta_depth()

    def get_potential(self):
        if self.potential is None:
            return lambda x: - self.amplitude * x
        else:
            return lambda x: - self.amplitude * x + self.potential.get_potential()(x)

    def get_eigenfunction(self, number=0):
        raise ValueError("No eigenstates in a uniform field")

    def get_eigenenergy(self, number=0):
        raise ValueError("No eigenstates in a uniform field")