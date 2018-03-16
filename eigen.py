import numpy as _np


def calculate_eigenstates(x, potential, n=1, energy_guess=0.0):
    """
    Calculates `n` eigenstates in the potential V(x).
    :param x: array of x coordinates
    :param potential: function V(x)
    :param n: number of eigenstates to calculate
    :param energy_guess: the expected level of energy of the first eigenstate
    :return: a tuple of eigenenergies (as a np.array) and eigenfunctions (as a list of np.array-s)
    """
    from scipy import sparse
    from scipy.sparse.linalg import eigsh
    import wavefunction

    dim = x.shape[0]
    dx = x[1] - x[0]
    v_arr = potential(x)
    A = sparse.dok_matrix((dim, dim))
    for i in range(dim):
        A[i, i] = 1.0 / dx ** 2 + v_arr[i]
        A[i, i - 1] = -0.5 / dx ** 2
    for i in range(dim - 1):
        A[i, i + 1] = -0.5 / dx ** 2
    A[dim - 1, 0] = -0.5 / dx ** 2
    A = sparse.csc_matrix(A)
    w, vectors = eigsh(A, k=n, sigma=energy_guess)
    psis0 = [v / _np.sqrt(wavefunction.norm(x, v)) for v in vectors.T]
    return w, psis0