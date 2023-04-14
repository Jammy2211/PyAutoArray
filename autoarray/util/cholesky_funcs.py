import numpy as np
from scipy import linalg
import math
import time
from numba import njit


@njit(fastmath=True)
def _choldowndate(U, x):
    n = x.size
    for k in range(n - 1):
        Ukk = U[k, k]
        xk = x[k]
        r = math.sqrt(Ukk**2 - xk**2)
        c = r / Ukk
        s = xk / Ukk
        U[k, k] = r
        U[k, k + 1 :] = (U[k, (k + 1) :] - s * x[k + 1 :]) / c
        x[k + 1 :] = c * x[k + 1 :] - s * U[k, k + 1 :]

    k = n - 1
    U[k, k] = math.sqrt(U[k, k] ** 2 - x[k] ** 2)
    return U


@njit(fastmath=True)
def _cholupdate(U, x):

    n = x.size
    for k in range(n - 1):

        Ukk = U[k, k]
        xk = x[k]

        r = np.sqrt(Ukk**2 + xk**2)

        c = r / Ukk
        s = xk / Ukk
        U[k, k] = r

        U[k, k + 1 :] = (U[k, (k + 1) :] + s * x[k + 1 :]) / c
        x[k + 1 :] = c * x[k + 1 :] - s * U[k, k + 1 :]

    k = n - 1
    U[k, k] = np.sqrt(U[k, k] ** 2 + x[k] ** 2)

    return U


def cholinsert(U, index, x):

    S = np.insert(np.insert(U, index, 0, axis=0), index, 0, axis=1)

    S[:index, index] = S12 = linalg.solve_triangular(
        U[:index, :index], x[:index], trans=1, lower=False, overwrite_b=True
    )

    S[index, index] = s22 = math.sqrt(x[index] - S12.dot(S12))

    if index == U.shape[0]:
        return S
    else:
        S[index, index + 1 :] = S23 = (x[index + 1 :] - S12.T @ U[:index, index:]) / s22
        _choldowndate(S[index + 1 :, index + 1 :], S23)  # S33
        return S


def cholinsertlast(U, x):
    """
    Update the Cholesky matrix U by inserting a vector at the end of the matrix
    Inserting a vector to the end of U doesn't require _cholupdate, so save some time.
    It's a special case of `cholinsert` (as shown above, if index == U.shape[0])
    As in current Cholesky scheme implemented in fnnls, we only use this kind of insertion, so I
        separate it out from the `cholinsert`.
    """
    index = U.shape[0]

    S = np.insert(np.insert(U, index, 0, axis=0), index, 0, axis=1)

    S[:index, index] = S12 = linalg.solve_triangular(
        U[:index, :index], x[:index], trans=1, lower=False, overwrite_b=True
    )

    S[index, index] = s22 = math.sqrt(x[index] - S12.dot(S12))

    return S


def choldeleteindexes(U, indexes):

    indexes = sorted(indexes, reverse=True)

    for index in indexes:

        L = np.delete(np.delete(U, index, axis=0), index, axis=1)

        # If the deleted index is at the end of matrix, then we do not need to update the U.

        if index == L.shape[0]:
            U = L
        else:
            _cholupdate(L[index:, index:], U[index, index + 1 :])
            U = L

    return U
