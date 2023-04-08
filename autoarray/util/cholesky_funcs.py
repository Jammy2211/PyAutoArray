import numpy as np
from scipy import linalg
import math 
import numba

'''
    Cholesky related functions that might be helpful for building the Cholesky scheme for fnnls.
    For the scheme currently used, we only use four functions listed here:
        * _cholupdate (might be replaced by _numba_cholupdate for speed up)
        * choldeleteindex 
        * choldeleteindexes
        * cholinsertlast
    Other possible schemes may use other functions, so we leave them here for future development.
'''

def cholupdate(U, x, lower=False):
    if lower:
        return cholupdate(np.transpose(U), x).T
        
    U = np.array(U)
    x = np.array(x)
    return _cholupdate(U, x)

@numba.jit(nopython=True)
def _numba_cholupdate(U, x, sgn=1):
    n = x.size
    for k in range(n):
        Ukk = U[k, k]
        xk = x[k]
        r = math.sqrt(Ukk**2 + sgn * xk**2)
        c = r / Ukk
        s = xk / Ukk
        U[k, k] = r
        for j in range(k + 1, n):
            U[k, j] = (U[k, j] + sgn * s * x[j]) / c
            x[j] = c * x[j] - s * U[k, j]
    return U

def _cholupdate(U, x):
    n = x.size
    for k in range(n - 1):
        Ukk = U[k, k]
        xk = x[k]
        r = math.sqrt(Ukk**2 + xk**2)
        c = r / Ukk
        s = xk / Ukk
        U[k, k] = r
        U[k, k+1:] = (U[k, (k+1):] + s * x[k+1:]) / c
        x[k + 1:] = c * x[k+1:] - s * U[k, k+1:]
    
    k = n - 1
    U[k, k] = math.sqrt(U[k, k]**2 + x[k]**2)
    return U

def choldowndate(U, x, lower=False):
    if lower:
        return choldowndate(np.transpose(U), x).T
        
    U = np.array(U)
    x = np.array(x)
    return _choldowndate(U, x)

def _choldowndate(U, x):
    n = x.size
    for k in range(n - 1):
        Ukk = U[k, k]
        xk = x[k]
        r = math.sqrt(Ukk**2 - xk**2)
        c = r / Ukk
        s = xk / Ukk
        U[k, k] = r
        U[k, k+1:] = (U[k, (k+1):] - s * x[k+1:]) / c
        x[k + 1:] = c * x[k+1:] - s * U[k, k+1:]
    
    k = n - 1
    U[k, k] = math.sqrt(U[k, k]**2 - x[k]**2)
    return U

def choldeleteindex(U, index):
    '''
        This function is called in the `fix_constraint_Cholesky` of fnnls.
        Although we have _numba_cholupdate here, but we doesn't use it as we doesn't gain much speed
            up for my tests (1000 source pixels). 
    '''
    s23 = U[index, index + 1:]
    L = np.delete(np.delete(U, index, axis=0), index, axis=1)
    if len(s23) == 0:
        # If the deleted index is at the end of matrix, then we do not need to update the U.
        return L
    else:
        _cholupdate(L[index:, index:], s23)
        return L

def choldeleteindexes(U, indexes):
    indexes = sorted(indexes, reverse=True)
    for i in indexes:
        U = choldeleteindex(U, i)

    return U

def cholinsert(U, index, x, lower=False):
    if lower:
        return cholinsert(np.transpose(U), index, x).T

    S = np.insert(np.insert(U, index, 0, axis=0), index, 0, axis=1)
    S[:index, index] = S12 = linalg.solve_triangular(U[:index, :index], x[:index], trans=1, lower=False)
    S[index, index] = s22 = math.sqrt(x[index] - S12.dot(S12))
    if index == U.shape[0]:
        return S
    else:
        S[index, index + 1:] = S23 = (x[index + 1:] - S12.T @ U[:index, index:]) / s22
        _choldowndate(S[index + 1:, index + 1:], S23) # S33
        return S

def cholinsertlast(U, x, lower=False):
    '''
        Update the Cholesky matrix U by inserting a vector at the end of the matrix
        Inserting a vector to the end of U doesn't require _cholupdate, so save some time.
        It's a special case of `cholinsert` (as shown above, if index == U.shape[0])
        As in current Cholesky scheme implemented in fnnls, we only use this kind of insertion, so I
            separate it out from the `cholinsert`.
    '''
    index = U.shape[0]
    if lower:
        return cholinsert(np.transpose(U), index, x).T
    S = np.insert(np.insert(U, index, 0, axis=0), index, 0, axis=1)
    S[:index, index] = S12 = linalg.solve_triangular(U[:index, :index], x[:index], trans=1, lower=False)
    S[index, index] = s22 = math.sqrt(x[index] - S12.dot(S12))
    return S


