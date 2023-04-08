import numpy as np
from scipy import linalg as slg
from autoarray.util.cholesky_funcs import cholinsertlast, choldeleteindexes

'''
    This file contains functions use the Bro & Jong (1997) algorithm to solve the non-negative least
        square problem. The `fnnls and fix_constraint` is orginally copied from 
        "https://github.com/jvendrow/fnnls".
    For our purpose in PyAutoArray, we create `fnnls_modefied` to take ZTZ and ZTx as inputs directly.
    Furthermore, we add two functions `fnnls_Cholesky and fix_constraint_Cholesky` to realize a scheme
        that solves the lstsq problem in the algorithm by Cholesky factorisation. For ~ 1000 free 
        parameters, we see a speed up by 2 times and should be more for more parameters.
    We have also noticed that by setting the P_initial to be `sla.solve(ZTZ, ZTx, assume_a='pos') > 0`
        will speed up our task (~ 1000 free parameters) by ~ 3 times as it significantly reduces the
        iteration time.
'''


def fnnls(
    Z,
    x,
    P_initial=np.zeros(0, dtype=int),
    lstsq=lambda A, x: slg.solve(A, x, assume_a="pos"),
):
    """
    Implementation of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong.

    This algorithm seeks to find min_d ||x - Zd|| subject to d >= 0

    Some of the comments, such as "B2", refer directly to the steps of
    the fnnls algorithm as presented in the paper by Bro et al.

    Parameters
    ----------
    Z: NumPy array
        Z is an m x n matrix.

    x: Numpy array
        x is a m x 1 vector.

    P_initial: Numpy array, dtype=int
        By default, an empty array. An estimate for
        the indices of the support of the solution.

        lstsq: function
        By default, numpy.linalg.lstsq with rcond=None.
        Least squares function to use when calculating the
        least squares solution min_x ||Ax - b||.
        Must be of the form x = f(A,b).

    Returns
    -------
    d: Numpy array
        d is a nx1 vector
    """

    Z, x, P_initial = map(np.asarray_chkfinite, (Z, x, P_initial))

    m, n = Z.shape

    if len(Z.shape) != 2:
        raise ValueError(
            "Expected a two-dimensional array, but Z is of shape {}".format(Z.shape)
        )
    if len(x.shape) != 1:
        raise ValueError(
            "Expected a one-dimensional array, but x is of shape {}".format(x.shape)
        )
    if len(P_initial.shape) != 1:
        raise ValueError(
            "Expected a one-dimensional array, but P_initial is of shape {}".format(
                P_initial.shape
            )
        )

    if not np.all((P_initial - P_initial.astype(int)) == 0):
        raise ValueError(
            "Expected only integer values, but P_initial has values {}".format(
                P_initial[(P_initial - P_initial.astype(int)) != 0]
            )
        )
    if np.any(P_initial >= n):
        raise ValueError(
            "Expected values between 0 and Z.shape[1], but P_initial has max value {}".format(
                np.max(P_initial)
            )
        )
    if np.any(P_initial < 0):
        raise ValueError(
            "Expected values between 0 and Z.shape[1], but P_initial has min value {}".format(
                np.min(P_initial)
            )
        )
    if P_initial.dtype != np.dtype("int64") and P_initial.dtype != np.dtype("int32"):
        raise TypeError(
            "Expected type int64 or int32, but P_initial is type {}".format(
                P_initial.dtype
            )
        )
    if x.shape[0] != m:
        raise ValueError(
            "Incompatable dimensions. The first dimension of Z should match the length of x, but Z is of shape {} and x is of shape {}".format(
                Z.shape, x.shape
            )
        )

    # Calculating ZTZ and ZTx in advance to improve the efficiency of calculations
    ZTZ = Z.T.dot(Z)
    ZTx = Z.T.dot(x)

    # Declaring constants for tolerance and max repetitions
    epsilon = 2.2204e-16
    tolerance = epsilon * n

    # number of contsecutive times the set P can remain unchanged loop until we terminate
    max_repetitions = 5

    # A1 + A2
    P = np.zeros(n, dtype=np.bool)
    P[P_initial] = True

    # A3
    d = np.zeros(n)

    # A4
    w = ZTx - (ZTZ) @ d

    # Initialize s
    s = np.zeros(n)

    # Count of amount of consecutive times set P has remained unchanged
    no_update = 0

    # Extra loop in case a support is set to update s and d
    if P_initial.shape[0] != 0:

        s[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])
        d = s.clip(min=0)

    # B1
    while (not np.all(P)) and np.max(w[~P]) > tolerance:

        current_P = (
            P.copy()
        )  # make copy of passive set to check for change at end of loop

        # B2 + B3
        P[np.argmax(w * ~P)] = True

        # B4
        s[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])

        # C1
        while np.any(P) and np.min(s[P]) <= tolerance:

            s, d, P = fix_constraint(ZTZ, ZTx, s, d, P, tolerance, lstsq)

        # B5
        d = s.copy()
        # B6
        w = ZTx - (ZTZ) @ d

        # check if there has been a change to the passive set
        if np.all(current_P == P):
            no_update += 1
        else:
            no_update = 0

        if no_update >= max_repetitions:
            break

    res = np.linalg.norm(x - Z @ d)  # Calculate residual loss ||x - Zd||

    return [d, res]


def fnnls_modified(
    ZTZ,
    ZTx,
    P_initial=np.zeros(0, dtype=int),
    lstsq=lambda A, x: slg.solve(A, x, assume_a="pos"),
):
    """
    Implementation of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong.

    This algorithm seeks to find min_d ||x - Zd|| subject to d >= 0

    Some of the comments, such as "B2", refer directly to the steps of
    the fnnls algorithm as presented in the paper by Bro et al.

    Parameters
    ----------
    Z: NumPy array
        Z is an m x n matrix.

    x: Numpy array
        x is a m x 1 vector.

    P_initial: Numpy array, dtype=int
        By default, an empty array. An estimate for
        the indices of the support of the solution.

        lstsq: function
        By default, numpy.linalg.lstsq with rcond=None.
        Least squares function to use when calculating the
        least squares solution min_x ||Ax - b||.
        Must be of the form x = f(A,b).

    Returns
    -------
    d: Numpy array
        d is a nx1 vector
    """

    # Z, x, P_initial = map(np.asarray_chkfinite, (Z, x, P_initial))

    n = np.shape(ZTZ)[0]

    # Calculating ZTZ and ZTx in advance to improve the efficiency of calculations
    # ZTZ = Z.T.dot(Z)
    # ZTx = Z.T.dot(x)

    # Declaring constants for tolerance and max repetitions
    epsilon = 2.2204e-16
    tolerance = epsilon * n

    # number of contsecutive times the set P can remain unchanged loop until we terminate
    max_repetitions = 2

    # A1 + A2
    P = np.zeros(n, dtype=np.bool)
    P[P_initial] = True

    # A3
    d = np.zeros(n)

    # A4
    w = ZTx - (ZTZ) @ d

    # Initialize s
    s = np.zeros(n)

    # Count of amount of consecutive times set P has remained unchanged
    no_update = 0

    #   count = 0
    #    sub_count = 0

    # Extra loop in case a support is set to update s and d
    if P_initial.shape[0] != 0:

        s[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])
        d = s.clip(min=0)

    # B1
    while (not np.all(P)) and np.max(w[~P]) > tolerance:

        #  count += 1

        current_P = (
            P.copy()
        )  # make copy of passive set to check for change at end of loop

        # B2 + B3
        P[np.argmax(w * ~P)] = True

        # B4
        s[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])

        # C1
        while np.any(P) and np.min(s[P]) <= tolerance:

            #      sub_count += 1

            s, d, P = fix_constraint(ZTZ, ZTx, s, d, P, tolerance, lstsq)

        # B5
        d = s.copy()
        # B6
        w = ZTx - (ZTZ) @ d

        # check if there has been a change to the passive set
        if np.all(current_P == P):
            no_update += 1
        else:
            no_update = 0

        if no_update >= max_repetitions:
            break

    # res = np.linalg.norm(x - Z@d) #Calculate residual loss ||x - Zd||

    # print(f"Total Iterations = {count} / {sub_count}")

    return d


def fix_constraint(
    ZTZ,
    ZTx,
    s,
    d,
    P,
    tolerance,
    lstsq=lambda A, x: slg.solve(A, x, assume_a="pos"),
):
    """
    The inner loop of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong.

    One iteration of the loop to adjust the new estimate s to satisfy the
    nonnegativity contraint of the solution.

    Some of the comments, such as "B2", refer directly to the steps of
    the fnnls algorithm as presented in the paper by Bro et al.

    Parameters
    ----------
    ZTZ: NumPy array
        ZTZ is an n x n matrix equal to Z.T * Z

    ZTx: Numpy array
        ZTx is an n x 1 vector equal to Z.T * x

    s: Numpy array
        The new estimate of the solution with possible
        negative values that do not meet the constraint

    d: Numpy array
        The previous estimate of the solution that satisfies
        the nonnegativity contraint

    P: Numpy array, dtype=np.bool
        The current passive set, which comtains the indices
        that are not fixed at the value zero.

    tolerance: float
        A tolerance, below which values are considered to be
        0, allowing for more reasonable convergence.

    lstsq: function
        By default, numpy.linalg.lstsq with rcond=None.
        Least squares function to use when calculating the
        least squares solution min_x ||Ax - b||.
        Must be of the form x = f(A,b).

    Returns
    -------
    s: Numpy array
        The updated new estimate of the solution.
    d: Numpy array
        The updated previous estimate, now as close
        as possible to s while maintaining nonnegativity.
    P: Numpy array, dtype=np.bool
        The updated passive set
    """
    # C2
    q = P * (s <= tolerance)
    alpha = np.min(d[q] / (d[q] - s[q]))

    # C3
    d = d + alpha * (
        s - d
    )  # set d as close to s as possible while maintaining non-negativity

    # C4
    P[d <= tolerance] = False

    # C5
    s[P] = lstsq((ZTZ)[P][:, P], (ZTx)[P])

    # C6
    s[~P] = 0.0

    return s, d, P

                                                                                                     
def fnnls_Cholesky(ZTZ, ZTx, P_initial = np.zeros(0, dtype=int),
            lstsq = lambda A, x: slg.solve(A, x, assume_a='pos')):
    """                                                                                              
        Similar to fnnls, but use solving the lstsq problem by updating Cholesky factorisation.
    """                                                                                              
    #print('Cholesky start!')
    n = np.shape(ZTZ)[0]
    epsilon = 2.2204e-16                                                                             
    tolerance = epsilon * n
    max_repetitions = 5
    no_update = 0
    loop_count = 0

    P = np.zeros(n, dtype=np.bool)
    P[P_initial] = True                                                                                  
    d = np.zeros(n)
    w = ZTx - (ZTZ) @ d
    s_chol = np.zeros(n)

    if P_initial.shape[0] != 0:
        P_number = np.arange(len(P), dtype='int')
        P_inorder = P_number[P_initial]
        s_chol[P] = lstsq((ZTZ)[P][:,P], (ZTx)[P])                                                   
        d = s_chol.clip(min=0)                                                                       
    else:
        P_inorder = np.array([], dtype='int')

    # P_inorder is similar as P. They are both used to select solutions in the passive set.
    # P_inorder saves the `indexes` of those passive solutions. 
    # P saves [True/False] for all solutions. True indicates a solution in the passive set while False
    #     indicates it's in the active set.
    # The benifit of P_inorder is that we are able to not only select out solutions in the passive set
    #     and can sort them in the order of added to the passive set. This will make updating the
    #     Cholesky factorisation simpler and thus save time. 
                                                                                                     
    while (not np.all(P))  and np.max(w[~P]) > tolerance:
        current_P = P.copy()
        # make copy of passive set to check for change at end of loop
        idmax = np.argmax(w * ~P)                                                                    
        P_inorder = np.append(P_inorder, int(idmax))                                                 
        if loop_count == 0:                                                                          
            U = slg.cholesky(ZTZ[P_inorder][:, P_inorder])
            # We need to initialize the Cholesky factorisation, U, for the first loop.
        else:                                                                                        
            U = cholinsertlast(U, ZTZ[idmax][P_inorder])
        s_chol[P_inorder] = slg.cho_solve((U, False), ZTx[P_inorder])
        # solve the lstsq problem by cho_solve
        P[idmax] = True
        while np.any(P) and np.min(s_chol[P]) <= tolerance:                                          
            s_chol, d, P, P_inorder, U = fix_constraint_Cholesky(
                        ZTZ=ZTZ, 
                        ZTx=ZTx,
                        s_chol=s_chol,
                        d=d,
                        P=P,
                        P_inorder=P_inorder,
                        U=U,
                        tolerance=tolerance)
                                                                                                     
        d = s_chol.copy()
        w = ZTx - (ZTZ) @ d
        loop_count += 1                                                                              
        if(np.all(current_P == P)):                                                                  
            no_update += 1                                                                           
        else:                                                                                        
            no_update = 0
        if no_update >= max_repetitions:                                                             
            break                                                                                    
                                                                                                     
    return d                    

def fix_constraint_Cholesky(ZTZ, ZTx, s_chol, d, P, P_inorder, U, tolerance):        
    """
        Similar to fix_constraint, but solve the lstsq by Cholesky factorisation.
        If this function is called, it means some solutions in the current passive sets needed to be
            taken out and put into the active set.
        So, this function involves 3 procedure:
            1. Identifying what solutions should be taken out of the current passive set.
            2. Updating the P, P_inorder and the Cholesky factorisation U.
            3. Solving the lstsq by using the new Cholesky factorisation U.
        As some solutions are taken out from the passive set, the Cholesky factorisation needs to be
                updated by choldeleteindexes. To realize that, we call the `choldeleteindexes` from
                cholesky_funcs.
    """                                                                                          
    q = P * (s_chol <= tolerance)                                                                    
    alpha = np.min(d[q] / (d[q] - s_chol[q]))
    d = d + alpha * (s_chol - d) #set d as close to s as possible while maintaining non-negativity

    id_delete = np.where(d[P_inorder] <= tolerance)[0]
    U = choldeleteindexes(U, id_delete) # update the Cholesky factorisation
    P_inorder = np.delete(P_inorder, id_delete) # update the P_inorder                        
    P[d <= tolerance] = False # update the P

    s_chol[P_inorder] = slg.cho_solve((U, False), ZTx[P_inorder])
    # solve the lstsq problem by cho_solve
    s_chol[~P] = 0.0 # set solutions taken out of the passive set to be 0

    return s_chol, d, P, P_inorder, U       



