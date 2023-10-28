'''This module contains functions to solve absorbing and ergodic Markov
chains. The functions absorbing and ergodic implement the theorems from
Grinstead and Snell - Introduction to Probability.

Typical usage examples:

1. absorbing Markov chain:
    # define the transition matrix, in canonical form
    P = np.array([
        [ 0, .5,  0, .5,  0],
        [.5,  0, .5,  0,  0],
        [ 0, .5,  0,  0, .5],
        [ 0,  0,  0,  1,  0],
        [ 0,  0,  0,  0,  1],
    ])
    # use the function absorbing to compute relevant quantities
    N, t, B = absorbing(P)
    
2. ergodic Markov chain:
    # define the transition matrix
    P = np.array([
        [0, 4, 0, 0, 0],
        [1, 0, 3, 0, 0],
        [0, 2, 0, 2, 0],
        [0, 0, 3, 0, 1],
        [0, 0, 0, 4, 0],
    ])/4
    # use the function ergodic to compute relevant quantities
    w, r, Z, M = ergodic(P)

'''

import numpy as np
from sympy import Matrix, eye, ones, zeros


def absorbing(P: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    '''Given an absorbing Markov chain with transition matrix P, compute
    t (time to absorption), B (absorption probs), N (fundamental matrix)

    Args:
        P (Matrix): Transition matrix (n x n, where n = # states) of 
            the absorbing Markov chain. We assume that P is given in 
            canonical form:
                P = [Q R]   (k transient states)
                    [0 I]   (n-k absorbing states).

    Returns:
        N (Matrix): Fundamental matrix (k x k).
            Def. N[i, j] := E(# times in j, given start in i).
            Thm. N = (I - Q)^(-1) (where I - Q is always invertible).

        t (Matrix): Vector of times to absorption (k x 1).
            Def. t[i] := E(# steps until absorption, given start in i).
            Thm. t = N c, where c = column vector with every entry 1

        B (Matrix): Matrix of absorption probabilities (k x (n-k)).
            Def. B[i, j] := P(absorption in j, given start in i).
            Thm. B = N R.
    
    '''
    n, _ = P.shape

    print(P)

    # compute k = number of transient states
    J = eye(n)
    print(J)
    k = n - 1
    while P[k, :] == J[k, :]:
        print(P[k, :])
        print(J[k, :])
        k -= 1
    k += 1

    # auxiliary matrices
    c = ones(k, 1)
    I = eye(k)
    Q = P[:k, :k]
    R = P[:k, k:]
    
    print(I)
    print(Q)
    print(I - Q)

    # compute N, t, B
    N = (I - Q).inv()
    t = N @ c
    B = N @ R

    return N, t, B


def ergodic(P: Matrix) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    '''Given an ergodic Markov chain with transition matrix P, compute
    w (fixed vector), r (mean recurrence times), Z (fundamental matrix),
    and M (mean first passage times).

    Args:
        P (Matrix): Transition matrix (n x n, where n = # states) of
        the ergodic Markov chain.

    Returns:
        w (Matrix): Fixed vector (1 x n).
            Def. w[i] := E(# times in i)
            Def. W := n x n matrix with every row equal to w
            Thm. w = w P and sum(w) = 1. ker(I - P) is 1-diml.

        r (Matrix): Mean recurrence times (1 x n).
            Def. r[i] := E(# steps to go from i back to i)
            Thm. r[i] = 1 / w[i]

        Z (Matrix): Fundamental matrix (n x n).
            Thm. I - P + W is invertible
            Def. Z := (I - P + W)^(-1)

        M (Matrix): Matrix of mean passage times (n x n).
            Def. M[i, j] := E(# steps to go from i to j). M[i, i] = 0.
            Thm. M = (diag(Z) - Z) / w    (broadcasting in this eq.)
        
    '''
    n, _ = P.shape
    I = eye(n)

    # compute fixed vector w, using w P = w and sum(w) = 1.
    A = ones(1, n).col_join((P.T - I)[1:, :])
    b = ones(1, 1).col_join(zeros(n - 1, 1))
    w = (A.inv() @ b).T
    W = ones(n, 1) @ w

    # compute r, Z, M
    r = Matrix([[1/w[0, i] for i in range(n)]])
    Z = (I - P + W).inv()
    M = Matrix([[(Z[j,j] - Z[i,j])/w[0,j] for j in range(n)] for i in range(n)])

    return w, r, Z, M