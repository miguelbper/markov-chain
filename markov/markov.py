"""This module contains functions to solve absorbing and ergodic Markov.

chains. The functions absorbing and ergodic implement the theorems from
Grinstead and Snell - Introduction to Probability.

Typical usage examples:

1. absorbing Markov chain:
    # define the transition matrix, in canonical form
    P = Matrix([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 2],
    ])/2
    # use the function absorbing to compute relevant quantities
    N, t, B = absorbing(P)

2. ergodic Markov chain:
    # define the transition matrix
    P = Matrix([
        [0, 4, 0, 0, 0],
        [1, 0, 3, 0, 0],
        [0, 2, 0, 2, 0],
        [0, 0, 3, 0, 1],
        [0, 0, 0, 4, 0],
    ])/4
    # use the function ergodic to compute relevant quantities
    w, r, Z, M = ergodic(P)
"""

from sympy import Matrix, eye, ones, zeros


def absorbing(P: Matrix, verbose=False) -> tuple[Matrix, Matrix, Matrix]:
    """Given an absorbing Markov chain with transition matrix P, compute t
    (time to absorption), B (absorption probs), N (fundamental matrix)

    Args:
        P (Matrix): Transition matrix (n x n, where n = # states) of
            the absorbing Markov chain. We assume that P is given in
            canonical form:
                P = [Q R]   (k transient states)
                    [0 I]   (n-k absorbing states).

    Returns:
        N (Matrix): Fundamental matrix (k x k)
            Def. N[i, j] := E(# times in j, given start in i)
            Thm. N = (I - Q)^(-1) (where I - Q is always invertible)

        t (Matrix): Vector of times to absorption (k x 1)
            Def. t[i] := E(# steps until absorption, given start in i)
            Thm. t = N 1

        B (Matrix): Matrix of absorption probabilities (k x (n-k))
            Def. B[i, j] := P(absorption in j, given start in i)
            Thm. B = N R
    """
    n, _ = P.shape

    # compute k = number of transient states
    J = eye(n)
    k = n - 1
    while P[k, :] == J[k, :]:
        k -= 1
    k += 1

    # auxiliary matrices
    c = ones(k, 1)
    I = eye(k)
    Q = P[:k, :k]
    R = P[:k, k:]

    # compute N, t, B
    N = (I - Q).inv()
    t = N @ c
    B = N @ R

    if verbose:
        txt = "Solution of absorbing Markov chain"
        print("\n")
        print("#" * (len(txt) + 4))
        print("# " + txt + " #")
        print("#" * (len(txt) + 4))

        # P
        _print(
            title="Transition matrix P",
            definition="P = transition matrix",
            theorem="P[i, j] = P(go to state j given start at i)",
            symbol="P",
            A=P,
        )

        # Q
        _print(
            title="Canonical form Q",
            definition="P = [[Q R], [0 I]]",
            theorem="Q = P[:k, :k]",
            symbol="Q",
            A=Q,
        )

        # R
        _print(
            title="Canonical form R",
            definition="P = [[Q R], [0 I]]",
            theorem="R = P[:k, k:]",
            symbol="R",
            A=R,
        )

        # N
        _print(
            title="Fundamental matrix (k x k)",
            definition="N[i, j] := E(# times in j, given start in i)",
            theorem="N = (I - Q)^(-1) (where I - Q is always invertible)",
            symbol="N",
            A=N,
        )

        # t
        _print(
            title="Vector of times to absorption (k x 1)",
            definition="t[i] := E(# steps until absorption, given start in i)",
            theorem="t = N 1",
            symbol="t.T",
            A=t.T,
        )

        # B
        _print(
            title="Matrix of absorption probabilities (k x (n-k))",
            definition="B[i, j] := P(absorption in j, given start in i)",
            theorem="B = N R",
            symbol="B",
            A=B,
        )

    return N, t, B


def ergodic(P: Matrix, verbose=False) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    """Given an ergodic Markov chain with transition matrix P, compute w (fixed
    vector), r (mean recurrence times), Z (fundamental matrix), and M (mean
    first passage times).

    Args:
        P (Matrix): Transition matrix (n x n, where n = # states) of
        the ergodic Markov chain.

    Returns:
        w (Matrix): Fixed vector (1 x n)
            Def. w[i] := E(# times in i)
            Def. W := n x n matrix with every row equal to w
            Thm. w = w P and sum(w) = 1. ker(I - P) is 1-diml

        r (Matrix): Mean recurrence times (1 x n)
            Def. r[i] := E(# steps to go from i back to i)
            Thm. r[i] = 1 / w[i]

        Z (Matrix): Fundamental matrix (n x n)
            Thm. I - P + W is invertible
            Def. Z := (I - P + W)^(-1)

        M (Matrix): Matrix of mean passage times (n x n)
            Def. M[i, j] := E(# steps to go from i to j). M[i, i] = 0
            Thm. M = (diag(Z) - Z) / w    (broadcasting in this eq.)
    """
    n, _ = P.shape
    I = eye(n)

    # compute fixed vector w, using w P = w and sum(w) = 1.
    A = ones(1, n).col_join((P.T - I)[1:, :])
    b = ones(1, 1).col_join(zeros(n - 1, 1))
    w = (A.inv() @ b).T
    W = ones(n, 1) @ w

    # compute r, Z, M
    r = Matrix([[1 / w[0, i] for i in range(n)]])
    Z = (I - P + W).inv()
    M = Matrix([[(Z[j, j] - Z[i, j]) / w[0, j] for j in range(n)] for i in range(n)])

    if verbose:
        txt = "Solution of ergodic Markov chain"
        print("\n")
        print("#" * (len(txt) + 4))
        print("# " + txt + " #")
        print("#" * (len(txt) + 4))

        # P
        _print(
            title="Transition matrix P",
            definition="P = transition matrix",
            theorem="P[i, j] = P(go to state j given start at i)",
            symbol="P",
            A=P,
        )

        # w
        _print(
            title="Fixed vector (1 x n)",
            definition="w[i] := E(# times in i)",
            theorem="w = w P and sum(w) = 1. ker(I - P) is 1-diml",
            symbol="w",
            A=w,
        )

        # r
        _print(
            title="Mean recurrence times (1 x n)",
            definition="r[i] := E(# steps to go from i back to i)",
            theorem="r = 1 / w",
            symbol="r",
            A=r,
        )

        # Z
        _print(
            title="Fundamental matrix (n x n)",
            definition="Z := (I - P + W)^(-1)",
            theorem="I - P + W is invertible",
            symbol="Z",
            A=Z,
        )

        # M
        _print(
            title="Matrix of mean passage times (n x n)",
            definition="M[i, j] := E(# steps to go from i to j). M[i, i] = 0",
            theorem="M = (diag(Z) - Z) / w",
            symbol="M",
            A=M,
        )

    return w, r, Z, M


def _print(
    title: str,
    definition: str,
    theorem: str,
    symbol: str,
    A: Matrix,
) -> None:
    equals = " = "
    print("\n" + title)
    print("-" * len(title))
    print("Def. " + definition)
    print("Thm. " + theorem)
    print(symbol + equals, end="")

    m, n = A.shape
    L = len(symbol) + len(equals)
    M = [max(len(str(A[i, j])) for i in range(m)) for j in range(n)]

    for i in range(m):
        print(" " * (L * bool(i)), end="")
        print("[", end="")
        for j in range(n):
            print(f"{str(A[i, j]):>{M[j]}}" + (", " if j < n - 1 else ""), end="")
        print("]")
