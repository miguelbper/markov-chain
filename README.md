<div align="center">

# Markov chain
[![Python](https://img.shields.io/badge/Python-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Code Quality](https://github.com/miguelbper/markov-chain/actions/workflows/code-quality.yaml/badge.svg)](https://github.com/miguelbper/markov-chain/actions/workflows/code-quality.yaml)
[![Unit Tests](https://github.com/miguelbper/markov-chain/actions/workflows/tests.yaml/badge.svg)](https://github.com/miguelbper/markov-chain/actions/workflows/tests.yaml)
[![codecov](https://codecov.io/gh/miguelbper/markov-chain/graph/badge.svg)](https://codecov.io/gh/miguelbper/markov-chain)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

</div>

A module offering functions which solve questions about absorbing or ergodic Markov chains. The implementation is based on the theorems from Grinstead and Snell - Introduction to Probability.

## Example for an absorbing Markov chain
Let $\mathbf{P}$ be the transition matrix of an absorbing Markov chain, written in canonical form:

$$
    \mathbf{P} =
    \begin{bmatrix}
        \mathbf{Q} & \mathbf{R} \\
        \mathbf{0} & \mathbf{I} \\
    \end{bmatrix}.
$$

The function `absorbing` computes the matrices/vectors $\mathbf{N}$, $\mathbf{t}$, $\mathbf{B}$, which are defined by

$$
    \begin{array}{rcl}
        N_{ij} & = & E(\text{number of times in $j$, given start in $i$}), \\
        t_{i}  & = & E(\text{number of steps until absorption, given start in $i$}), \\
        B_{ij} & = & P(\text{absorption in $j$, given start in $i$}).
    \end{array}
$$

Example usage:
```python
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
```
Output:
```console
######################################
# Solution of absorbing Markov chain #
######################################

Transition matrix P
-------------------
Def. P = transition matrix
Thm. P[i, j] = P(go to state j given start at i)
P = [  0, 1/2,   0, 1/2,   0]
    [1/2,   0, 1/2,   0,   0]
    [  0, 1/2,   0,   0, 1/2]
    [  0,   0,   0,   1,   0]
    [  0,   0,   0,   0,   1]

Canonical form Q
----------------
Def. P = [[Q R], [0 I]]
Thm. Q = P[:k, :k]
Q = [  0, 1/2,   0]
    [1/2,   0, 1/2]
    [  0, 1/2,   0]

Canonical form R
----------------
Def. P = [[Q R], [0 I]]
Thm. R = P[:k, k:]
R = [1/2,   0]
    [  0,   0]
    [  0, 1/2]

Fundamental matrix (k x k)
--------------------------
Def. N[i, j] := E(# times in j, given start in i)
Thm. N = (I - Q)^(-1) (where I - Q is always invertible)
N = [3/2, 1, 1/2]
    [  1, 2,   1]
    [1/2, 1, 3/2]

Vector of times to absorption (k x 1)
-------------------------------------
Def. t[i] := E(# steps until absorption, given start in i)
Thm. t = N 1
t.T = [3, 4, 3]

Matrix of absorption probabilities (k x (n-k))
----------------------------------------------
Def. B[i, j] := P(absorption in j, given start in i)
Thm. B = N R
B = [3/4, 1/4]
    [1/2, 1/2]
    [1/4, 3/4]
```

## Example for an ergodic Markov chain
Let $\mathbf{P}$ be the transition matrix of an ergodic Markov chain. The function `ergodic` computes the matrices/vectors $\mathbf{w}$, $\mathbf{r}$, $\mathbf{Z}$, $\mathbf{M}$, which are defined by

$$
    \begin{array}{rcl}
        w_{i}      & = & E(\text{number of times in $i$}), \\
        r_{i}      & = & E(\text{number of steps to go from $i$ back to $i$}), \\
        \textbf{Z} & = & (\mathbf{I} - \mathbf{P} + \mathbf{W})^{-1}, \quad \text{where $\mathbf{W} = $ matrix where every row is $\mathbf{w}$}, \\
        M_{ij}     & = & E(\text{number of steps to go from $i$ to $j$}).
    \end{array}
$$

Example usage:
```python
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
```
Output:
```console
####################################
# Solution of ergodic Markov chain #
####################################

Transition matrix P
-------------------
Def. P = transition matrix
Thm. P[i, j] = P(go to state j given start at i)
P = [  0,   1,   0,   0,   0]
    [1/4,   0, 3/4,   0,   0]
    [  0, 1/2,   0, 1/2,   0]
    [  0,   0, 3/4,   0, 1/4]
    [  0,   0,   0,   1,   0]

Fixed vector (1 x n)
--------------------
Def. w[i] := E(# times in i)
Thm. w = w P and sum(w) = 1. ker(I - P) is 1-diml
w = [1/16, 1/4, 3/8, 1/4, 1/16]

Mean recurrence times (1 x n)
-----------------------------
Def. r[i] := E(# steps to go from i back to i)
Thm. r = 1 / w
r = [16, 4, 8/3, 4, 16]

Fundamental matrix (n x n)
--------------------------
Def. Z := (I - P + W)^(-1)
Thm. I - P + W is invertible
Z = [109/96,  19/24, -3/16, -13/24, -19/96]
    [ 19/96,  25/24,  3/16,  -7/24, -13/96]
    [ -1/32,    1/8, 13/16,    1/8,  -1/32]
    [-13/96,  -7/24,  3/16,  25/24,  19/96]
    [-19/96, -13/24, -3/16,  19/24, 109/96]

Matrix of mean passage times (n x n)
------------------------------------
Def. M[i, j] := E(# steps to go from i to j). M[i, i] = 0
Thm. M = (diag(Z) - Z) / w
M = [   0,    1, 8/3, 19/3, 64/3]
    [  15,    0, 5/3, 16/3, 61/3]
    [56/3, 11/3,   0, 11/3, 56/3]
    [61/3, 16/3, 5/3,    0,   15]
    [64/3, 19/3, 8/3,    1,    0]
```
