# Markov chain
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
P = np.array([
    [ 0, .5,  0, .5,  0],
    [.5,  0, .5,  0,  0],
    [ 0, .5,  0,  0, .5],
    [ 0,  0,  0,  1,  0],
    [ 0,  0,  0,  0,  1],
])
# use the function absorbing to compute relevant quantities
N, t, B = absorbing(P)
```

## Example for an ergodic Markov chain
Let $\mathbf{P}$ be the transition matrix of an ergodic Markov chain. The function `ergodic` computes the matrices/vectors $\mathbf{w}$, $\mathbf{r}$, $\mathbf{Z}$, $\mathbf{M}$, which are defined by

$$
    \begin{array}{rcl}
        w_{i}      & = & E(\text{number of times in $i$}), \\
        r_{i}      & = & E(\text{number of steps to go from $i$ back to $i$}), \\
        \textbf{Z} & = & (\mathbf{I} - \mathbf{P} + \mathbf{W})^{-1}, \quad \text{where $\mathbf{W} = $ matrix where every row is $\mathbf{w}$}, \\
        M_{ij}     & = & E(\text{\# steps to go from $i$ to $j$}).
    \end{array}
$$

Example usage:
```python
# define the transition matrix
P = np.array([
    [0, 4, 0, 0, 0],
    [1, 0, 3, 0, 0],
    [0, 2, 0, 2, 0],
    [0, 0, 3, 0, 1],
    [0, 0, 0, 4, 0],
])/4
# use the function absorbing to compute relevant quantities
w, r, Z, M = ergodic(P)
```