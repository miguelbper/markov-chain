from sympy import Matrix, Rational

from markov import absorbing, ergodic


class TestAbsorbing:
    def test_drunkard_walk_4(self):
        # fmt: off
        # input
        P = Matrix([
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 2],
        ])/2

        # output
        N = Matrix([
            [3, 2, 1],
            [2, 4, 2],
            [1, 2, 3],
        ])/2
        t = Matrix([[3], [4], [3]])
        B = Matrix([
            [3, 1],
            [2, 2],
            [1, 3],
        ])/4
        # fmt: on

        # predictions
        N_, t_, B_ = absorbing(P)
        assert N == N_ and t == t_ and B == B_

    def test_drunkard_walk_5(self):
        # fmt: off
        # input
        P = Matrix([
            [0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 2],
        ])/2

        # output
        N = Matrix([
            [16, 12,  8,  4],
            [12, 24, 16,  8],
            [ 8, 16, 24, 12],
            [ 4,  8, 12, 16],
        ])/10
        t = Matrix([[4], [6], [6], [4]])
        B = Matrix([
            [8, 2],
            [6, 4],
            [4, 6],
            [2, 8],
        ])/10
        # fmt: on

        # predictions
        N_, t_, B_ = absorbing(P)
        assert N == N_ and t == t_ and B == B_


class TestErgodic:
    def test_land_of_oz(self):
        # fmt: off
        # input
        P = Matrix([
            [2, 1, 1],
            [2, 0, 2],
            [1, 1, 2],
        ]) / 4

        # output
        w = Matrix([[4, 2, 4]])/10
        # fmt: on

        # predictions
        w_, _, _, _ = ergodic(P)
        assert w == w_

    def test_ehrenfest(self):
        # fmt: off
        # input
        P = Matrix([
            [0, 4, 0, 0, 0],
            [1, 0, 3, 0, 0],
            [0, 2, 0, 2, 0],
            [0, 0, 3, 0, 1],
            [0, 0, 0, 4, 0],
        ])/4

        # output
        w = Matrix([[1, 4, 6, 4, 1]])/16
        r = Matrix([[16, 4, Rational(8, 3), 4, 16]])
        M = Matrix([
            [ 0,  3, 8, 19, 64],
            [45,  0, 5, 16, 61],
            [56, 11, 0, 11, 56],
            [61, 16, 5,  0, 45],
            [64, 19, 8,  3,  0],
        ])/3
        # fmt: on

        # predictions
        w_, r_, _, M_ = ergodic(P)
        assert w == w_ and r == r_ and M == M_
