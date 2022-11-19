import numpy as np
from markov import absorbing, ergodic


class TestAbsorbing():
    def test_drunkard_walk_4(self):
        # input
        P = np.array([[ 0, .5,  0, .5,  0],
                      [.5,  0, .5,  0,  0],
                      [ 0, .5,  0,  0, .5],
                      [ 0,  0,  0,  1,  0],
                      [ 0,  0,  0,  0,  1]])
        
        # output
        N = np.array([[1.5, 1,  .5],
                      [  1, 2,   1],
                      [ .5, 1, 1.5]])
        t = np.array([[3, 4, 3]]).T
        B = np.array([[.75, .25],
                      [ .5,  .5],
                      [.25, .75]])
        
        # predictions
        N_, t_, B_ = absorbing(P)
        assert np.allclose(N, N_) and np.allclose(t, t_) and np.allclose(B, B_)


    def test_drunkard_walk_5(self):
        # input
        P = np.array([
            [ 0, .5,  0,  0, .5,  0],
            [.5,  0, .5,  0,  0,  0],
            [ 0, .5,  0, .5,  0,  0],
            [ 0,  0, .5,  0,  0, .5],
            [ 0,  0,  0,  0,  1,  0],
            [ 0,  0,  0,  0,  0,  1],
        ])
        
        # output
        N = np.array([
            [1.6, 1.2,  .8,  .4],
            [1.2, 2.4, 1.6,  .8],
            [ .8, 1.6, 2.4, 1.2],
            [ .4,  .8, 1.2, 1.6],
        ])
        t = np.array([[4, 6, 6, 4]]).T
        B = np.array([
            [.8, .2],
            [.6, .4],
            [.4, .6],
            [.2, .8],
        ])
        
        # predictions
        N_, t_, B_ = absorbing(P)
        assert np.allclose(N, N_) and np.allclose(t, t_) and np.allclose(B, B_)


class TestErgodic():
    def test_land_of_oz(self):
        # input
        P = np.array([
            [2, 1, 1],
            [2, 0, 2],
            [1, 1, 2],
        ]) / 4
        
        # output
        w = np.array([.4, .2, .4])
        
        # predictions
        w_, _, _, _ = ergodic(P)
        assert np.allclose(w, w_)


    def test_ehrenfest(self):
        # input
        P = np.array([
            [0, 4, 0, 0, 0],
            [1, 0, 3, 0, 0],
            [0, 2, 0, 2, 0],
            [0, 0, 3, 0, 1],
            [0, 0, 0, 4, 0],
        ])/4

        # output
        w = np.array([.0625, .25, .375, .25, .0625])
        r = np.array([16, 4, 8/3, 4, 16])
        M = np.array([
            [ 0,  3, 8, 19, 64],
            [45,  0, 5, 16, 61],
            [56, 11, 0, 11, 56],
            [61, 16, 5,  0, 45],
            [64, 19, 8,  3,  0],
        ])/3

        # predictions
        w_, r_, _, M_ = ergodic(P)
        assert np.allclose(w, w_) and np.allclose(r, r_) and np.allclose(M, M_)