#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.stats

def almost_eq(a, b):
    return (np.fabs(a - b) <= 1e-5).all()

def rank(Q):
    _, S, _ = np.linalg.svd(Q)
    return int(np.sum(S >= 1e-10))

def random_orthogonal_matrix(m, n):
    Q, _ = np.linalg.qr(np.random.random((m, n)))
    assert rank(Q) == min(m, n)
    return Q

def random_orthonormal_matrix(n):
    Q = random_orthogonal_matrix(n, n)
    assert almost_eq(np.dot(Q.T, Q), np.eye(n))
    return Q

def sample(target_ppdf, proposal_cond_ppdf, proposal_cond_draw, niters, x0, rng):
    """
    target_ppdf        - (proportional) density
    proposal_cond_ppdf - conditioned on arg0, (proportional) density
    proposal_cond_draw - conditioned on arg0, draw a value
    """

    xt = x0
    values = []
    while niters:
        # sample xprop ~ Q(.|xt)
        xprop = proposal_cond_draw(xt, rng)

        alpha_1 = target_ppdf(xprop)/target_ppdf(xcur)
        alpha_2 = proposal_cond_ppdf(xprop, xt)/proposal_cond_ppdf(xt, xprop)
        alpha   = alpha_1*alpha_2

        if alpha >= 1.0 or rng.random() <= alpha:
            xt = xprop

        niters -= 1
        values.append(xt)

    return np.array(values)

def normal_ppdf(x, mean, cov):
    """
    assumes cov is PD
    """


def main():

    mean = np.ones(2)
    Q    = random_orthonormal_matrix(2)
    cov  = np.dot(np.dot(Q, np.diag([1.0, 0.5])), Q.T)
    target_ppdf = lambda x: sp.stats.multivariate_normal.pdf(x, mean=mean, cov=cov)

    sigma2 = 0.1
    proposal_cond_ppdf = lambda xt, x: sp.stats.multivariate_normal.pdf(x, mean=xt, cov=sigma2*np.eye(2))
    proposal_cond_draw = lambda xt, rng: rng.multivariate_normal(mean=xt, cov=sigma2*np.eye(2))

    sample(target_ppdf, proposal_cond_ppdf, proposal_cond_draw, 100, np.zeros(2), np.random)

if __name__ == '__main__':
    main()
