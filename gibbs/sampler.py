#!/usr/bin/env python
import numpy as np
import scipy as sp
import scipy.special
import itertools as it
import matplotlib.pylab as plt
import math

def discrete_sample(pmf):
    # XXX: does numpy have something to do this?
    coin = np.random.random()
    cdf = np.cumsum(pmf)
    a = np.where(coin >= cdf)[0]
    if not a.shape[0]:
        return 0
    return min(a[-1]+1, pmf.shape[0]-1)

def almost_eq(a, b):
    return math.fabs(a - b) <= 1e-5

def kl(a, b):
    assert a.shape == b.shape
    assert almost_eq(a.sum(), 1)
    assert almost_eq(b.sum(), 1)
    return np.sum([p*np.log(p/q) for p, q in zip(a, b)])

def gibbs_beta_bernoulli(Y, K, alpha, beta, gamma, niters):
    N, D = Y.shape
    alpha, beta, gamma = map(float, [alpha, beta, gamma])

    # start with random assignment
    assignments = np.random.randint(0, K, size=N)

    # initialize the sufficient statistics (cluster sums) accordingly
    sums = np.zeros((K, D))
    cnts = np.zeros(K)
    for yi, ci in zip(Y, assignments):
        sums[ci] += yi
        cnts[ci] += 1

    history = np.zeros((niters, N))
    for t in xrange(niters):
        for i, (yi, ci) in enumerate(zip(Y, assignments)):
            # remove from SS
            sums[ci] -= yi
            cnts[ci] -= 1

            # build P(c_i=k | c_{\i}, Y)
            def fn(k):
                lg_term1 = np.log( cnts[k] + alpha/K )
                lg_term2 = np.log( N - 1 + alpha )
                lg_term3 = D*np.log( beta + gamma + cnts[k] + 1 )
                def fn1(tup):
                    d, yid = tup
                    if yid:
                        return np.log(beta + sums[k, d] + 1)
                    else:
                        return np.log(gamma + cnts[k] - sums[k, d])
                lg_term4 = sum(map(fn1, enumerate(yi)))
                return np.exp(lg_term1 - lg_term2 - lg_term3 + lg_term4)

            dist = np.array(map(fn, xrange(K)))
            dist /= dist.sum()

            # reassign
            ci = discrete_sample(dist)
            assignments[i] = ci
            sums[ci] += yi
            cnts[ci] += 1
        history[t] = assignments

    return history

def pr_joint(C, Y, K, alpha, beta, gamma):
    """
    compute P(C,Y) = P(C)P(Y|C)
    """
    N, D = Y.shape
    nks = np.bincount(C, minlength=K)
    assert nks.shape[0] == K
    assert C.shape[0] == N

    # log P(C)
    gammaln = sp.special.gammaln
    betaln = sp.special.betaln
    term1 = gammaln(alpha) - gammaln(N + alpha) - K*gammaln(alpha/K)
    term2 = sum(gammaln(nk + alpha/K) for nk in nks)
    lg_pC = term1 + term2

    # log P(Y|C)
    term1 = K*D*betaln(beta, gamma)
    term2 = D*sum(gammaln(nk + beta + gamma) for nk in nks)
    sums = np.zeros((K, D))
    for yi, ci in zip(Y, C):
        sums[ci] += yi
    def fn1(nk, sum_yid):
        assert nk >= sum_yid
        return gammaln(sum_yid + beta) + gammaln(nk - sum_yid + gamma)
    term3 = sum(sum(fn1(nk, yid) for yid in row) for nk, row in zip(nks, sums))
    lg_pYgC = -term1 - term2 + term3

    #print 'term1:', term1
    #print 'term2:', term2
    #print 'term3:', term3

    #print 'lg_pC:', lg_pC
    #print 'lg_pYgC:', lg_pYgC
    #print 'exp(lg_pC + lg_pYgC):', np.exp(lg_pC + lg_pYgC)

    return np.exp(lg_pC + lg_pYgC)

def brute_force_posterior(Y, K, alpha, beta, gamma):
    """
    compute p(C|Y) by brute force enuemration and normalization
    """
    N, _ = Y.shape

    # enumerate K^N cluster assignments
    pis = np.array([pr_joint(np.array(C), Y, K, alpha, beta, gamma) for C in it.product(range(K), repeat=N)])
    pis /= pis.sum()

    return pis

def histify(history, K):
    """
    create a histogram of the history samples

    only works for small K, N
    """
    _, N = history.shape
    # generate an ID for each K^N
    idmap = { C : i for i, C in enumerate(it.product(range(K), repeat=N)) }
    hist = np.zeros(K**N, dtype=np.float)
    for h in history:
        hist[idmap[tuple(h)]] += 1.0
    return hist

def main():

    # generate some data according to the beta-bernoulli model

    alpha, beta, gamma = 1., 1., 2.
    K = 2
    D = 3
    N = 5
    skip = 100
    smoothing = 1e-5
    n_samples = 50000

    pis = np.random.dirichlet( alpha/K * np.ones(K) )
    print pis
    cis = np.array([discrete_sample(pis) for _ in xrange(N)])
    aks = np.random.beta(beta, gamma, size=(K, D))

    def bernoulli(p):
        return 1. if np.random.random() <= p else 0.

    Y = np.zeros((N, D))
    for i in xrange(N):
        Y[i] = np.array([bernoulli(aks[cis[i], d]) for d in xrange(D)])

    actual = brute_force_posterior(Y, K, alpha, beta, gamma)
    history = gibbs_beta_bernoulli(Y, K, alpha, beta, gamma, n_samples)

    def fn(i):
        hist = histify(history[:i:skip], K) + smoothing
        hist /= hist.sum()
        return kl(actual, hist)

    iters = range(0, n_samples, skip)[1:]
    kls = map(fn, iters)
    plt.plot(iters, kls)
    plt.xlabel('Iterations')
    plt.ylabel('KL-divergence')
    plt.show()

if __name__ == '__main__':
    main()
