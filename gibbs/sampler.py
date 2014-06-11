#!/usr/bin/env python
import numpy as np

def discrete_sample(pmf):
    # XXX: does numpy have something to do this?
    coin = np.random.random()
    cdf = np.cumsum(pmf)
    a = np.where(coin >= cdf)[0]
    if not a.shape[0]:
        return 0
    return min(a[-1]+1, pmf.shape[0]-1)

def gibbs_beta_bernoulli(Y, K, alpha, beta, gamma, niters):
    N, D = Y.shape
    alpha, beta, gamma = map(float, [alpha, beta, gamma])

    # start with random assignment
    assignments = np.random.randint(0, K, size=N)

    # initialize the sufficient statistics (cluster sums) accordingly
    sums = np.zeros((K, D))
    cnts = np.zeros(K)
    for data, idx in zip(Y, assignments):
        sums[idx] += data
        cnts[idx] += 1

    for _ in xrange(niters):
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

    return assignments

def main():

    # generate some data according to the beta-bernoulli model

    alpha, beta, gamma = 0.5, 1., 2.
    K = 3
    D = 5
    N = 100

    pis = np.random.dirichlet( alpha/K * np.ones(K), size=1 )
    cis = np.array([discrete_sample(pis) for _ in xrange(N)])
    aks = np.random.beta(beta, gamma, size=(K, D))

    def bernoulli(p):
        return 1. if np.random.random() <= p else 0.

    Y = np.zeros((N, D))
    for i in xrange(N):
        Y[i] = np.array([bernoulli(aks[cis[i], d]) for d in xrange(D)])

    assignments = gibbs_beta_bernoulli(Y, K, alpha, beta, gamma, 100)

if __name__ == '__main__':
    main()
