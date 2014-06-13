import numpy as np
import scipy as sp
import scipy.misc
import scipy.io
import math

def discrete_sample(pmf):
    coin = np.random.random()
    acc = 0.0
    n, = pmf.shape
    for i in xrange(n):
        acc0 = acc + pmf[i]
        if coin <= acc0:
            return i
        acc = acc0
    return n-1

def gibbs_beta_bernoulli(Y, K, alpha, beta, gamma, niters):
    N, D = Y.shape
    alpha, beta, gamma = map(float, [alpha, beta, gamma])

    # start with random assignment
    assignments = np.random.randint(0, K, size=N)

    # initialize the sufficient statistics (cluster sums) accordingly
    sums = np.zeros((D, K), dtype=np.int64)
    cnts = np.zeros(K, dtype=np.int64)
    for yi, ci in zip(Y, assignments):
        sums[:,ci] += yi
        cnts[ci] += 1
    #assert cnts.sum() == N

    history = np.zeros((niters, N), dtype=np.int64)

    # precomputations
    nplog = np.log
    npexp = np.exp
    nparray = np.array
    logsumexp = sp.misc.logsumexp
    lg_denom = nplog(N - 1 + alpha)
    alpha_over_K = alpha/K
    beta_plus_gamma = beta + gamma

    for t in xrange(niters):
        for i, (yi, ci) in enumerate(zip(Y, assignments)):
            # remove from SS
            sums[:,ci] -= yi
            cnts[ci] -= 1

            lg_term1 = nplog(cnts + alpha_over_K) - lg_denom - D*nplog(beta_plus_gamma + cnts)
            lg_term2 = nplog(beta + sums)
            lg_term3 = nplog(gamma + cnts - sums)

            lg_dist = lg_term1 + (lg_term2*yi[:,np.newaxis] + lg_term3*(1-yi[:,np.newaxis])).sum(axis=0)
            lg_dist -= logsumexp(lg_dist) # normalize

            # reassign
            ci = discrete_sample(npexp(lg_dist))
            assignments[i] = ci
            sums[:,ci] += yi
            cnts[ci] += 1

        history[t] = assignments

    return history

def gibbs_beta_bernoulli_old(Y, K, alpha, beta, gamma, niters):
    N, D = Y.shape
    alpha, beta, gamma = map(float, [alpha, beta, gamma])

    # start with random assignment
    assignments = np.random.randint(0, K, size=N)

    # initialize the sufficient statistics (cluster sums) accordingly
    sums = np.zeros((K, D), dtype=np.int64)
    cnts = np.zeros(K, dtype=np.int64)
    for yi, ci in zip(Y, assignments):
        sums[ci] += yi
        cnts[ci] += 1
    #assert cnts.sum() == N

    history = np.zeros((niters, N), dtype=np.int64)

    # precomputations
    nplog = np.log
    npexp = np.exp
    nparray = np.array
    logsumexp = sp.misc.logsumexp
    lg_term2 = nplog(N - 1 + alpha)
    alpha_over_K = alpha/K
    beta_plus_gamma = beta + gamma

    for t in xrange(niters):
        for i, (yi, ci) in enumerate(zip(Y, assignments)):
            # remove from SS
            #assert cnts[ci] >= 1
            sums[ci] -= yi
            cnts[ci] -= 1

            # build log P(c_i=k | c_{\i}, Y)
            def fn(k):
                lg_term1 = nplog(cnts[k] + alpha_over_K)
                lg_term3 = D*nplog(beta_plus_gamma + cnts[k])
                def fn1(tup):
                    d, yid = tup
                    #assert yid == 0 or yid == 1
                    #assert d >= 0 and d < D
                    #assert cnts[k] >= sums[k, d]
                    if yid:
                        return nplog(beta + sums[k, d])
                    else:
                        return nplog(gamma + cnts[k] - sums[k, d])
                lg_term4 = sum(map(fn1, enumerate(yi)))
                return lg_term1 - lg_term2 - lg_term3 + lg_term4

            lg_dist = nparray(map(fn, xrange(K)))
            lg_dist -= logsumexp(lg_dist) # normalize
            dist = npexp(lg_dist)
            #assert almost_eq(dist.sum(), 1.0)

            # reassign
            ci = discrete_sample(dist)
            assignments[i] = ci
            sums[ci] += yi
            cnts[ci] += 1

        #assert cnts.sum() == N
        history[t] = assignments

    return history

### Some test code for discrete_sample(), feel free to ignore ###
def almost_eq(a, b, tol=1e-5):
    return math.fabs(a - b) <= tol

def pmfs_almost_eq(a, b, tol):
    assert a.shape == b.shape
    return (np.fabs(a - b) <= tol).all()

def test_discrete_sample0(fn, pmf, tol=1e-3):
    samples = np.array(
        np.bincount(
            [fn(pmf) for _ in xrange(500000)],
            minlength=pmf.shape[0]),
        dtype=np.float)
    samples /= samples.sum()
    if not pmfs_almost_eq(pmf, samples, tol):
        print 'max distance:', np.fabs(pmf - samples).max()
        assert False

def test_discrete_sample_impl(fn):
    pmf = np.arange(1, 6, dtype=np.float)
    pmf /= pmf.sum()
    test_discrete_sample0(fn, pmf, 1e-2)
    test_discrete_sample0(fn, np.array([0.95, 0.05])) # binary case 1
    test_discrete_sample0(fn, np.array([0.01, 0.99])) # binary case 2

def test():
    test_discrete_sample_impl( discrete_sample )

if __name__ == '__main__':
    usps_digits = sp.io.loadmat('data/usps_resampled/usps_resampled.mat')
    usps_class_2 = usps_digits['train_labels'][2,:] == 1
    usps_Y0 = usps_digits['train_patterns'][:,usps_class_2]
    usps_Y = np.zeros(usps_Y0.shape)
    usps_Y[usps_Y0>0.1] = 1.0
    usps_Y = np.array(usps_Y.T, dtype=np.int64)

    #priors
    usps_alpha, usps_beta, usps_gamma = 50., .5, .5
    usps_K = 40
    usps_N, usps_D = usps_Y.shape
    usps_niters = 100
    print usps_alpha, usps_beta, usps_gamma
    print usps_K, usps_N, usps_D

    import cProfile
    cProfile.run('gibbs_beta_bernoulli(usps_Y, usps_K, usps_alpha, usps_beta, usps_gamma, 10)')
