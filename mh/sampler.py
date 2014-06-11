#!/usr/bin/env python

import math
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pylab as plt

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

def sample(lg_target_ppdf, lg_proposal_cond_ppdf, proposal_cond_draw, niters, x0, rng):
    """
    lg_target_ppdf        - (proportional) density
    lg_proposal_cond_ppdf - conditioned on arg0, (proportional) density
    proposal_cond_draw    - conditioned on arg0, draw a value
    """

    xt = x0
    values = []
    naccepts = 0
    while niters:
        # sample xprop ~ Q(.|xt)
        xprop = proposal_cond_draw(xt, rng)

        lg_alpha_1 = lg_target_ppdf(xprop) - lg_target_ppdf(xt)
        lg_alpha_2 = lg_proposal_cond_ppdf(xprop, xt) - lg_proposal_cond_ppdf(xt, xprop)
        lg_alpha   = lg_alpha_1 + lg_alpha_2

        if lg_alpha >= 0.0 or rng.random() <= np.exp(lg_alpha):
            naccepts += 1
            xt = xprop

        niters -= 1
        values.append(xt)

    return (np.array(values), naccepts)

def hist(data, bins):
    H, _ = np.histogram(data, bins=bins, density=False)
    return H

def hist2d(data, xbins, ybins):
    H, _, _ = np.histogram2d(data[:,0], data[:,1], bins=[xbins, ybins])
    return H

def kl(a, b, dA):
    return np.sum([p*np.log(p/q)*dA for p, q in zip(a, b)])

def mixture():
    mu1 = -1.
    mu2 = 1.
    sigma2 = 0.1
    sigma2_prop = 0.001
    smoothing = 0.001

    def normpdf(x, mu, sigma2):
        return 1./np.sqrt(2.*math.pi*sigma2) * np.exp(-1./(2.*sigma2)*((x-mu)**2))

    def lgnormpdf(x, mu, sigma2):
        return -0.5*np.log(2.*math.pi*sigma2) - 1./(2.*sigma2)*((x-mu)**2)

    lg_target_ppdf = lambda x: np.log(0.5*normpdf(x, mu1, sigma2) + 0.5*normpdf(x, mu2, sigma2))
    lg_proposal_cond_ppdf = lambda xt, x: lgnormpdf(x, xt, sigma2_prop)
    proposal_cond_draw = lambda xt, rng: rng.normal(loc=xt, scale=np.sqrt(sigma2_prop))

    n_samples = 10000

    values, naccepts = sample(lg_target_ppdf, lg_proposal_cond_ppdf, proposal_cond_draw, n_samples * 2, 0.0, np.random)
    values = values[n_samples:]

    actual_samples = [np.random.normal(loc=mu1 if np.random.random() < 0.5 else mu2, scale=np.sqrt(sigma2)) for _ in xrange(n_samples)]

    bins = np.linspace(-2, 2, 1000)
    mh_hist = hist(values, bins) + smoothing
    actual_hist = hist(actual_samples, bins) + smoothing

    mh_hist /= mh_hist.sum()
    actual_hist /= actual_hist.sum()

    print kl(mh_hist, actual_hist, (bins[1]-bins[0]))

    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, mh_hist, align='center', width=width, color='r')
    plt.savefig('mixture_mh.pdf')
    plt.close()

    plt.bar(center, actual_hist, align='center', width=width, color='g')
    plt.savefig('mixture_actual.pdf')

def main():
    mean = np.ones(2)
    Q    = random_orthonormal_matrix(2)
    diag_elems = [1.0, 0.1]
    cov  = np.dot(np.dot(Q, np.diag(diag_elems)), Q.T)
    covinv = np.dot(np.dot(Q, np.diag([1./x for x in diag_elems])), Q.T)
    smoothing = 1e-5
    n_samples = 50000
    skip = 100
    bins = np.linspace(0, 2, 100)
    def lg_target_ppdf(x):
        diff = x - mean
        return -0.5 * np.dot(diff, np.dot(covinv, diff))

    sigma2_props = [0.01, 0.1, 1.0, 10.0]
    all_samples = []
    for sigma2_prop in sigma2_props:
        propcov = sigma2_prop*np.eye(2)
        def lg_proposal_cond_ppdf(xt, x):
            diff = x - xt
            return -0.5 / sigma2_prop * np.dot(diff, diff)
        proposal_cond_draw = lambda xt, rng: rng.multivariate_normal(mean=xt, cov=propcov)
        samples, _ = sample(lg_target_ppdf, lg_proposal_cond_ppdf, proposal_cond_draw, n_samples, np.zeros(2), np.random)
        all_samples.append(samples)

    # sample from actual distribution
    actual_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=n_samples)
    actual_hist = (hist2d(actual_samples, bins, bins) + smoothing).flatten()
    actual_hist /= actual_hist.sum()

    all_bufs = []
    for j in xrange(len(sigma2_props)):
        buf = []
        for i in range(0, n_samples, skip)[10:]:
            effective_samples = all_samples[j][:i:skip]
            mh_hist = (hist2d(effective_samples, bins, bins) + smoothing).flatten()
            mh_hist /= mh_hist.sum()
            score = kl(mh_hist, actual_hist, (bins[1]-bins[0])**2)
            buf.append((i+1, score))
        all_bufs.append(buf)

    for buf in all_bufs:
        plt.plot([x[0] for x in buf], [x[1] for x in buf])
    plt.xlabel('Iterations')
    plt.ylabel('KL-divergence')
    plt.show()

    #plt.plot( values[:,0], values[:,1], 'rx' )
    #plt.plot( actual_samples[:,0], actual_samples[:,1], 'gx' )
    #plt.show()

if __name__ == '__main__':
    main()
    #mixture()
