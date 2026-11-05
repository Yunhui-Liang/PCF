"""PCF computation core; imported by the public PCF API."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp

def vote(Z, K):
    N, M = Z.shape
    v = np.zeros((N, K))
    idx = np.arange(N)
    for m in range(M):
        v[idx, Z[:, m]] += 1
    return v

def ds(Z, K, niter=100, prior_diag=1.0, smooth=0.1, init='majority', tol=1e-06, verbose=False):
    N, M = Z.shape
    counts = vote(Z, K)
    if init == 'majority':
        q = (counts + 0.01) / (M + K * 0.01)
    elif isinstance(init, int):
        q = np.full((N, K), 0.05 / (K - 1))
        q[np.arange(N), Z[:, init]] = 0.95
    elif init == 'random':
        rng = np.random.default_rng(0)
        q = rng.dirichlet(np.ones(K), size=N)
    else:
        raise ValueError(init)
    prior = np.maximum(q.mean(0), 1e-08)
    C = np.empty((M, K, K))
    trace = []
    for t in range(niter):
        for m in range(M):
            counts_m = np.zeros((K, K))
            for k in range(K):
                counts_m[k] = np.bincount(Z[:, m], weights=q[:, k], minlength=K)
            counts_m += smooth
            counts_m[np.arange(K), np.arange(K)] += prior_diag
            C[m] = counts_m / counts_m.sum(axis=1, keepdims=True)
        prior = np.maximum(q.mean(0), 1e-12)
        prior /= prior.sum()
        logp = np.tile(np.log(prior), (N, 1))
        for m in range(M):
            logp += np.log(np.maximum(C[m, :, Z[:, m]].T, 1e-300)) if False else np.log(C[m, :, Z[:, m]])
        lognorm = logsumexp(logp, axis=1)
        q_new = np.exp(logp - lognorm[:, None])
        delta = np.mean(np.abs(q_new - q))
        q = q_new
        trace.append(lognorm.sum())
        if delta < tol:
            break
    return (q, C, prior, dict(iter=t + 1, ll=trace[-1], delta=delta))
