"""PCF computation core; imported by the public PCF API."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .evidence import UnifiedTruth, _clean
from .interfaces import from_labels, from_rankings

EPS = 1e-12

def posterior_temperature(q, t=8.0):
    logits = np.log(np.maximum(q, EPS)) / t
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

def source_bias(P):
    logp = np.log(_clean(P))
    b = logp.mean(axis=0)
    return b - b.mean(axis=0, keepdims=True)

def rectify_source_class(P, b, eta):
    logits = np.log(_clean(P)) - eta * b[None, :, :]
    return np.exp(logits - logsumexp(logits, axis=2, keepdims=True))

class V9Truth:

    def __init__(self, variant='F', K=None, eta=1.5, pool_weight=0.45, posterior_temperature_value=8.0, collapse_screen=False, **kwargs):
        self.variant = variant
        self.K = K
        self.eta = eta
        self.pool_weight = pool_weight
        self.posterior_temperature_value = posterior_temperature_value
        self.collapse_screen = collapse_screen
        self.kwargs = {'fuzzy_strength': 2.0, 'fuzzy_shrink': 10.0, **kwargs}

    def _adapt(self, X):
        if self.variant == 'D':
            if self.K is None:
                raise ValueError('K needed for D, to preserve unseen classes')
            return from_labels(X, self.K)
        if self.variant == 'R':
            return from_rankings(X)
        if self.variant == 'F':
            return _clean(X)
        raise ValueError(self.variant)

    def fit(self, X):
        P = self._adapt(X)
        if self.collapse_screen:
            Z = P.argmax(axis=2)
            N, M = Z.shape
            freq = np.array([np.bincount(Z[:, m], minlength=P.shape[2]).max() for m in range(M)])
            active = N - freq > np.sqrt(N)
            if active.sum() < min(3, M):
                active[:] = True
        else:
            active = np.ones(P.shape[1], dtype=bool)
        self.hard_active_mask = active
        self.orig = UnifiedTruth(hard_active_mask=active, **self.kwargs).fit(P)
        self.K = P.shape[2]
        self.M = P.shape[1]
        self.use_second = self.variant != 'D' and self.M >= 3 and (self.K >= 3)
        if self.use_second:
            self.b = source_bias(P)
            self.corrected = UnifiedTruth(hard_active_mask=active, **self.kwargs).fit(rectify_source_class(P, self.b, self.eta))
        return self

    def predict(self, X):
        P = self._adapt(X)
        if P.shape[1:] != (self.M, self.K):
            raise ValueError('shape mismatch')
        original = self.orig.predict(P)
        which = self.variant
        first = original[which]
        if not self.use_second:
            return first
        corrected = self.corrected.predict(rectify_source_class(P, self.b, self.eta))[which]
        p0 = posterior_temperature(first, self.posterior_temperature_value)
        p1 = posterior_temperature(corrected, self.posterior_temperature_value)
        return (1 - self.pool_weight) * p0 + self.pool_weight * p1
