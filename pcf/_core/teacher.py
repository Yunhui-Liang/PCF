"""PCF computation core; imported by the public PCF API."""
from __future__ import annotations
import numpy as np
from .calibration import V9Truth
from .dirichlet import dirichlet_em, predict_dirichlet
from .evidence import _clean
from .interfaces import from_labels, from_rankings

class V11Truth:

    def __init__(self, variant='F', K=None, sds_weight=0.1, init_blend=0.25, sds_outer=50, **v9_kwargs):
        if variant not in ('D', 'R', 'F'):
            raise ValueError('variant in D,R,F')
        if not 0 <= sds_weight <= 1:
            raise ValueError('sds_weight must be [0,1]')
        if not 0 <= init_blend <= 1:
            raise ValueError('init_blend must be [0,1]')
        self.variant = variant
        self.K = K
        self.sds_weight = sds_weight
        self.init_blend = init_blend
        self.sds_outer = sds_outer
        self.v9_kwargs = v9_kwargs

    def _adapt(self, X):
        if self.variant == 'D':
            if self.K is None:
                raise ValueError('K required for D')
            return from_labels(X, self.K)
        if self.variant == 'R':
            return from_rankings(X)
        return _clean(X)

    def fit(self, X):
        P = self._adapt(X)
        self.K = P.shape[2]
        self.M = P.shape[1]
        self.v9 = V9Truth(variant=self.variant, K=self.K, **self.v9_kwargs).fit(X)
        self.with_sds = self.variant == 'F' and self.K >= 4 and (self.M >= 3) and (self.sds_weight > 0) and (not np.all((P < 1e-08) | (P > 1 - 1e-08)))
        if self.with_sds:
            qd = self.v9.orig.predict(P)['D']
            qf = self.v9.predict(X)
            q0 = (1 - self.init_blend) * qd + self.init_blend * qf
            _, mdl = dirichlet_em(P, initial_q=q0, outer=self.sds_outer, inner=4, polyak=0.025, return_model=True)
            self.alpha = mdl['alpha']
            self.prior = mdl['prior']
        return self

    def predict(self, X):
        P = self._adapt(X)
        if P.shape[1:] != (self.M, self.K):
            raise ValueError('source/class shape mismatch')
        q = self.v9.predict(X)
        if not self.with_sds:
            return q
        s = predict_dirichlet(P, self.alpha, self.prior)
        out = (1 - self.sds_weight) * q + self.sds_weight * s
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-20)
