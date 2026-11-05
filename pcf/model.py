"""Public interface for Progressive Candidate Fusion."""
from __future__ import annotations
import numpy as np
from ._core.translation import V17Truth

def normalize_probabilities(P):
    p=np.asarray(P,dtype=np.float64)
    if p.ndim!=3 or min(p.shape[:2])<1 or p.shape[2]<2:
        raise ValueError('Expected a nonempty N x M x K tensor with K >= 2.')
    if not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Probabilities must be finite and nonnegative.')
    if np.any(p.sum(axis=2)<=0):
        raise ValueError('Every source prediction must have positive total mass.')
    p=np.maximum(p,1e-12)
    return p/p.sum(axis=2,keepdims=True)

class PCF:
    """Unsupervised fusion of aligned source probabilities.

    fit(P) receives only probabilities, with shape (samples, sources, classes).
    predict_proba(P) returns (samples, classes). Class and source order must
    be consistent between fitting and inference.
    """
    def __init__(self, *, n_jobs=1, random_state=2026):
        self.n_jobs=int(n_jobs)
        self.random_state=int(random_state)
        if self.n_jobs<1:raise ValueError('n_jobs must be positive.')
        self._model=None

    def fit(self,P):
        import torch
        p=normalize_probabilities(P)
        torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        self._model=V17Truth(K=p.shape[2],n_jobs=self.n_jobs).fit(p)
        self.n_sources_=p.shape[1]
        self.n_classes_=p.shape[2]
        return self

    def predict_proba(self,P):
        if self._model is None:raise RuntimeError('Call fit(P) before prediction.')
        p=normalize_probabilities(P)
        if p.shape[1:]!=(self.n_sources_,self.n_classes_):
            raise ValueError('Source/class dimensions differ from the fitted model.')
        return np.asarray(self._model.predict(p))

    def predict(self,P):
        return self.predict_proba(P).argmax(axis=1)

    def fit_predict_proba(self,P):
        return self.fit(P).predict_proba(P)
