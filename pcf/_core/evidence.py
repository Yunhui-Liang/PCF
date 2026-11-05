"""PCF computation core; imported by the public PCF API."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .decision import ds

_EPS = 1e-12

def softmax(a):
    return np.exp(a - logsumexp(a, axis=1, keepdims=True))

def _clean(P):
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 3 or P.shape[2] < 2 or np.any(P < 0) or (not np.isfinite(P).all()):
        raise ValueError('Need nonnegative finite NxMxK')
    P = np.maximum(P, 1e-12)
    return P / P.sum(2, keepdims=True)

def _ranks(P):
    order = np.argsort(-P, axis=2, kind='stable')
    rank = np.argsort(order, axis=2)
    return (order, rank)

def ds_predict(Z, C, prior, leave=None):
    N, M = Z.shape
    out = np.broadcast_to(np.log(prior + _EPS), (N, len(prior))).copy()
    for m in range(M):
        if m != leave:
            out += np.log(C[m, :, Z[:, m]] + _EPS)
    return softmax(out)

class UnifiedTruth:

    def __init__(self, topn=12, shrink_rank=30.0, shrink_target=50.0, rank_strength=2.0, fuzzy_strength=1.0, fuzzy_shrink=30.0, num_bins=4, refine=True, hard_active_mask=None):
        self.topn = topn
        self.shrink_rank = shrink_rank
        self.shrink_target = shrink_target
        self.rank_strength = rank_strength
        self.fuzzy_strength = fuzzy_strength
        self.fuzzy_shrink = fuzzy_shrink
        self.num_bins = num_bins
        self.refine = refine
        self.hard_active_mask = hard_active_mask

    def _fit_rank(self, order, rank, qloo, Z, K):
        N, M, _ = order.shape
        L = min(self.topn, K)
        self.rank_lut = np.zeros((M, L, K, K), dtype=np.float32)
        self.truthrank_lut = np.zeros((M, K, K), dtype=np.float32)
        for m in range(M):
            qm = qloo[m]
            for r in range(1, L):
                obs = order[:, m, r]
                counts = np.zeros((K, K))
                for k in range(K):
                    counts[k] = np.bincount(obs, weights=qm[:, k], minlength=K)
                marg = np.bincount(obs, minlength=K).astype(float) + 1
                marg /= marg.sum()
                cond = (counts + self.shrink_rank * marg) / (counts.sum(1, keepdims=True) + self.shrink_rank)
                self.rank_lut[m, r] = ((np.log(cond + _EPS) - np.log(marg[None, :] + _EPS)) / np.sqrt(r)).astype('float32')
            for k in range(K):
                ix = rank[:, m, k]
                ok = ix > 0
                if not ok.any():
                    continue
                pos = np.bincount(ix[ok], weights=qm[ok, k], minlength=K).astype(float)
                neg = np.bincount(ix[ok], weights=1 - qm[ok, k], minlength=K).astype(float)
                null = np.bincount(ix[ok], minlength=K).astype(float) + 1
                null = null / (null.sum() + _EPS)
                sh = self.shrink_target
                pp = (pos + sh * null) / (pos.sum() + sh)
                pn = (neg + sh * null) / (neg.sum() + sh)
                self.truthrank_lut[m, k] = (np.log(pp + _EPS) - np.log(pn + _EPS)).astype('float32')
        self.truthrank_lut[:, :, 0] = 0

    def rank_score(self, P):
        order, rank = _ranks(P)
        N, M, K = P.shape
        L = min(self.topn, K)
        rs = np.zeros((N, M, K), dtype=np.float32)
        for m in range(M):
            for r in range(1, L):
                rs[:, m, :] += self.rank_lut[m, r, :, order[:, m, r]]
            rs[:, m, :] += self.truthrank_lut[m, np.arange(K)[None, :], rank[:, m, :]]
        return rs

    def _features(self, P):
        N, M, K = P.shape
        order, rank = _ranks(P)
        sortedP = np.take_along_axis(P, order, axis=2)
        gap = np.log(sortedP[:, :, 0] + _EPS) - np.log(sortedP[:, :, 1] + _EPS)
        top = np.log(sortedP[:, :, 0] + _EPS)
        raw = top[:, :, None] - np.log(P + _EPS)
        raw = np.where(rank == 0, gap[:, :, None], raw)
        bucket = np.where(rank == 0, 0, np.where(rank == 1, 1, np.where(rank == 2, 2, np.where(rank <= 5, 3, 4))))
        return (raw, bucket, rank)

    def _fit_fuzzy(self, P, qloo):
        raw, bucket, _ = self._features(P)
        N, M, K = P.shape
        B = self.num_bins
        G = 5
        self.fuzzy_edges = np.zeros((M, G, B - 1))
        self.fuzzy_lut = np.zeros((M, G, B), dtype=np.float32)
        for m in range(M):
            w = qloo[m]
            for g in range(G):
                mask = bucket[:, m, :] == g
                vals = raw[:, m, :][mask]
                if len(vals) < B * 3:
                    continue
                edges = np.quantile(vals, np.arange(1, B) / B)
                self.fuzzy_edges[m, g] = edges
                inds = np.searchsorted(edges, vals, side='right')
                pos = np.bincount(inds, weights=w[mask], minlength=B).astype(float)
                neg = np.bincount(inds, weights=(1 - w)[mask], minlength=B).astype(float)
                null = (np.bincount(inds, minlength=B) + 1).astype(float)
                null /= null.sum()
                alpha = self.fuzzy_shrink
                pp = (pos + alpha * null) / (pos.sum() + alpha)
                pn = (neg + alpha * null) / (neg.sum() + alpha)
                self.fuzzy_lut[m, g] = np.log(pp + _EPS) - np.log(pn + _EPS)

    def fuzzy_score(self, P):
        raw, bucket, _ = self._features(P)
        N, M, K = P.shape
        out = np.zeros((N, M, K), dtype=np.float32)
        for m in range(M):
            for g in range(5):
                mask = bucket[:, m, :] == g
                if mask.any():
                    ids = np.searchsorted(self.fuzzy_edges[m, g], raw[:, m, :][mask], side='right')
                    out[:, m, :][mask] = self.fuzzy_lut[m, g, ids]
        return out

    def fit(self, P):
        P = _clean(P)
        N, M, K = P.shape
        self.K = K
        self.M = M
        Z = P.argmax(2)
        if self.hard_active_mask is None:
            active = np.ones(M, dtype=bool)
        else:
            active = np.asarray(self.hard_active_mask, dtype=bool)
            if active.shape != (M,) or not active.any():
                raise ValueError('bad hard source mask')
        self.hard_active_mask_used = active.copy()
        qd, Cactive, prior, info = ds(Z[:, active], K)
        C = np.ones((M, K, K), dtype=float)
        C[active] = Cactive
        self.C = C
        self.prior = prior
        self.train_info = info
        self.hard_fallback = M < 3 or K < 3 or np.all((P < 1e-08) | (P > 1 - 1e-08))
        if self.hard_fallback:
            return self
        order, rank = _ranks(P)
        qloo = [ds_predict(Z, C, prior, leave=m) for m in range(M)]
        self._fit_rank(order, rank, qloo, Z, K)
        if self.refine:
            rsc = self.rank_score(P)
            teachers = []
            base = np.log(ds_predict(Z, C, prior) + _EPS)
            for m in range(M):
                zscore = np.log(ds_predict(Z, C, prior, leave=m) + _EPS)
                teachers.append(softmax(zscore + self.rank_strength * (rsc.sum(1) - rsc[:, m, :])))
            self._fit_rank(order, rank, teachers, Z, K)
            qloo = teachers
        self._fit_fuzzy(P, qloo)
        return self

    def predict(self, P):
        P = _clean(P)
        if P.shape[1] != self.M or P.shape[2] != self.K:
            raise ValueError('Source/class shape mismatch')
        Z = P.argmax(2)
        D = ds_predict(Z, self.C, self.prior)
        if self.hard_fallback:
            return {'D': D, 'R': D.copy(), 'F': D.copy()}
        rs = self.rank_score(P).sum(1)
        R = softmax(np.log(D + _EPS) + self.rank_strength * rs)
        fs = self.fuzzy_score(P).sum(1)
        F = softmax(np.log(R + _EPS) + self.fuzzy_strength * fs)
        return {'D': D, 'R': R, 'F': F}
