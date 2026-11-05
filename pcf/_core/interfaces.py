"""PCF computation core; imported by the public PCF API."""
from __future__ import annotations
import numpy as np

def from_labels(labels, K):
    z = np.asarray(labels)
    if z.ndim != 2 or not np.issubdtype(z.dtype, np.integer) or np.any(z < 0) or np.any(z >= K):
        raise ValueError('labels must be integer NxM within [0,K)')
    return np.eye(K, dtype=float)[z]

def from_rankings(order):
    order = np.asarray(order)
    if order.ndim != 3 or not np.issubdtype(order.dtype, np.integer):
        raise ValueError('orders must be NxMxK integers')
    K = order.shape[2]
    if not np.all(np.sort(order, axis=2) == np.arange(K)[None, None, :]):
        raise ValueError('Every row must be a full category permutation')
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(K)[None, None, :], axis=2)
    val = np.exp(-ranks / max(K - 1, 1))
    return val / val.sum(axis=2, keepdims=True)
