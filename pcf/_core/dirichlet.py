"""PCF computation core; imported by the public PCF API."""
from __future__ import annotations
import numpy as np
from scipy.special import gammaln, logsumexp

def _softmax(x):
    return np.exp(x - logsumexp(x, axis=1, keepdims=True))

def dirichlet_em(P, initial_q=None, outer=35, inner=4, step=0.002, polyak=0.025, eps=1e-07, seed=0, verbose=False, return_model=False, regularization=0.0):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    P = np.asarray(P, dtype=np.float32)
    N, M, K = P.shape
    P = np.maximum(P, eps)
    P /= P.sum(axis=2, keepdims=True)
    if initial_q is None:
        initial_q = np.mean(P, axis=1)
    q = np.asarray(initial_q, dtype=float).copy()
    q /= q.sum(axis=1, keepdims=True)
    mean = np.einsum('nk,nml->mkl', q, P, optimize=True) / (q.sum(0)[None, :, None] + 1e-12)
    mean = np.maximum(mean, 0.0001)
    mean /= mean.sum(2, keepdims=True)
    alpha_init = mean * (K + 1) + 0.5
    raw = torch.nn.Parameter(torch.tensor(np.log(np.expm1(alpha_init)), dtype=torch.float32))
    optim = torch.optim.AdamW([raw], lr=step, weight_decay=0.0001)
    logP = np.log(P)
    lp = np.transpose(logP, (1, 0, 2))
    logPtorch = torch.tensor(lp, dtype=torch.float32)
    prior = np.maximum(q.mean(0), 1e-12)
    prior /= prior.sum()
    n_inner = int(inner)
    last = None
    anchor = torch.tensor(np.log(alpha_init), dtype=torch.float32)
    for it in range(outer):
        qt = torch.tensor(q.T, dtype=torch.float32)
        stats = torch.einsum('kn,mnl->mkl', qt, logPtorch)
        counts = torch.tensor(q.sum(0), dtype=torch.float32)
        for t in range(n_inner):
            optim.zero_grad()
            al = torch.nn.functional.softplus(raw) + 0.0001
            norms = torch.lgamma(al.sum(2)) - torch.lgamma(al).sum(2)
            ll = torch.sum((al - 1) * stats) + torch.sum(norms * counts[None, :])
            loss = -ll / max(N, 1)
            if regularization > 0:
                loss = loss + regularization * torch.mean((torch.log(al) - anchor) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([raw], 10.0)
            optim.step()
        aa = (torch.nn.functional.softplus(raw) + 0.0001).detach().cpu().numpy()
        gg = gammaln(aa.sum(2)) - gammaln(aa).sum(2)
        logits = np.broadcast_to(np.log(prior + 1e-12), (N, K)).copy()
        for m in range(M):
            logits += logP[:, m, :] @ (aa[m] - 1).T + gg[m][None, :]
        q_new = _softmax(logits)
        q = (1 - polyak) * q + polyak * q_new
        prior = np.maximum(q.mean(0), 1e-12)
        prior /= prior.sum()
        last = float(np.mean(np.abs(q_new - q)))
        if verbose and it % 10 == 0:
            print('sds', it, last, flush=True)
    info = {'iterations': outer, 'polyak': polyak, 'alpha_min': float(aa.min()), 'delta': last}
    if return_model:
        info.update({'alpha': aa, 'prior': prior})
    return (q, info)

def predict_dirichlet(P, alpha, prior, eps=1e-07):
    P = np.asarray(P, dtype=np.float32)
    P = np.maximum(P, eps)
    P /= P.sum(2, keepdims=True)
    alpha = np.asarray(alpha, dtype=float)
    prior = np.asarray(prior, dtype=float)
    from scipy.special import gammaln, logsumexp
    ll = np.log(np.maximum(prior, 1e-12))[None, :] + np.zeros((P.shape[0], alpha.shape[1]))
    gg = gammaln(alpha.sum(2)) - gammaln(alpha).sum(2)
    for m in range(P.shape[1]):
        ll += np.log(P[:, m, :]) @ (alpha[m] - 1).T + gg[m][None, :]
    return np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
