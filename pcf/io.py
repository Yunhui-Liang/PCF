"""Data loading, canonical paper splits and post-prediction evaluation."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .model import normalize_probabilities

DATASETS = ('cifar19', 'flower17')

def load_probabilities(path):
    path=Path(path)
    if path.suffix.lower()=='.mat':
        # Do not read gt while loading model inputs.
        d=loadmat(path,variable_names=['P_cell'])
        if 'P_cell' not in d:raise ValueError('MAT input must contain P_cell.')
        p=np.stack([np.asarray(x,dtype=np.float64) for x in d['P_cell'].flat],axis=1)
    elif path.suffix.lower()=='.npy':
        p=np.load(path,allow_pickle=False)
    elif path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as d:
            key='P' if 'P' in d else 'probabilities'
            if key not in d:raise ValueError('NPZ input needs P or probabilities.')
            p=d[key]
    else:raise ValueError('Use .mat, .npy or .npz input.')
    return normalize_probabilities(p)

def load_labels(path,n,k):
    path=Path(path)
    if path.suffix.lower()=='.mat':
        d=loadmat(path,variable_names=['gt'])
        if 'gt' not in d:raise ValueError('Evaluation needs gt in the MAT file.')
        y=d['gt']
    elif path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as d:
            key='y' if 'y' in d else 'gt'
            if key not in d:raise ValueError('Evaluation needs y or gt in the NPZ file.')
            y=d[key]
    else:raise ValueError('Evaluation labels must be stored in a MAT or NPZ input.')
    y=np.asarray(y).ravel()
    if y.shape!=(n,) or not np.isfinite(y).all() or not np.all(y==np.floor(y)):
        raise ValueError('Labels must be N integer class IDs.')
    y=y.astype(np.int64)
    if y.min()==1 and y.max()==k:y=y-1
    if y.min()<0 or y.max()>=k:raise ValueError('Label IDs must be in [0,K).')
    return y

def split_indices(n,seed):
    perm=np.random.default_rng(int(seed)).permutation(n)
    count=int(round(n*.2))
    if count<1 or count>=n:raise ValueError('Need at least one fit and one test sample.')
    return np.sort(perm[count:]),np.sort(perm[:count])

def classification_metrics(y,q):
    from sklearn.metrics import f1_score
    q=np.asarray(q,dtype=np.float64)
    pred=q.argmax(axis=1);k=q.shape[1]
    return {'ACC':float(100*np.mean(pred==y)),
            'MacroF1':float(100*f1_score(y,pred,labels=np.arange(k),average='macro',zero_division=0)),
            'Brier':float(np.mean(np.sum((q-np.eye(k)[y])**2,axis=1)))}
