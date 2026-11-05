"""PCF computation core; imported by the public PCF API."""
from __future__ import annotations
import numpy as np
import warnings
from .teacher import V11Truth
from .evidence import _clean

def source_features(Pm, feature_mode='full'):
    a = np.asarray(Pm, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError('Expected N by K')
    order = np.argsort(-a, axis=1, kind='stable')
    rank = np.argsort(order, axis=1).astype(np.float32)
    if feature_mode == 'rank':
        return rank
    if feature_mode == 'hard':
        return np.eye(a.shape[1], dtype=np.float32)[a.argmax(1)]
    if feature_mode == 'full':
        return np.concatenate([np.log(np.maximum(a, 1e-09)), rank, a], axis=1).astype(np.float32)
    raise ValueError('Unknown feature_mode')

class V17Truth:

    def __init__(self, K=None, student_weight=0.5, n_estimators=45, num_leaves=15, feature_mode='full', n_jobs=4, min_classes_for_students=4):
        self.K = K
        self.student_weight = float(student_weight)
        self.n_estimators = int(n_estimators)
        self.num_leaves = int(num_leaves)
        self.feature_mode = feature_mode
        self.n_jobs = n_jobs
        self.min_classes_for_students = min_classes_for_students
        if not 0 <= self.student_weight <= 1:
            raise ValueError('student_weight must be in [0,1]')

    def fit(self, P):
        from lightgbm import LGBMClassifier
        P = _clean(P).astype(np.float32)
        self.M = P.shape[1]
        self.K = P.shape[2]
        self.teacher = V11Truth(variant='F', K=self.K).fit(P)
        self.trained = self.K >= self.min_classes_for_students and self.M >= 3 and (self.student_weight > 0)
        self.students = []
        if not self.trained:
            return self
        pseudo = self.teacher.predict(P).argmax(1)
        for m in range(self.M):
            classes = np.unique(pseudo)
            if classes.size == 1:
                self.students.append(('constant', int(classes[0])))
                continue
            model = LGBMClassifier(n_estimators=self.n_estimators, num_leaves=self.num_leaves, min_child_samples=50, max_bin=63, learning_rate=0.06, reg_lambda=3.0, verbose=-1, n_jobs=self.n_jobs, random_state=123)
            model.fit(source_features(P[:, m, :], self.feature_mode), pseudo)
            self.students.append(('model', model))
        return self

    def predict_components(self, P):
        P = _clean(P).astype(np.float32)
        if P.shape[1:] != (self.M, self.K):
            raise ValueError('Different number of oracles/classes')
        teacher = np.asarray(self.teacher.predict(P), dtype=np.float32)
        if not self.trained:
            return dict(teacher=teacher, translated=teacher.copy(), final=teacher.copy())
        experts = []
        for m, (typ, obj) in enumerate(self.students):
            pred = np.full((len(P), self.K), 1e-06, dtype=np.float32)
            if typ == 'constant':
                pred[:, obj] = 1.0
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='X does not have valid feature names, but LGBMClassifier was fitted with feature names')
                    x = obj.predict_proba(source_features(P[:, m, :], self.feature_mode))
                pred[:, obj.classes_.astype(int)] = x
            pred /= pred.sum(1, keepdims=True)
            experts.append(pred)
        experts = np.stack(experts, axis=0)
        h = -np.sum(experts * np.log(np.maximum(experts, 1e-08)), axis=2)
        weights = 1 / (h + 0.1)
        translated = np.sum(weights[:, :, None] * experts, axis=0) / np.maximum(weights.sum(axis=0)[:, None], 1e-12)
        final = (1 - self.student_weight) * teacher + self.student_weight * translated
        return dict(teacher=teacher, translated=translated, final=final, per_source=experts)

    def predict(self, P):
        return self.predict_components(P)['final']
