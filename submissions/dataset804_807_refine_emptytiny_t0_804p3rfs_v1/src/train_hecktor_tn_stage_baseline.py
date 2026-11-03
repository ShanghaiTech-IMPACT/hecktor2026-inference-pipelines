from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Compatibility shim required by the saved HECKTOR T/N joblib models."""

    def __init__(self, estimator: Any, balanced_sample_weight: bool = False):
        self.estimator = estimator
        self.balanced_sample_weight = balanced_sample_weight

    def fit(self, x: Any, y: Any) -> "LabelEncodedClassifier":
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.estimator_ = clone(self.estimator)
        fit_kwargs: dict[str, Any] = {}
        if self.balanced_sample_weight:
            fit_kwargs["sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_encoded)
        try:
            self.estimator_.fit(x, y_encoded, **fit_kwargs)
        except TypeError:
            self.estimator_.fit(x, y_encoded)
        return self

    def predict(self, x: Any) -> np.ndarray:
        y_encoded = self.estimator_.predict(x)
        return self.label_encoder_.inverse_transform(np.asarray(y_encoded, dtype=int).ravel())

    def predict_proba(self, x: Any) -> np.ndarray:
        return self.estimator_.predict_proba(x)
