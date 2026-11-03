from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class StageProbabilityEnsemble:
    """Average class probabilities from several sklearn/CatBoost stage classifiers."""

    def __init__(self, members: list[Any], labels: list[str]) -> None:
        self.members = list(members)
        self.labels = [str(label) for label in labels]
        self.classes_ = np.asarray(self.labels, dtype=object)

    @staticmethod
    def _align_proba(member: Any, proba: np.ndarray, labels: list[str]) -> np.ndarray:
        aligned = np.zeros((proba.shape[0], len(labels)), dtype=np.float64)
        classes = np.asarray(getattr(member, "classes_", []), dtype=str)
        if classes.size == 0 and hasattr(member, "named_steps") and "classifier" in member.named_steps:
            classes = np.asarray(getattr(member.named_steps["classifier"], "classes_", []), dtype=str)
        class_to_idx = {str(cls): idx for idx, cls in enumerate(classes)}
        for out_idx, label in enumerate(labels):
            src_idx = class_to_idx.get(str(label))
            if src_idx is not None and src_idx < proba.shape[1]:
                aligned[:, out_idx] = proba[:, src_idx]
        return aligned

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if not self.members:
            raise RuntimeError("StageProbabilityEnsemble has no members")
        probas = [
            self._align_proba(member, np.asarray(member.predict_proba(x), dtype=np.float64), self.labels)
            for member in self.members
        ]
        mean_proba = np.mean(np.stack(probas, axis=0), axis=0)
        row_sums = mean_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 1e-12, 1.0, row_sums)
        return mean_proba / row_sums

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]


class RfsFusionRiskModel:
    """Monotonic relapse/time fusion risk model for HECKTOR RFS submission."""

    def __init__(
        self,
        classifier: Any,
        regressor: Any,
        classifier_train_mean: float,
        classifier_train_std: float,
        regressor_train_mean: float,
        regressor_train_std: float,
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.classifier_train_mean = float(classifier_train_mean)
        self.classifier_train_std = max(float(classifier_train_std), 1e-8)
        self.regressor_train_mean = float(regressor_train_mean)
        self.regressor_train_std = max(float(regressor_train_std), 1e-8)

    def raw_risk(self, x: pd.DataFrame) -> np.ndarray:
        relapse_prob = np.asarray(self.classifier.predict_proba(x)[:, 1], dtype=np.float64)
        pred_log_time = np.asarray(self.regressor.predict(x), dtype=np.float64)
        relapse_z = (relapse_prob - self.classifier_train_mean) / self.classifier_train_std
        time_z = (pred_log_time - self.regressor_train_mean) / self.regressor_train_std
        return relapse_z - time_z

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.raw_risk(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        risk = self.raw_risk(x)
        probability_like = 1.0 / (1.0 + np.exp(-np.clip(risk, -50.0, 50.0)))
        return np.column_stack([1.0 - probability_like, probability_like])


class RfsP3FusionRiskModel:
    """5-seed P3 fusion: relapse ExtraTrees + ExtraSurvivalTrees risk."""

    def __init__(
        self,
        members: list[dict[str, Any]],
        feature_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self.members = members
        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)
        self.numeric_columns = [col for col in self.feature_columns if col not in self.categorical_columns]

    @staticmethod
    def _norm_missing(value: Any) -> str:
        if value is None:
            return "__MISSING__"
        if isinstance(value, float) and np.isnan(value):
            return "__MISSING__"
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}:
            return "__MISSING__"
        return text

    def _prepare_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame([{col: row.get(col, np.nan) for col in self.feature_columns} for row in x.to_dict("records")])
        for col in self.categorical_columns:
            if col in out.columns:
                out[col] = out[col].map(self._norm_missing)
        for col in self.numeric_columns:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out[self.feature_columns]

    @staticmethod
    def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
        scale = max(float(std), 1e-8)
        return (np.asarray(values, dtype=np.float64) - float(mean)) / scale

    def raw_risk(self, x: pd.DataFrame) -> np.ndarray:
        frame = self._prepare_frame(x)
        risks: list[np.ndarray] = []
        for member in self.members:
            clf = member["classifier"]
            surv = member["survival_model"]
            relapse_prob = np.asarray(clf.predict_proba(frame)[:, 1], dtype=np.float64)
            survival_risk = np.asarray(surv.predict(frame), dtype=np.float64)
            relapse_z = self._zscore(relapse_prob, member["classifier_train_mean"], member["classifier_train_std"])
            survival_z = self._zscore(survival_risk, member["survival_train_mean"], member["survival_train_std"])
            risks.append(relapse_z + survival_z)
        return np.mean(np.stack(risks, axis=0), axis=0)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.raw_risk(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        risk = self.raw_risk(x)
        probability_like = 1.0 / (1.0 + np.exp(-np.clip(risk, -50.0, 50.0)))
        return np.column_stack([1.0 - probability_like, probability_like])


class RfsP3PureFusionRiskModel:
    """Dependency-light P3 fusion using exported preprocessing and tree arrays."""

    def __init__(self, members: list[dict[str, Any]], feature_columns: list[str], categorical_columns: list[str]) -> None:
        self.members = members
        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)

    @staticmethod
    def _norm_missing(value: Any) -> str:
        if value is None:
            return "__MISSING__"
        if isinstance(value, float) and np.isnan(value):
            return "__MISSING__"
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}:
            return "__MISSING__"
        return text

    def _input_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame([{col: row.get(col, np.nan) for col in self.feature_columns} for row in x.to_dict("records")])

    @classmethod
    def _transform(cls, frame: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
        parts: list[np.ndarray] = []
        numeric_cols = list(spec["numeric_columns"])
        if numeric_cols:
            numeric = frame[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            medians = np.asarray(spec["numeric_medians"], dtype=np.float64)
            means = np.asarray(spec["numeric_means"], dtype=np.float64)
            scales = np.asarray(spec["numeric_scales"], dtype=np.float64)
            missing = ~np.isfinite(numeric)
            if missing.any():
                numeric[missing] = np.take(medians, np.where(missing)[1])
            scales = np.where(np.abs(scales) < 1e-8, 1.0, scales)
            parts.append((numeric - means) / scales)

        categorical_cols = list(spec["categorical_columns"])
        categories = spec["categories"]
        for col, cats in zip(categorical_cols, categories):
            values = frame[col].map(cls._norm_missing).to_numpy(dtype=object)
            cats_arr = np.asarray(cats, dtype=object)
            encoded = np.zeros((len(frame), len(cats_arr)), dtype=np.float64)
            lookup = {str(value): idx for idx, value in enumerate(cats_arr)}
            for row_idx, value in enumerate(values):
                cat_idx = lookup.get(str(value))
                if cat_idx is not None:
                    encoded[row_idx, cat_idx] = 1.0
            parts.append(encoded)

        if not parts:
            return np.zeros((len(frame), 0), dtype=np.float32)
        return np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _apply_tree(x: np.ndarray, tree: dict[str, Any]) -> np.ndarray:
        children_left = tree["children_left"]
        children_right = tree["children_right"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        leaves = np.zeros(x.shape[0], dtype=np.int64)
        for row_idx in range(x.shape[0]):
            node = 0
            while children_left[node] != -1:
                if x[row_idx, feature[node]] <= threshold[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            leaves[row_idx] = node
        return leaves

    @classmethod
    def _predict_classifier_proba(cls, x: np.ndarray, trees: list[dict[str, Any]]) -> np.ndarray:
        proba = np.zeros((x.shape[0], 2), dtype=np.float64)
        for tree in trees:
            leaves = cls._apply_tree(x, tree)
            values = tree["values"][leaves]
            proba += values
        proba /= max(len(trees), 1)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 1e-12, 1.0, row_sums)
        return proba / row_sums

    @classmethod
    def _predict_survival_risk(cls, x: np.ndarray, trees: list[dict[str, Any]]) -> np.ndarray:
        risk = np.zeros(x.shape[0], dtype=np.float64)
        for tree in trees:
            leaves = cls._apply_tree(x, tree)
            values = tree["chf_values"][leaves]
            event_mask = tree["is_event_time"]
            risk += values[:, event_mask].sum(axis=1)
        return risk / max(len(trees), 1)

    @staticmethod
    def _zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
        scale = max(float(std), 1e-8)
        return (np.asarray(values, dtype=np.float64) - float(mean)) / scale

    def raw_risk(self, x: pd.DataFrame) -> np.ndarray:
        frame = self._input_frame(x)
        risks: list[np.ndarray] = []
        for member in self.members:
            clf_x = self._transform(frame, member["classifier_preprocessor"])
            surv_x = self._transform(frame, member["survival_preprocessor"])
            relapse_prob = self._predict_classifier_proba(clf_x, member["classifier_trees"])[:, 1]
            survival_risk = self._predict_survival_risk(surv_x, member["survival_trees"])
            relapse_z = self._zscore(relapse_prob, member["classifier_train_mean"], member["classifier_train_std"])
            survival_z = self._zscore(survival_risk, member["survival_train_mean"], member["survival_train_std"])
            risks.append(relapse_z + survival_z)
        return np.mean(np.stack(risks, axis=0), axis=0)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.raw_risk(x)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        risk = self.raw_risk(x)
        probability_like = 1.0 / (1.0 + np.exp(-np.clip(risk, -50.0, 50.0)))
        return np.column_stack([1.0 - probability_like, probability_like])
