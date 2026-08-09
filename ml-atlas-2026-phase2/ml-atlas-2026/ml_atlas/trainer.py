from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from sklearn.model_selection import train_test_split

from .registry import ALGORITHMS, RANDOM_STATE

@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, float]
    y_true: Any = None
    y_pred: Any = None
    embedding: Any = None
    labels: Any = None

def _safe_round(v):
    return float(round(float(v), 4))

def train_supervised(algorithm_name, X, y, task):
    spec = ALGORITHMS[algorithm_name]
    model = spec.builder()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE,
        stratify=y if task == "classification" else None
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if task == "classification":
        metrics = {
            "accuracy": _safe_round(accuracy_score(y_test, pred)),
            "precision_weighted": _safe_round(precision_score(y_test, pred, average="weighted", zero_division=0)),
            "recall_weighted": _safe_round(recall_score(y_test, pred, average="weighted", zero_division=0)),
            "f1_weighted": _safe_round(f1_score(y_test, pred, average="weighted", zero_division=0)),
        }
    else:
        rmse = mean_squared_error(y_test, pred) ** 0.5
        metrics = {
            "mae": _safe_round(mean_absolute_error(y_test, pred)),
            "rmse": _safe_round(rmse),
            "r2": _safe_round(r2_score(y_test, pred)),
        }

    return TrainResult(model=model, metrics=metrics, y_true=y_test, y_pred=pred)

def run_clustering(algorithm_name, X):
    model = ALGORITHMS[algorithm_name].builder()
    labels = model.fit_predict(X)
    unique = set(labels)
    non_noise = [x for x in unique if x != -1]
    metrics = {
        "clusters_found": float(len(non_noise)),
        "noise_points": float(np.sum(labels == -1)),
    }
    if len(non_noise) >= 2:
        mask = labels != -1
        if mask.sum() > len(non_noise):
            metrics["silhouette"] = _safe_round(silhouette_score(X[mask], labels[mask]))
    return TrainResult(model=model, metrics=metrics, labels=labels)

def run_reduction(algorithm_name, X):
    model = ALGORITHMS[algorithm_name].builder()
    embedding = model.fit_transform(X)
    return TrainResult(model=model, metrics={"components": float(embedding.shape[1])}, embedding=embedding)
