from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class ModelService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self):
        path = Path(self.model_path)
        if not path.exists():
            self.model = None
            return
        with path.open("rb") as f:
            self.model = pickle.load(f)

    def info(self) -> Dict[str, Any]:
        if not self.is_loaded:
            return {
                "loaded": False,
                "model_path": self.model_path,
                "estimator": None,
                "classes": None,
            }

        estimator = self.model
        if hasattr(estimator, "named_steps"):
            inner = estimator.named_steps.get("model")
        else:
            inner = estimator

        classes = None
        if hasattr(self.model, "classes_"):
            classes = [str(x) for x in self.model.classes_]
        elif inner is not None and hasattr(inner, "classes_"):
            classes = [str(x) for x in inner.classes_]

        return {
            "loaded": True,
            "model_path": self.model_path,
            "estimator": type(inner).__name__ if inner is not None else type(self.model).__name__,
            "classes": classes,
        }

    def predict(self, df: pd.DataFrame) -> Tuple[List[Any], List[Dict[str, float]] | None]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        pred = self.model.predict(df)
        predictions = [x.item() if hasattr(x, "item") else x for x in pred]

        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(df)
            classes = list(self.model.classes_)
            probabilities = [
                {str(cls): float(prob) for cls, prob in zip(classes, row)}
                for row in probs
            ]

        return predictions, probabilities
