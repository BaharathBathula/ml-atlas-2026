from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import pickle

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

RANDOM_STATE = 42


@dataclass
class BenchmarkResult:
    leaderboard: pd.DataFrame
    best_model_name: str
    best_model: Any
    X_test: pd.DataFrame
    y_test: pd.Series
    best_predictions: np.ndarray
    feature_names: List[str]


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)


def _classification_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }


def _regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=220, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def _classification_metrics(model, X_test, y_test, pred):
    row = {
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    }

    try:
        if len(pd.Series(y_test).unique()) == 2 and hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            # binary labels may be strings, so map via model classes
            positive = model.classes_[1] if hasattr(model, "classes_") else sorted(pd.Series(y_test).unique())[-1]
            y_binary = (pd.Series(y_test).values == positive).astype(int)
            row["roc_auc"] = roc_auc_score(y_binary, probs)
        else:
            row["roc_auc"] = np.nan
    except Exception:
        row["roc_auc"] = np.nan

    return row


def _regression_metrics(y_test, pred):
    return {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "r2": r2_score(y_test, pred),
    }


def benchmark_models(
    df: pd.DataFrame,
    target: str,
    task: str,
    test_size: float = 0.25,
    max_models: int | None = None,
) -> BenchmarkResult:
    work = df.dropna(subset=[target]).copy()
    if len(work) < 30:
        raise ValueError("At least 30 rows with a non-missing target are required.")

    y = work[target]
    X = work.drop(columns=[target])

    # Remove columns that are entirely missing.
    all_missing = [c for c in X.columns if X[c].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after preprocessing.")

    stratify = None
    if task == "classification":
        counts = y.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    model_map = _classification_models() if task == "classification" else _regression_models()
    if max_models:
        model_map = dict(list(model_map.items())[:max_models])

    rows = []
    fitted = {}
    predictions = {}
    feature_names = []

    for name, estimator in model_map.items():
        preprocessor = _make_preprocessor(X)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        try:
            pipeline.fit(X_train, y_train)
            pred = pipeline.predict(X_test)

            if task == "classification":
                metrics = _classification_metrics(pipeline, X_test, y_test, pred)
            else:
                metrics = _regression_metrics(y_test, pred)

            row = {"model": name, "status": "success", **metrics}
            rows.append(row)
            fitted[name] = pipeline
            predictions[name] = pred
        except Exception as exc:
            rows.append({"model": name, "status": f"failed: {type(exc).__name__}"})

    leaderboard = pd.DataFrame(rows)
    successes = leaderboard[leaderboard["status"] == "success"].copy()
    if successes.empty:
        raise RuntimeError("All benchmark models failed. Review target type and feature columns.")

    if task == "classification":
        successes = successes.sort_values(["f1_weighted", "accuracy"], ascending=False)
    else:
        successes = successes.sort_values(["r2", "rmse"], ascending=[False, True])

    # Put successful models first, then failed rows.
    failed = leaderboard[leaderboard["status"] != "success"]
    leaderboard = pd.concat([successes, failed], ignore_index=True)

    best_name = str(successes.iloc[0]["model"])
    best_model = fitted[best_name]
    best_pred = predictions[best_name]

    try:
        feature_names = best_model.named_steps["preprocessor"].get_feature_names_out().tolist()
    except Exception:
        feature_names = []

    return BenchmarkResult(
        leaderboard=leaderboard,
        best_model_name=best_name,
        best_model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_predictions=best_pred,
        feature_names=feature_names,
    )


def feature_importance_table(result: BenchmarkResult, top_n: int = 20) -> pd.DataFrame:
    estimator = result.best_model.named_steps["model"]
    names = result.feature_names

    values = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    if values is None or not names or len(values) != len(names):
        return pd.DataFrame(columns=["feature", "importance"])

    out = pd.DataFrame({"feature": names, "importance": values})
    return out.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def serialize_model(model) -> bytes:
    return pickle.dumps(model)
