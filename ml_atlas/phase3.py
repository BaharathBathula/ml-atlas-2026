from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import io
import json
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_validate,
)

from .automl import BenchmarkResult, _classification_models, _regression_models, _make_preprocessor


RANDOM_STATE = 42


@dataclass
class CrossValidationResult:
    model_name: str
    task: str
    folds: int
    metrics: pd.DataFrame
    summary: Dict[str, float]


@dataclass
class TuningResult:
    model_name: str
    best_estimator: Any
    best_params: Dict[str, Any]
    best_score: float
    scoring: str
    cv_results: pd.DataFrame
    elapsed_seconds: float


@dataclass
class ThresholdResult:
    threshold_table: pd.DataFrame
    recommended_threshold: float
    metric_name: str
    metric_value: float


@dataclass
class ExplainabilityResult:
    importance: pd.DataFrame
    method: str


def _get_X_y(df: pd.DataFrame, target: str):
    work = df.dropna(subset=[target]).copy()
    X = work.drop(columns=[target])
    y = work[target]
    all_missing = [c for c in X.columns if X[c].isna().all()]
    if all_missing:
        X = X.drop(columns=all_missing)
    return X, y


def _build_pipeline_for_model(model_name: str, task: str, X: pd.DataFrame):
    from sklearn.pipeline import Pipeline

    models = _classification_models() if task == "classification" else _regression_models()
    if model_name not in models:
        raise ValueError(f"Unsupported model for Phase 3: {model_name}")

    return Pipeline([
        ("preprocessor", _make_preprocessor(X)),
        ("model", clone(models[model_name])),
    ])


def run_cross_validation(
    df: pd.DataFrame,
    target: str,
    task: str,
    model_name: str,
    folds: int = 5,
) -> CrossValidationResult:
    X, y = _get_X_y(df, target)
    if len(X) < folds * 5:
        raise ValueError("Dataset is too small for the selected number of folds.")

    pipeline = _build_pipeline_for_model(model_name, task, X)

    if task == "classification":
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "accuracy": "accuracy",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "f1_weighted": "f1_weighted",
        }
    else:
        cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    rows = {}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        vals = np.asarray(values, dtype=float)
        if metric in {"mae", "rmse"}:
            vals = -vals
        rows[metric] = vals

    metrics = pd.DataFrame(rows)
    metrics.insert(0, "fold", np.arange(1, len(metrics) + 1))

    summary = {}
    for col in metrics.columns:
        if col == "fold":
            continue
        summary[f"{col}_mean"] = float(metrics[col].mean())
        summary[f"{col}_std"] = float(metrics[col].std(ddof=1)) if len(metrics) > 1 else 0.0

    return CrossValidationResult(
        model_name=model_name,
        task=task,
        folds=folds,
        metrics=metrics,
        summary=summary,
    )


def _tuning_space(model_name: str, task: str) -> Dict[str, List[Any]]:
    if task == "classification":
        spaces = {
            "Logistic Regression": {
                "model__C": [0.1, 1.0, 10.0],
            },
            "Decision Tree": {
                "model__max_depth": [4, 8, 12, None],
                "model__min_samples_split": [2, 5, 10],
            },
            "Random Forest": {
                "model__n_estimators": [120, 220],
                "model__max_depth": [None, 8, 14],
                "model__min_samples_split": [2, 5],
            },
            "Extra Trees": {
                "model__n_estimators": [120, 220],
                "model__max_depth": [None, 8, 14],
            },
            "Gradient Boosting": {
                "model__n_estimators": [80, 140],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
            "k-NN": {
                "model__n_neighbors": [3, 5, 9, 15],
                "model__weights": ["uniform", "distance"],
            },
            "SVM": {
                "model__C": [0.5, 1.0, 3.0],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf"],
            },
        }
    else:
        spaces = {
            "Linear Regression": {},
            "Ridge Regression": {
                "model__alpha": [0.1, 1.0, 10.0, 30.0],
            },
            "Lasso Regression": {
                "model__alpha": [0.001, 0.01, 0.1, 1.0],
            },
            "Decision Tree": {
                "model__max_depth": [4, 8, 12, None],
                "model__min_samples_split": [2, 5, 10],
            },
            "Random Forest": {
                "model__n_estimators": [120, 220],
                "model__max_depth": [None, 8, 14],
                "model__min_samples_split": [2, 5],
            },
            "Extra Trees": {
                "model__n_estimators": [120, 220],
                "model__max_depth": [None, 8, 14],
            },
            "Gradient Boosting": {
                "model__n_estimators": [80, 140],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        }
    if model_name not in spaces:
        raise ValueError(f"No tuning space defined for {model_name}.")
    return spaces[model_name]


def tune_model(
    df: pd.DataFrame,
    target: str,
    task: str,
    model_name: str,
    folds: int = 5,
) -> TuningResult:
    X, y = _get_X_y(df, target)
    pipeline = _build_pipeline_for_model(model_name, task, X)
    params = _tuning_space(model_name, task)

    if task == "classification":
        scoring = "f1_weighted"
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    else:
        scoring = "neg_root_mean_squared_error"
        cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    start = time.time()
    search = GridSearchCV(
        pipeline,
        params,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X, y)
    elapsed = time.time() - start

    results = pd.DataFrame(search.cv_results_)
    keep = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    results = results[keep].sort_values("rank_test_score").reset_index(drop=True)

    best_score = float(search.best_score_)
    if task == "regression":
        best_score = -best_score

    return TuningResult(
        model_name=model_name,
        best_estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_score=best_score,
        scoring="F1 weighted" if task == "classification" else "RMSE",
        cv_results=results,
        elapsed_seconds=float(elapsed),
    )


def binary_curve_data(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, pd.DataFrame]:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Selected model does not expose predict_proba.")
    classes = list(model.classes_)
    if len(classes) != 2:
        raise ValueError("ROC/PR curve analysis requires binary classification.")

    positive = classes[1]
    y_true = (pd.Series(y_test).values == positive).astype(int)
    prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, roc_thr = roc_curve(y_true, prob)
    precision, recall, pr_thr = precision_recall_curve(y_true, prob)

    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr})
    pr_threshold = np.append(pr_thr, np.nan)
    pr_df = pd.DataFrame({"recall": recall, "precision": precision, "threshold": pr_threshold})

    return {
        "roc": roc_df,
        "pr": pr_df,
        "roc_auc": pd.DataFrame({"value": [roc_auc_score(y_true, prob)]}),
        "average_precision": pd.DataFrame({"value": [average_precision_score(y_true, prob)]}),
    }


def threshold_analysis(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    optimize_for: str = "f1",
) -> ThresholdResult:
    if not hasattr(model, "predict_proba"):
        raise ValueError("Threshold analysis requires predict_proba.")
    classes = list(model.classes_)
    if len(classes) != 2:
        raise ValueError("Threshold analysis requires binary classification.")

    negative, positive = classes[0], classes[1]
    prob = model.predict_proba(X_test)[:, 1]
    y_arr = np.asarray(y_test)

    rows = []
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    for t in thresholds:
        pred = np.where(prob >= t, positive, negative)
        rows.append({
            "threshold": float(t),
            "precision_positive": precision_score(y_arr, pred, pos_label=positive, zero_division=0),
            "recall_positive": recall_score(y_arr, pred, pos_label=positive, zero_division=0),
            "f1_positive": f1_score(y_arr, pred, pos_label=positive, zero_division=0),
            "accuracy": accuracy_score(y_arr, pred),
        })

    table = pd.DataFrame(rows)
    metric_map = {
        "f1": "f1_positive",
        "recall": "recall_positive",
        "precision": "precision_positive",
        "accuracy": "accuracy",
    }
    metric_col = metric_map.get(optimize_for, "f1_positive")
    best = table.sort_values(metric_col, ascending=False).iloc[0]

    return ThresholdResult(
        threshold_table=table,
        recommended_threshold=float(best["threshold"]),
        metric_name=metric_col,
        metric_value=float(best[metric_col]),
    )


def permutation_explain(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task: str,
    top_n: int = 20,
) -> ExplainabilityResult:
    scoring = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=7,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=-1,
    )
    out = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    out = out.sort_values("importance_mean", ascending=False).head(top_n).reset_index(drop=True)
    return ExplainabilityResult(importance=out, method="Permutation importance")


def residual_diagnostics(y_true, y_pred) -> Dict[str, pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred

    residual_df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
        "residual": residual,
        "absolute_residual": np.abs(residual),
    })

    summary = pd.DataFrame({
        "metric": ["MAE", "RMSE", "R2", "Mean residual", "Residual std"],
        "value": [
            mean_absolute_error(y_true, y_pred),
            mean_squared_error(y_true, y_pred) ** 0.5,
            r2_score(y_true, y_pred),
            float(np.mean(residual)),
            float(np.std(residual)),
        ],
    })
    return {"residuals": residual_df, "summary": summary}


def classification_diagnostics(y_true, y_pred) -> Dict[str, pd.DataFrame]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm)
    return {"report": report_df, "confusion_matrix": cm_df}


def build_experiment_report(
    dataset_name: str,
    target: str,
    task: str,
    benchmark: BenchmarkResult,
    cv_result: Optional[CrossValidationResult] = None,
    tuning_result: Optional[TuningResult] = None,
    explainability: Optional[ExplainabilityResult] = None,
    threshold_result: Optional[ThresholdResult] = None,
) -> str:
    lines = []
    lines.append("# ML Atlas 2026 — Experiment Report")
    lines.append("")
    lines.append(f"- Dataset: {dataset_name}")
    lines.append(f"- Target: `{target}`")
    lines.append(f"- Task: {task}")
    lines.append(f"- Best benchmark model: **{benchmark.best_model_name}**")
    lines.append("")
    lines.append("## Benchmark leaderboard")
    lines.append("")
    lines.append(benchmark.leaderboard.to_markdown(index=False))
    lines.append("")

    if cv_result is not None:
        lines.append("## Cross-validation")
        lines.append("")
        lines.append(f"- Model: {cv_result.model_name}")
        lines.append(f"- Folds: {cv_result.folds}")
        for k, v in cv_result.summary.items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")

    if tuning_result is not None:
        lines.append("## Hyperparameter tuning")
        lines.append("")
        lines.append(f"- Model: {tuning_result.model_name}")
        lines.append(f"- Objective: {tuning_result.scoring}")
        lines.append(f"- Best score: {tuning_result.best_score:.4f}")
        lines.append(f"- Best parameters: `{json.dumps(tuning_result.best_params, default=str)}`")
        lines.append("")

    if threshold_result is not None:
        lines.append("## Threshold analysis")
        lines.append("")
        lines.append(f"- Recommended threshold: {threshold_result.recommended_threshold:.2f}")
        lines.append(f"- Optimized metric: {threshold_result.metric_name}")
        lines.append(f"- Metric value: {threshold_result.metric_value:.4f}")
        lines.append("")

    if explainability is not None:
        lines.append("## Explainability")
        lines.append("")
        lines.append(f"- Method: {explainability.method}")
        lines.append("")
        lines.append(explainability.importance.to_markdown(index=False))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "This report is generated by an educational AutoML-style application. "
        "Model quality must be validated against domain-specific objectives, leakage risks, fairness, drift, and deployment constraints before production use."
    )
    return "\n".join(lines)


def serialize_any(obj) -> bytes:
    return pickle.dumps(obj)
