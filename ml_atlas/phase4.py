from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.model_selection import learning_curve


RANDOM_STATE = 42


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp_utc: str
    dataset_name: str
    target: str
    task: str
    model_name: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    notes: str = ""


@dataclass
class RegistryEntry:
    version: int
    model_name: str
    stage: str
    experiment_id: str
    registered_at_utc: str
    notes: str = ""


def _utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def make_experiment_id(
    dataset_name: str,
    target: str,
    task: str,
    model_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    payload = json.dumps(
        {
            "dataset_name": dataset_name,
            "target": target,
            "task": task,
            "model_name": model_name,
            "params": params or {},
            "time_ns": time.time_ns(),
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"exp-{digest}"


def make_experiment_record(
    dataset_name: str,
    target: str,
    task: str,
    model_name: str,
    metrics: Dict[str, float],
    params: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> ExperimentRecord:
    clean_metrics = {}
    for k, v in metrics.items():
        try:
            clean_metrics[k] = float(v)
        except Exception:
            continue

    return ExperimentRecord(
        experiment_id=make_experiment_id(dataset_name, target, task, model_name, params),
        timestamp_utc=_utc_now(),
        dataset_name=dataset_name,
        target=target,
        task=task,
        model_name=model_name,
        metrics=clean_metrics,
        params=params or {},
        notes=notes,
    )


def experiments_to_frame(records: List[Dict[str, Any] | ExperimentRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        if isinstance(record, ExperimentRecord):
            record = asdict(record)
        row = {
            "experiment_id": record.get("experiment_id"),
            "timestamp_utc": record.get("timestamp_utc"),
            "dataset_name": record.get("dataset_name"),
            "target": record.get("target"),
            "task": record.get("task"),
            "model_name": record.get("model_name"),
            "notes": record.get("notes", ""),
        }
        for k, v in (record.get("metrics") or {}).items():
            row[f"metric_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def register_model(
    registry: List[Dict[str, Any] | RegistryEntry],
    model_name: str,
    experiment_id: str,
    stage: str,
    notes: str = "",
) -> RegistryEntry:
    versions = []
    for item in registry:
        if isinstance(item, RegistryEntry):
            versions.append(item.version)
        else:
            versions.append(int(item.get("version", 0)))
    version = max(versions, default=0) + 1
    return RegistryEntry(
        version=version,
        model_name=model_name,
        stage=stage,
        experiment_id=experiment_id,
        registered_at_utc=_utc_now(),
        notes=notes,
    )


def registry_to_frame(registry: List[Dict[str, Any] | RegistryEntry]) -> pd.DataFrame:
    rows = []
    for item in registry:
        rows.append(asdict(item) if isinstance(item, RegistryEntry) else item)
    return pd.DataFrame(rows)


def population_stability_index(
    baseline: pd.Series,
    current: pd.Series,
    bins: int = 10,
) -> float:
    base = pd.Series(baseline).dropna()
    curr = pd.Series(current).dropna()
    if base.empty or curr.empty:
        return float("nan")

    if pd.api.types.is_numeric_dtype(base) and pd.api.types.is_numeric_dtype(curr):
        quantiles = np.unique(np.nanquantile(base.astype(float), np.linspace(0, 1, bins + 1)))
        if len(quantiles) < 3:
            return 0.0
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        base_counts = pd.cut(base.astype(float), bins=quantiles, include_lowest=True).value_counts(sort=False)
        curr_counts = pd.cut(curr.astype(float), bins=quantiles, include_lowest=True).value_counts(sort=False)
        base_pct = base_counts / max(base_counts.sum(), 1)
        curr_pct = curr_counts / max(curr_counts.sum(), 1)
    else:
        categories = sorted(set(base.astype(str).unique()) | set(curr.astype(str).unique()))
        base_pct = base.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)
        curr_pct = curr.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0)

    eps = 1e-6
    base_pct = np.clip(np.asarray(base_pct, dtype=float), eps, 1)
    curr_pct = np.clip(np.asarray(curr_pct, dtype=float), eps, 1)
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    max_features: int = 50,
) -> pd.DataFrame:
    common = [c for c in baseline_df.columns if c in current_df.columns][:max_features]
    rows = []
    for col in common:
        psi = population_stability_index(baseline_df[col], current_df[col])
        if pd.isna(psi):
            status = "unavailable"
        elif psi < 0.10:
            status = "stable"
        elif psi < 0.25:
            status = "moderate_shift"
        else:
            status = "significant_shift"
        rows.append({
            "feature": col,
            "psi": psi,
            "status": status,
            "baseline_missing_pct": float(baseline_df[col].isna().mean() * 100),
            "current_missing_pct": float(current_df[col].isna().mean() * 100),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def validate_schema(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    cols = sorted(set(baseline_df.columns) | set(current_df.columns))
    rows = []
    for col in cols:
        in_base = col in baseline_df.columns
        in_curr = col in current_df.columns
        base_dtype = str(baseline_df[col].dtype) if in_base else None
        curr_dtype = str(current_df[col].dtype) if in_curr else None

        if not in_base:
            status = "new_column"
        elif not in_curr:
            status = "missing_column"
        elif base_dtype != curr_dtype:
            status = "dtype_changed"
        else:
            status = "ok"

        rows.append({
            "column": col,
            "baseline_dtype": base_dtype,
            "current_dtype": curr_dtype,
            "status": status,
        })
    return pd.DataFrame(rows)


def fairness_report(
    y_true,
    y_pred,
    sensitive: pd.Series,
    positive_label=None,
    min_group_size: int = 20,
) -> pd.DataFrame:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    sensitive = pd.Series(sensitive).reset_index(drop=True).astype(str)

    if positive_label is None:
        classes = sorted(pd.Series(y_true).dropna().unique().tolist(), key=str)
        if len(classes) != 2:
            raise ValueError("Fairness report currently supports binary classification.")
        positive_label = classes[1]

    rows = []
    for group, idx in sensitive.groupby(sensitive).groups.items():
        idx = list(idx)
        if len(idx) < min_group_size:
            continue
        yt = y_true.iloc[idx]
        yp = y_pred.iloc[idx]
        rows.append({
            "group": group,
            "n": len(idx),
            "accuracy": accuracy_score(yt, yp),
            "precision_positive": precision_score(yt, yp, pos_label=positive_label, zero_division=0),
            "recall_positive": recall_score(yt, yp, pos_label=positive_label, zero_division=0),
            "f1_positive": f1_score(yt, yp, pos_label=positive_label, zero_division=0),
            "positive_prediction_rate": float(np.mean(np.asarray(yp) == positive_label)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    best_recall = out["recall_positive"].max()
    worst_recall = out["recall_positive"].min()
    best_ppr = out["positive_prediction_rate"].max()
    worst_ppr = out["positive_prediction_rate"].min()
    out["recall_gap_vs_best"] = best_recall - out["recall_positive"]
    out["positive_rate_gap_vs_best"] = best_ppr - out["positive_prediction_rate"]
    out.attrs["recall_gap"] = float(best_recall - worst_recall)
    out.attrs["positive_rate_gap"] = float(best_ppr - worst_ppr)
    return out


def calibration_report(model, X_test: pd.DataFrame, y_test: pd.Series, bins: int = 10):
    if not hasattr(model, "predict_proba"):
        raise ValueError("Calibration requires predict_proba.")
    classes = list(model.classes_)
    if len(classes) != 2:
        raise ValueError("Calibration currently supports binary classification.")

    positive = classes[1]
    y_bin = (pd.Series(y_test).values == positive).astype(int)
    probs = model.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_bin, probs, n_bins=bins, strategy="quantile")
    brier = brier_score_loss(y_bin, probs)
    return pd.DataFrame({
        "mean_predicted_probability": mean_pred,
        "fraction_positive": frac_pos,
    }), float(brier)


def learning_curve_report(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    folds: int = 5,
) -> pd.DataFrame:
    if task == "classification":
        scoring = "f1_weighted"
    else:
        scoring = "neg_root_mean_squared_error"

    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=folds,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    if task == "regression":
        train_scores = -train_scores
        val_scores = -val_scores

    return pd.DataFrame({
        "train_size": sizes,
        "train_score_mean": train_scores.mean(axis=1),
        "train_score_std": train_scores.std(axis=1),
        "validation_score_mean": val_scores.mean(axis=1),
        "validation_score_std": val_scores.std(axis=1),
    })


def batch_predict(
    model,
    new_df: pd.DataFrame,
    include_proba: bool = True,
) -> pd.DataFrame:
    pred = model.predict(new_df)
    out = new_df.copy()
    out["prediction"] = pred

    if include_proba and hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(new_df)
            classes = list(model.classes_)
            for i, cls in enumerate(classes):
                out[f"probability_{cls}"] = probs[:, i]
        except Exception:
            pass
    return out


def benchmark_metric_snapshot(benchmark, task: str) -> Dict[str, float]:
    row = benchmark.leaderboard.loc[
        benchmark.leaderboard["model"] == benchmark.best_model_name
    ].iloc[0]
    if task == "classification":
        keys = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"]
    else:
        keys = ["mae", "rmse", "r2"]
    metrics = {}
    for key in keys:
        if key in row and pd.notna(row[key]):
            metrics[key] = float(row[key])
    return metrics


def generate_model_card(
    model_name: str,
    task: str,
    target: str,
    dataset_name: str,
    metrics: Dict[str, float],
    experiment_id: Optional[str] = None,
    notes: str = "",
) -> str:
    lines = [
        f"# Model Card — {model_name}",
        "",
        "## Overview",
        f"- Dataset: {dataset_name}",
        f"- Task: {task}",
        f"- Target: `{target}`",
    ]
    if experiment_id:
        lines.append(f"- Experiment ID: `{experiment_id}`")
    lines.extend(["", "## Evaluation metrics"])
    for k, v in metrics.items():
        lines.append(f"- {k}: {v:.4f}")
    lines.extend([
        "",
        "## Intended use",
        "Educational and portfolio demonstration of an ML workflow. Validate independently before any production or high-stakes use.",
        "",
        "## Known limitations",
        "- Metrics are dataset- and split-dependent.",
        "- Fairness analysis requires an appropriate user-selected grouping field.",
        "- Drift analysis is statistical monitoring, not proof of causality.",
        "- Community Cloud session state is not a durable production registry.",
        "",
        "## Notes",
        notes or "No additional notes.",
    ])
    return "\n".join(lines)
