import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

from ml_atlas.automl import benchmark_models
from ml_atlas.phase3 import (
    run_cross_validation,
    tune_model,
    threshold_analysis,
    permutation_explain,
    residual_diagnostics,
    build_experiment_report,
)


def test_phase3_cross_validation_classification():
    bunch = load_iris(as_frame=True)
    df = bunch.frame.rename(columns={"target": "label"})
    result = run_cross_validation(df, "label", "classification", "Logistic Regression", folds=3)
    assert result.folds == 3
    assert "f1_weighted_mean" in result.summary


def test_phase3_cross_validation_regression():
    bunch = load_diabetes(as_frame=True)
    df = bunch.frame.rename(columns={"target": "label"})
    result = run_cross_validation(df, "label", "regression", "Ridge Regression", folds=3)
    assert "rmse_mean" in result.summary


def test_phase3_tuning_classification():
    bunch = load_iris(as_frame=True)
    df = bunch.frame.rename(columns={"target": "label"})
    result = tune_model(df, "label", "classification", "Logistic Regression", folds=3)
    assert result.best_estimator is not None
    assert "model__C" in result.best_params


def test_threshold_analysis_binary():
    df = pd.DataFrame({
        "x1": list(range(100)),
        "x2": [0, 1] * 50,
        "target": [0] * 50 + [1] * 50,
    })
    result = benchmark_models(df, "target", "classification", max_models=1)
    out = threshold_analysis(result.best_model, result.X_test, result.y_test, optimize_for="f1")
    assert 0.1 <= out.recommended_threshold <= 0.9
    assert not out.threshold_table.empty


def test_permutation_importance():
    bunch = load_iris(as_frame=True)
    df = bunch.frame.rename(columns={"target": "label"})
    result = benchmark_models(df, "label", "classification", max_models=1)
    exp = permutation_explain(result.best_model, result.X_test, result.y_test, "classification", top_n=3)
    assert len(exp.importance) <= 3


def test_residual_diagnostics():
    out = residual_diagnostics([1, 2, 3], [1.1, 1.8, 3.2])
    assert "summary" in out
    assert "residuals" in out


def test_experiment_report():
    bunch = load_iris(as_frame=True)
    df = bunch.frame.rename(columns={"target": "label"})
    result = benchmark_models(df, "label", "classification", max_models=1)
    report = build_experiment_report("iris.csv", "label", "classification", result)
    assert "ML Atlas 2026" in report
    assert "Benchmark leaderboard" in report
