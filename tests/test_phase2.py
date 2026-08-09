import io
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes

from ml_atlas.data_lab import detect_task, drop_identifier_like_columns
from ml_atlas.profiling import profile_dataframe, potential_issues
from ml_atlas.automl import benchmark_models, feature_importance_table, serialize_model


def test_detect_task_classification():
    y = pd.Series([0, 1, 0, 1, 1, 0])
    result = detect_task(y)
    assert result.task == "classification"


def test_detect_task_regression():
    y = pd.Series([0.1, 1.2, 2.7, 3.5, 4.8, 5.9, 7.1, 8.3, 9.4, 10.6] * 4)
    result = detect_task(y)
    assert result.task == "regression"


def test_identifier_drop():
    df = pd.DataFrame({
        "customer_id": range(100),
        "x": range(100),
        "target": [0, 1] * 50,
    })
    out, removed = drop_identifier_like_columns(df, "target")
    assert "customer_id" in removed
    assert "customer_id" not in out.columns


def test_profile():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "x", "y"]})
    p = profile_dataframe(df)
    assert p.rows == 3
    assert p.columns == 2
    assert p.missing_cells == 1


def test_classification_benchmark_small():
    bunch = load_iris(as_frame=True)
    df = bunch.frame.rename(columns={"target": "label"})
    result = benchmark_models(df, "label", "classification", max_models=3)
    assert result.best_model_name
    assert not result.leaderboard.empty
    assert serialize_model(result.best_model)


def test_regression_benchmark_small():
    bunch = load_diabetes(as_frame=True)
    df = bunch.frame.rename(columns={"target": "label"})
    result = benchmark_models(df, "label", "regression", max_models=3)
    assert result.best_model_name
    assert "r2" in result.leaderboard.columns
