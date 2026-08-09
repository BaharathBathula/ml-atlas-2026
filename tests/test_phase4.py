import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from ml_atlas.automl import benchmark_models
from ml_atlas.phase4 import (
    make_experiment_record,
    experiments_to_frame,
    register_model,
    registry_to_frame,
    population_stability_index,
    drift_report,
    validate_schema,
    fairness_report,
    calibration_report,
    batch_predict,
    benchmark_metric_snapshot,
    generate_model_card,
)


def test_experiment_record_and_registry():
    rec = make_experiment_record("demo.csv", "target", "classification", "SVM", {"accuracy": 0.8})
    frame = experiments_to_frame([rec])
    assert frame.iloc[0]["experiment_id"].startswith("exp-")

    reg = register_model([], "SVM", rec.experiment_id, "candidate")
    assert reg.version == 1
    reg_df = registry_to_frame([reg])
    assert reg_df.iloc[0]["stage"] == "candidate"


def test_psi_stable_and_shifted():
    rng = np.random.default_rng(42)
    base = pd.Series(rng.normal(0, 1, 1000))
    same = pd.Series(rng.normal(0, 1, 1000))
    shifted = pd.Series(rng.normal(2, 1, 1000))
    assert population_stability_index(base, same) < population_stability_index(base, shifted)


def test_drift_and_schema():
    base = pd.DataFrame({"a": [1,2,3,4], "b": ["x","y","x","y"]})
    curr = pd.DataFrame({"a": [5,6,7,8], "b": ["x","x","x","y"], "c": [1,1,1,1]})
    report = drift_report(base, curr)
    assert "psi" in report.columns
    schema = validate_schema(base, curr)
    assert "new_column" in schema["status"].values


def test_fairness_report_binary():
    y_true = pd.Series([0,1,0,1] * 20)
    y_pred = pd.Series([0,1,0,0] * 20)
    sensitive = pd.Series(["A"] * 40 + ["B"] * 40)
    report = fairness_report(y_true, y_pred, sensitive, positive_label=1, min_group_size=10)
    assert len(report) == 2
    assert "recall_positive" in report.columns


def test_calibration_and_batch_prediction():
    df = pd.DataFrame({
        "x1": list(range(120)),
        "x2": [0,1] * 60,
        "target": [0] * 60 + [1] * 60,
    })
    result = benchmark_models(df, "target", "classification", max_models=1)
    cal, brier = calibration_report(result.best_model, result.X_test, result.y_test, bins=5)
    assert not cal.empty
    assert brier >= 0

    batch = batch_predict(result.best_model, result.X_test.head(5))
    assert "prediction" in batch.columns
    assert len(batch) == 5


def test_metric_snapshot_and_model_card():
    bunch = load_iris(as_frame=True)
    df = bunch.frame.rename(columns={"target": "label"})
    result = benchmark_models(df, "label", "classification", max_models=1)
    metrics = benchmark_metric_snapshot(result, "classification")
    assert "accuracy" in metrics
    card = generate_model_card("Logistic Regression", "classification", "label", "iris.csv", metrics)
    assert "Model Card" in card
