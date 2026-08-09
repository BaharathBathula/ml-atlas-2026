from sklearn.datasets import load_iris, load_diabetes
from ml_atlas.registry import names_for_task
from ml_atlas.trainer import train_supervised, run_clustering, run_reduction
from ml_atlas.recommender import recommend

def test_classification_training():
    X, y = load_iris(return_X_y=True, as_frame=True)
    result = train_supervised("Logistic Regression", X, y, "classification")
    assert result.metrics["accuracy"] > 0.8

def test_regression_training():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    result = train_supervised("Ridge Regression", X, y, "regression")
    assert "r2" in result.metrics

def test_clustering():
    X, _ = load_iris(return_X_y=True, as_frame=True)
    result = run_clustering("K-Means", X)
    assert result.metrics["clusters_found"] >= 2

def test_reduction():
    X, _ = load_iris(return_X_y=True, as_frame=True)
    result = run_reduction("PCA", X)
    assert result.embedding.shape[1] == 2

def test_registry_has_core_tasks():
    assert names_for_task("classification")
    assert names_for_task("regression")
    assert names_for_task("clustering")
    assert names_for_task("dimensionality_reduction")

def test_recommender():
    p, s, notes = recommend("classification", 1000, 20, True, False)
    assert p and s
