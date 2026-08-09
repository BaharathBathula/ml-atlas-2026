from dataclasses import dataclass
from typing import Callable, Dict

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.decomposition import PCA, TruncatedSVD

RANDOM_STATE = 42

@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    task: str
    builder: Callable
    description: str

def _voting():
    return VotingClassifier(
        estimators=[
            ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
            ("dt", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
            ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
        ],
        voting="soft",
    )

def _stacking():
    base = [
        ("rf", RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE)),
        ("gb", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ]
    return StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=1000)
    )

ALGORITHMS: Dict[str, AlgorithmSpec] = {
    "Logistic Regression": AlgorithmSpec(
        "Logistic Regression", "classification",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Linear probabilistic classifier; strong baseline for separable tabular data."
    ),
    "Naive Bayes": AlgorithmSpec(
        "Naive Bayes", "classification", lambda: GaussianNB(),
        "Fast probabilistic classifier based on conditional independence assumptions."
    ),
    "SVM": AlgorithmSpec(
        "SVM", "classification",
        lambda: make_pipeline(StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE)),
        "Margin-based classifier that can model nonlinear boundaries using kernels."
    ),
    "Decision Tree": AlgorithmSpec(
        "Decision Tree", "classification",
        lambda: DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Interpretable tree model for nonlinear decision boundaries."
    ),
    "k-NN": AlgorithmSpec(
        "k-NN", "classification",
        lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Instance-based learner that predicts from nearby observations."
    ),
    "Random Forest": AlgorithmSpec(
        "Random Forest", "classification",
        lambda: RandomForestClassifier(n_estimators=180, random_state=RANDOM_STATE),
        "Bagged tree ensemble that reduces variance and handles nonlinear interactions."
    ),
    "AdaBoost": AlgorithmSpec(
        "AdaBoost", "classification",
        lambda: AdaBoostClassifier(n_estimators=120, random_state=RANDOM_STATE),
        "Sequential boosting method that focuses on difficult examples."
    ),
    "Gradient Boosting": AlgorithmSpec(
        "Gradient Boosting", "classification",
        lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Additive tree boosting optimized stage by stage."
    ),
    "Voting Classifier": AlgorithmSpec(
        "Voting Classifier", "classification", _voting,
        "Combines multiple classifiers through soft probability voting."
    ),
    "Stacking Classifier": AlgorithmSpec(
        "Stacking Classifier", "classification", _stacking,
        "Uses a meta-model to learn how to combine base-model predictions."
    ),
    "Linear Regression": AlgorithmSpec(
        "Linear Regression", "regression", lambda: LinearRegression(),
        "Ordinary least-squares regression baseline."
    ),
    "Ridge Regression": AlgorithmSpec(
        "Ridge Regression", "regression",
        lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "L2-regularized regression, useful under multicollinearity."
    ),
    "Lasso Regression": AlgorithmSpec(
        "Lasso Regression", "regression",
        lambda: make_pipeline(StandardScaler(), Lasso(alpha=0.05, max_iter=10000)),
        "L1-regularized regression that can drive coefficients to zero."
    ),
    "Polynomial Regression": AlgorithmSpec(
        "Polynomial Regression", "regression",
        lambda: make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=1.0)),
        "Adds polynomial features before regularized regression."
    ),
    "K-Means": AlgorithmSpec(
        "K-Means", "clustering",
        lambda: make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)),
        "Centroid-based clustering optimized for compact, roughly spherical groups."
    ),
    "DBSCAN": AlgorithmSpec(
        "DBSCAN", "clustering",
        lambda: make_pipeline(StandardScaler(), DBSCAN(eps=0.8, min_samples=5)),
        "Density-based clustering that can mark sparse observations as noise."
    ),
    "Agglomerative": AlgorithmSpec(
        "Agglomerative", "clustering",
        lambda: make_pipeline(StandardScaler(), AgglomerativeClustering(n_clusters=3)),
        "Hierarchical bottom-up clustering."
    ),
    "Mean Shift": AlgorithmSpec(
        "Mean Shift", "clustering",
        lambda: make_pipeline(StandardScaler(), MeanShift()),
        "Mode-seeking clustering that does not require a predeclared cluster count."
    ),
    "PCA": AlgorithmSpec(
        "PCA", "dimensionality_reduction",
        lambda: make_pipeline(StandardScaler(), PCA(n_components=2)),
        "Linear projection preserving maximum variance."
    ),
    "Truncated SVD": AlgorithmSpec(
        "Truncated SVD", "dimensionality_reduction",
        lambda: make_pipeline(StandardScaler(), TruncatedSVD(n_components=2, random_state=RANDOM_STATE)),
        "Low-rank matrix factorization useful for high-dimensional data."
    ),
}

def names_for_task(task: str):
    return [name for name, spec in ALGORITHMS.items() if spec.task == task]
