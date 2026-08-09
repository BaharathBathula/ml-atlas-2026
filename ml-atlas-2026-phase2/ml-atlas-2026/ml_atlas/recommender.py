def recommend(task: str, n_samples: int, n_features: int, interpretability: bool, nonlinear: bool):
    notes = []
    if task == "classification":
        if interpretability:
            primary = "Logistic Regression" if not nonlinear else "Decision Tree"
        elif nonlinear:
            primary = "Random Forest"
        else:
            primary = "Logistic Regression"
        if n_samples < 5000:
            secondary = "SVM"
        else:
            secondary = "Gradient Boosting"
    elif task == "regression":
        if interpretability:
            primary = "Ridge Regression"
        elif nonlinear:
            primary = "Polynomial Regression"
        else:
            primary = "Linear Regression"
        secondary = "Lasso Regression" if n_features > 20 else "Ridge Regression"
    elif task == "clustering":
        primary = "K-Means" if n_samples > 200 else "Agglomerative"
        secondary = "DBSCAN"
        notes.append("Use DBSCAN when irregular cluster shapes and noise detection matter.")
    else:
        primary = "PCA"
        secondary = "Truncated SVD"

    if n_features > 100:
        notes.append("High dimensionality: consider dimensionality reduction or feature selection before modeling.")
    if n_samples < 200:
        notes.append("Small sample size: prefer simpler models and cross-validation; avoid overclaiming performance.")
    return primary, secondary, notes
