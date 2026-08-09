from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes

DATASETS = {
    "Iris (classification)": ("classification", load_iris),
    "Wine (classification)": ("classification", load_wine),
    "Breast Cancer (classification)": ("classification", load_breast_cancer),
    "Diabetes (regression)": ("regression", load_diabetes),
}

def load_dataset(name: str):
    task, loader = DATASETS[name]
    bunch = loader(as_frame=True)
    X = bunch.data
    y = bunch.target
    return task, X, y, bunch
