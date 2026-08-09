from sklearn.datasets import load_iris
from ml_atlas.trainer import train_supervised

X, y = load_iris(return_X_y=True, as_frame=True)
result = train_supervised("Random Forest", X, y, "classification")
print(result.metrics)
