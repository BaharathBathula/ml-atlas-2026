from pathlib import Path
import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("artifacts/demo_iris_pipeline.pkl")
OUT.parent.mkdir(parents=True, exist_ok=True)

bunch = load_iris(as_frame=True)
X = bunch.data
y = bunch.target

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, random_state=42),
)
model.fit(X, y)

with OUT.open("wb") as f:
    pickle.dump(model, f)

print(f"Saved demo model to {OUT}")
print("Expected input columns:")
for col in X.columns:
    print(f" - {col}")
