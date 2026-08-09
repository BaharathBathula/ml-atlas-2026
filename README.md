# ML Atlas 2026 — Interactive Machine Learning Taxonomy & Algorithm Playground

A portfolio-grade educational project that turns a machine-learning taxonomy into an **interactive explorer, runnable model playground, and algorithm recommender**.

> The taxonomy is inspired by a 2026 machine-learning roadmap diagram. Product/model names in the Generative AI section are treated as examples, not a benchmark of current model leadership.

## What this repository demonstrates

- A structured taxonomy covering supervised, unsupervised, reinforcement, self-supervised, semi-supervised, transfer, deep, generative, ensemble, and probabilistic learning.
- Interactive model training on built-in datasets.
- Classification, regression, clustering, dimensionality reduction, and ensemble demos.
- Model evaluation and visualization.
- A practical "Which algorithm should I try?" recommender.
- Clean Python package structure.
- Unit tests.
- GitHub Actions CI.
- Docker support.
- A Streamlit UI suitable for a GitHub/LinkedIn portfolio demo.

## Architecture

```text
ml-atlas-2026/
├─ app.py
├─ ml_atlas/
│  ├─ datasets.py
│  ├─ registry.py
│  ├─ trainer.py
│  └─ recommender.py
├─ data/taxonomy.json
├─ examples/
├─ tests/
├─ docs/
├─ .github/workflows/ci.yml
├─ requirements.txt
├─ Dockerfile
├─ Makefile
└─ LICENSE
```

## Quick start

```bash
git clone <your-repo-url>
cd ml-atlas-2026

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

## Docker

```bash
docker build -t ml-atlas-2026 .
docker run --rm -p 8501:8501 ml-atlas-2026
```

Open `http://localhost:8501`.

## Test

```bash
pytest -q
```

## Core playground coverage

### Supervised learning
- Logistic Regression
- Decision Tree
- k-NN
- SVM
- Naive Bayes
- Linear Regression
- Ridge
- Lasso
- Polynomial Regression

### Unsupervised learning
- K-Means
- DBSCAN
- Agglomerative Clustering
- Mean Shift
- PCA
- Truncated SVD

### Ensemble learning
- Random Forest
- AdaBoost
- Gradient Boosting
- Voting Classifier
- Stacking Classifier

The full taxonomy is larger than the executable scikit-learn playground. Deep learning, RL, self-supervised learning, transfer learning, generative AI, and probabilistic graphical models are included as **taxonomy/learning modules** and extension points because implementing every family in one environment would create unnecessary dependency and compute overhead.

## Portfolio talking points

1. **Taxonomy-to-software translation** — transformed a conceptual ML map into a structured software system.
2. **Registry pattern** — algorithms are registered centrally, making the app extensible.
3. **Task-aware evaluation** — metrics change automatically by ML task.
4. **Practical MLOps hygiene** — tests, Docker, CI, deterministic seeds, modular code.
5. **Responsible scope** — avoids pretending every advanced model family can be meaningfully trained on a laptop.

## Suggested GitHub repository description

> Interactive ML taxonomy, algorithm playground, model recommender, evaluation dashboard, tests, Docker, and CI — built as a practical map of the modern machine-learning landscape.

## Suggested topics

`machine-learning` `data-science` `streamlit` `scikit-learn` `mlops` `deep-learning` `generative-ai` `portfolio-project`

## Roadmap

- [ ] Add PyTorch deep-learning mini-labs
- [ ] Add Gymnasium reinforcement-learning demos
- [ ] Add association-rule mining
- [ ] Add UMAP
- [ ] Add SHAP explainability
- [ ] Add model export/import
- [ ] Add user-uploaded CSV support
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add benchmark tracking

## Disclaimer

This is an educational portfolio project. It is not a production AutoML platform and should not be used for high-stakes decisions without domain-specific validation, monitoring, governance, and security controls.


## Phase 2 — Real Dataset Intelligence

The project now supports real user datasets in addition to built-in examples.

### Dataset Lab capabilities

- Upload `.csv`, `.xlsx`, or `.xls`
- Preview and profile the dataset
- Detect missing values, duplicates, constants, high-cardinality columns, and identifier-like fields
- Review numeric descriptive statistics and strongest numeric correlations
- Select the target column
- Infer classification vs regression from the target
- Override the inferred task when domain knowledge says otherwise
- Apply leakage-safe preprocessing:
  - median imputation for numeric fields
  - standard scaling for numeric fields
  - most-frequent imputation for categoricals
  - one-hot encoding with unknown-category handling
- Benchmark multiple candidate models
- Rank models using task-appropriate metrics
- Visualize confusion matrices or actual-vs-predicted performance
- Show feature importance for supported models
- Download the fitted preprocessing + model pipeline
- Download held-out predictions

### Candidate models

**Classification**
- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- k-NN
- SVM

**Regression**
- Linear Regression
- Ridge
- Lasso
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting

### Design constraint

This is an AutoML-style portfolio application, not a claim to production AutoML. It intentionally avoids unsafe shortcuts such as fitting preprocessing on the full dataset before the train/test split.


## Phase 3 — Optimization & Explainability

Phase 3 upgrades ML Atlas from simple hold-out benchmarking into a more credible model-evaluation workflow.

### Added capabilities

- K-fold cross-validation
- Stratified cross-validation for classification
- Bounded GridSearchCV hyperparameter tuning
- ROC curve and ROC-AUC for binary classifiers
- Precision–Recall curve and average precision
- Decision-threshold analysis
- Threshold optimization for positive-class F1, recall, precision, or accuracy
- Regression residual diagnostics
- Permutation feature importance on held-out data
- Tuned model export
- Downloadable Markdown experiment report
- Identifier-like classification-target safety check

### Why Phase 3 matters

A single train/test split can produce unstable conclusions. Phase 3 explicitly measures out-of-sample variability, separates model selection from business-threshold selection, and adds diagnostics that reveal when a model's headline score hides weak behavior.

Hyperparameter spaces are intentionally bounded so the public Streamlit deployment remains usable on modest Community Cloud compute.
