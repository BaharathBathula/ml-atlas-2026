# Architecture

## Components

### Taxonomy layer
`data/taxonomy.json` is the source of truth for the conceptual ML map.

### Algorithm registry
`ml_atlas/registry.py` centralizes algorithm metadata and constructors.

### Training/evaluation layer
`ml_atlas/trainer.py` contains task-aware training and evaluation functions.

### Recommendation layer
`ml_atlas/recommender.py` provides simple rules to suggest starting algorithms.

### UI
`app.py` exposes the taxonomy, playground, and recommender in Streamlit.

## Why a registry pattern?

The application should not scatter algorithm-specific initialization throughout the UI. A registry:
- makes algorithms discoverable;
- keeps UI logic generic;
- enables tests to enumerate capabilities;
- provides one extension point for new algorithms.

## Extension strategy

Future packages can add separate optional modules:
- `ml_atlas/deep_learning/` — PyTorch labs
- `ml_atlas/rl/` — Gymnasium
- `ml_atlas/generative/` — Hugging Face / provider APIs
- `ml_atlas/graph/` — PyTorch Geometric
- `ml_atlas/probabilistic/` — pgmpy / PyMC
