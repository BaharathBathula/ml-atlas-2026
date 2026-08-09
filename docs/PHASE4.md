# Phase 4 — Production ML & Governance

## Delivered

- Experiment history with unique experiment IDs
- Lightweight session-scoped model registry
- Model stages: candidate, staging, champion, archived
- Schema validation for incoming datasets
- PSI-based drift report
- Calibration curve and Brier score
- Group fairness metrics
- Learning curves
- Batch inference and CSV export
- Model-card generation

## Recommended commit

```text
feat: add Phase 4 production ML governance and inference
```

## Runtime validation

1. Run a Dataset Lab benchmark.
2. Open Production ML Lab.
3. Save the current experiment.
4. Register it as candidate.
5. Upload a second dataset for drift analysis.
6. Review calibration/fairness where applicable.
7. Generate a learning curve.
8. Upload unseen features for batch prediction.
9. Download the model card.

## Important limitation

Experiment tracking and model registry are session-scoped in the Streamlit demo. Durable experiment and registry storage is intentionally deferred to the production-packaging phase.
