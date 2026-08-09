# Deployment Guide

## Streamlit Community Cloud

Deploy `app.py` from the `main` branch.

## Local Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Local API

```bash
python scripts/train_demo_model.py
uvicorn api.main:app --reload --port 8000
```

API docs:

```text
http://localhost:8000/docs
```

## Docker Compose

```bash
docker compose up --build
```

Services:

- Streamlit UI: `http://localhost:8501`
- FastAPI: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

## Custom production model

Set:

```bash
export ML_ATLAS_MODEL_PATH=/path/to/fitted_pipeline.pkl
```

The artifact must expose `.predict(...)`; classification probability output is included when `.predict_proba(...)` exists.
