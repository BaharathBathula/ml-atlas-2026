import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

from ml_atlas.datasets import DATASETS, load_dataset
from ml_atlas.registry import ALGORITHMS, names_for_task
from ml_atlas.trainer import train_supervised, run_clustering, run_reduction
from ml_atlas.recommender import recommend

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="ML Atlas 2026", page_icon="🧠", layout="wide")

@st.cache_data
def taxonomy():
    return json.loads((ROOT / "data" / "taxonomy.json").read_text(encoding="utf-8"))

def flatten(node, path=()):
    rows = []
    if isinstance(node, dict):
        for k, v in node.items():
            rows.append({"path": " → ".join(path + (k,)), "node": k, "depth": len(path)})
            rows.extend(flatten(v, path + (k,)))
    elif isinstance(node, list):
        for item in node:
            rows.append({"path": " → ".join(path + (item,)), "node": item, "depth": len(path)})
    return rows

st.title("🧠 ML Atlas 2026")
st.caption("Interactive machine-learning taxonomy • algorithm playground • practical model recommender")

tab1, tab2, tab3, tab4 = st.tabs(["Taxonomy Explorer", "Model Playground", "Algorithm Recommender", "Project Notes"])

with tab1:
    st.subheader("Explore the machine-learning landscape")
    data = taxonomy()
    rows = flatten(data)
    df = pd.DataFrame(rows)
    search = st.text_input("Search a method, model family, or concept", placeholder="e.g. clustering, transformer, Bayesian")
    if search:
        df = df[df["path"].str.contains(search, case=False, regex=False)]
    st.dataframe(df[["path"]], use_container_width=True, hide_index=True)
    st.info("The taxonomy is intentionally broader than the runnable playground. Advanced families are represented as learning paths and extension points.")

with tab2:
    st.subheader("Train and evaluate representative algorithms")
    mode = st.radio("Mode", ["Supervised", "Clustering", "Dimensionality Reduction"], horizontal=True)

    if mode == "Supervised":
        dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
        task, X, y, bunch = load_dataset(dataset_name)
        algo = st.selectbox("Algorithm", names_for_task(task))
        st.write(ALGORITHMS[algo].description)

        if st.button("Train model", type="primary"):
            result = train_supervised(algo, X, y, task)
            cols = st.columns(len(result.metrics))
            for c, (k, v) in zip(cols, result.metrics.items()):
                c.metric(k.replace("_", " ").title(), v)

            if task == "classification":
                cm = confusion_matrix(result.y_true, result.y_pred)
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                                labels={"x": "Predicted", "y": "Actual"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                pred_df = pd.DataFrame({"Actual": result.y_true.values, "Predicted": result.y_pred})
                fig = px.scatter(pred_df, x="Actual", y="Predicted", title="Actual vs Predicted")
                st.plotly_chart(fig, use_container_width=True)

    elif mode == "Clustering":
        dataset_name = st.selectbox("Dataset", ["Iris (classification)", "Wine (classification)"])
        _, X, _, _ = load_dataset(dataset_name)
        algo = st.selectbox("Algorithm", names_for_task("clustering"))
        if st.button("Run clustering", type="primary"):
            result = run_clustering(algo, X)
            st.json(result.metrics)
            proj = run_reduction("PCA", X).embedding
            plot_df = pd.DataFrame({"PC1": proj[:, 0], "PC2": proj[:, 1], "Cluster": result.labels.astype(str)})
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", title=f"{algo} clusters projected with PCA")
            st.plotly_chart(fig, use_container_width=True)

    else:
        dataset_name = st.selectbox("Dataset", ["Iris (classification)", "Wine (classification)", "Breast Cancer (classification)"])
        _, X, y, _ = load_dataset(dataset_name)
        algo = st.selectbox("Algorithm", names_for_task("dimensionality_reduction"))
        if st.button("Reduce to 2D", type="primary"):
            result = run_reduction(algo, X)
            emb = result.embedding
            plot_df = pd.DataFrame({"Component 1": emb[:,0], "Component 2": emb[:,1], "Target": y.astype(str)})
            fig = px.scatter(plot_df, x="Component 1", y="Component 2", color="Target", title=f"{algo} 2D projection")
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Which algorithm should I try first?")
    task = st.selectbox("Problem type", ["classification", "regression", "clustering", "dimensionality_reduction"])
    n_samples = st.number_input("Approximate number of samples", min_value=20, value=1000, step=100)
    n_features = st.number_input("Approximate number of features", min_value=1, value=20, step=1)
    interpretability = st.checkbox("Interpretability is important", value=True)
    nonlinear = st.checkbox("I expect nonlinear patterns", value=False)

    if st.button("Recommend"):
        primary, secondary, notes = recommend(task, n_samples, n_features, interpretability, nonlinear)
        c1, c2 = st.columns(2)
        c1.success(f"Start with: **{primary}**")
        c2.info(f"Compare against: **{secondary}**")
        for note in notes:
            st.warning(note)
        st.caption("This is a heuristic starting point, not an AutoML decision engine.")

with tab4:
    st.markdown("""
### Why this project is intentionally scoped

A single repository claiming to *fully implement* every branch of modern ML would be misleading. Reinforcement learning, self-supervised learning, LLMs, diffusion models, GNNs, and graphical models each deserve their own environments, datasets, compute budgets, and evaluation methodology.

This repository therefore separates:
- **taxonomy coverage** — broad conceptual map;
- **runnable coverage** — representative classical ML algorithms;
- **extension architecture** — clean places to add PyTorch, Gymnasium, Hugging Face, or probabilistic programming later.

That trade-off makes the project more credible in a technical portfolio.
""")
