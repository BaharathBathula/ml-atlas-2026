import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

from ml_atlas.datasets import DATASETS, load_dataset
from ml_atlas.registry import ALGORITHMS, names_for_task
from ml_atlas.trainer import train_supervised, run_clustering, run_reduction
from ml_atlas.recommender import recommend
from ml_atlas.data_lab import load_uploaded_dataframe, detect_task, drop_identifier_like_columns
from ml_atlas.profiling import profile_dataframe, numeric_summary, top_correlations, potential_issues
from ml_atlas.automl import benchmark_models, feature_importance_table, serialize_model

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
st.caption("Interactive machine-learning taxonomy • dataset intelligence • model benchmarking • practical recommendations")

tabs = st.tabs([
    "Taxonomy Explorer",
    "Model Playground",
    "Dataset Lab",
    "Algorithm Recommender",
    "Project Notes"
])

with tabs[0]:
    st.subheader("Explore the machine-learning landscape")
    data = taxonomy()
    rows = flatten(data)
    df = pd.DataFrame(rows)
    search = st.text_input("Search a method, model family, or concept", placeholder="e.g. clustering, transformer, Bayesian")
    if search:
        df = df[df["path"].str.contains(search, case=False, regex=False)]
    st.dataframe(df[["path"]], use_container_width=True, hide_index=True)
    st.info("The taxonomy is intentionally broader than the runnable playground. Advanced families are represented as learning paths and extension points.")

with tabs[1]:
    st.subheader("Train and evaluate representative algorithms")
    mode = st.radio("Mode", ["Supervised", "Clustering", "Dimensionality Reduction"], horizontal=True)

    if mode == "Supervised":
        dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), key="play_dataset")
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
        dataset_name = st.selectbox("Dataset", ["Iris (classification)", "Wine (classification)"], key="cluster_dataset")
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
        dataset_name = st.selectbox("Dataset", ["Iris (classification)", "Wine (classification)", "Breast Cancer (classification)"], key="reduce_dataset")
        _, X, y, _ = load_dataset(dataset_name)
        algo = st.selectbox("Algorithm", names_for_task("dimensionality_reduction"))
        if st.button("Reduce to 2D", type="primary"):
            result = run_reduction(algo, X)
            emb = result.embedding
            plot_df = pd.DataFrame({"Component 1": emb[:,0], "Component 2": emb[:,1], "Target": y.astype(str)})
            fig = px.scatter(plot_df, x="Component 1", y="Component 2", color="Target", title=f"{algo} 2D projection")
            st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("📊 Dataset Lab — Phase 2")
    st.write("Upload a real dataset, profile its quality, infer the ML task, benchmark multiple models, and export the best pipeline.")

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        try:
            df = load_uploaded_dataframe(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            st.stop()

        st.success(f"Loaded {len(df):,} rows × {df.shape[1]:,} columns")
        st.dataframe(df.head(50), use_container_width=True)

        profile = profile_dataframe(df)

        st.markdown("### 1. Dataset profile")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{profile.rows:,}")
        m2.metric("Columns", profile.columns)
        m3.metric("Missing cells", f"{profile.missing_cells:,}")
        m4.metric("Duplicate rows", f"{profile.duplicate_rows:,}")
        m5.metric("Memory", f"{profile.memory_mb} MB")

        with st.expander("Column quality summary", expanded=True):
            st.dataframe(profile.column_summary, use_container_width=True, hide_index=True)

        issues = potential_issues(df)
        if issues:
            with st.expander("Potential data-quality issues"):
                for issue in issues:
                    st.warning(issue)
        else:
            st.success("No obvious structural data-quality issues detected.")

        numeric_df = numeric_summary(df)
        if not numeric_df.empty:
            with st.expander("Numeric statistics"):
                st.dataframe(numeric_df, use_container_width=True, hide_index=True)

        corr_df = top_correlations(df)
        if not corr_df.empty:
            with st.expander("Strongest numeric correlations"):
                st.dataframe(corr_df, use_container_width=True, hide_index=True)

        st.markdown("### 2. Configure prediction target")
        target = st.selectbox("Target column", options=df.columns.tolist())

        detection = detect_task(df[target])
        st.info(f"Suggested task: **{detection.task.title()}** ({detection.confidence} confidence) — {detection.reason}")

        task = st.radio(
            "Problem type",
            ["classification", "regression"],
            index=0 if detection.task == "classification" else 1,
            horizontal=True
        )

        cleaned_df, removed_ids = drop_identifier_like_columns(df, target)
        if removed_ids:
            st.caption("Identifier-like columns automatically excluded: " + ", ".join(removed_ids))

        st.markdown("### 3. Benchmark models")
        test_size = st.slider("Test-set size", 0.20, 0.40, 0.25, 0.05)
        st.caption("Preprocessing is learned on the training split only: median imputation + scaling for numeric features, most-frequent imputation + one-hot encoding for categorical features.")

        if st.button("Run model benchmark", type="primary", key="benchmark"):
            try:
                with st.spinner("Training candidate models..."):
                    result = benchmark_models(cleaned_df, target=target, task=task, test_size=test_size)
                st.session_state["phase2_result"] = result
                st.session_state["phase2_task"] = task
                st.session_state["phase2_target"] = target
            except Exception as exc:
                st.error(f"Benchmark failed: {exc}")

        result = st.session_state.get("phase2_result")
        if result is not None and st.session_state.get("phase2_target") == target:
            leaderboard = result.leaderboard.copy()

            st.markdown("### 4. Model leaderboard")
            numeric_cols = leaderboard.select_dtypes(include=np.number).columns
            leaderboard[numeric_cols] = leaderboard[numeric_cols].round(4)
            st.dataframe(leaderboard, use_container_width=True, hide_index=True)

            st.success(f"Best model: **{result.best_model_name}**")

            if task == "classification":
                cm = confusion_matrix(result.y_test, result.best_predictions)
                fig = px.imshow(cm, text_auto=True, title=f"{result.best_model_name} — Confusion Matrix",
                                labels={"x": "Predicted", "y": "Actual"})
                st.plotly_chart(fig, use_container_width=True)
            else:
                pred_df = pd.DataFrame({
                    "Actual": np.asarray(result.y_test),
                    "Predicted": np.asarray(result.best_predictions)
                })
                fig = px.scatter(pred_df, x="Actual", y="Predicted",
                                 title=f"{result.best_model_name} — Actual vs Predicted")
                st.plotly_chart(fig, use_container_width=True)

            fi = feature_importance_table(result)
            if not fi.empty:
                st.markdown("### 5. Feature importance")
                fig = px.bar(fi.sort_values("importance"), x="importance", y="feature", orientation="h",
                             title=f"Top features — {result.best_model_name}")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 6. Export best pipeline")
            model_bytes = serialize_model(result.best_model)
            st.download_button(
                "Download trained model (.pkl)",
                data=model_bytes,
                file_name=f"{result.best_model_name.lower().replace(' ', '_')}_pipeline.pkl",
                mime="application/octet-stream"
            )

            prediction_export = pd.DataFrame({
                "actual": np.asarray(result.y_test),
                "predicted": np.asarray(result.best_predictions),
            }).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download test predictions (.csv)",
                data=prediction_export,
                file_name="test_predictions.csv",
                mime="text/csv"
            )

with tabs[3]:
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

with tabs[4]:
    st.markdown("""
### Phase 2 status

The project now has two layers:

**Learning layer**
- ML taxonomy explorer
- built-in algorithm playground
- algorithm recommender

**Real dataset layer**
- CSV / Excel upload
- dataset profiling
- missing-value and duplicate detection
- target-column selection
- task inference
- identifier-like column removal
- leakage-safe preprocessing pipelines
- multi-model benchmarking
- leaderboard
- classification/regression evaluation
- feature importance where supported
- model export
- prediction export

### Scope discipline

This is still an educational and portfolio AutoML-style system rather than a production AutoML platform. Production use would require stronger schema contracts, model governance, fairness checks, drift monitoring, security controls, persistent experiment tracking, and reproducible artifact storage.
""")
