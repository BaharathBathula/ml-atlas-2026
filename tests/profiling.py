from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    missing_cells: int
    duplicate_rows: int
    memory_mb: float
    column_summary: pd.DataFrame


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in datetime_cols]

    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "missing_pct": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })

    return DatasetProfile(
        rows=int(df.shape[0]),
        columns=int(df.shape[1]),
        numeric_columns=numeric,
        categorical_columns=categorical,
        datetime_columns=datetime_cols,
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        memory_mb=round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 3),
        column_summary=summary,
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    out = numeric.describe().T.reset_index().rename(columns={"index": "column"})
    return out


def top_correlations(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "correlation"])

    corr = numeric.corr(numeric_only=True)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                pairs.append((cols[i], cols[j], float(value)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pd.DataFrame(pairs[:limit], columns=["feature_1", "feature_2", "correlation"])


def potential_issues(df: pd.DataFrame) -> List[str]:
    issues = []
    if df.empty:
        return ["Dataset has no rows."]

    missing_pct = df.isna().mean()
    high_missing = missing_pct[missing_pct > 0.4]
    for col, pct in high_missing.items():
        issues.append(f"{col}: {pct:.0%} missing values.")

    constants = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    for col in constants:
        issues.append(f"{col}: constant or near-empty column.")

    high_card = []
    for col in df.select_dtypes(exclude=np.number).columns:
        nunique = df[col].nunique(dropna=True)
        if nunique > max(50, int(len(df) * 0.5)):
            high_card.append(col)
    for col in high_card:
        issues.append(f"{col}: high-cardinality categorical column ({df[col].nunique(dropna=True)} unique values).")

    if df.duplicated().sum():
        issues.append(f"{int(df.duplicated().sum())} duplicate rows detected.")

    return issues
