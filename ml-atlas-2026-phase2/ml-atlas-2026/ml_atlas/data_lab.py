from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TaskDetection:
    task: str
    confidence: str
    reason: str


def load_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload CSV or Excel.")


def detect_task(y: pd.Series) -> TaskDetection:
    clean = y.dropna()
    if clean.empty:
        return TaskDetection("classification", "low", "Target contains no usable values.")

    unique = int(clean.nunique())
    n = len(clean)

    if pd.api.types.is_bool_dtype(clean):
        return TaskDetection("classification", "high", "Boolean target detected.")

    if not pd.api.types.is_numeric_dtype(clean):
        return TaskDetection("classification", "high", "Non-numeric target detected.")

    # Numeric discrete targets are often labels, while continuous targets indicate regression.
    ratio = unique / max(n, 1)
    integer_like = np.all(np.isclose(clean.astype(float), np.round(clean.astype(float))))

    if unique <= 20 and integer_like:
        return TaskDetection(
            "classification",
            "high" if unique <= 10 else "medium",
            f"Numeric target has only {unique} discrete integer-like values."
        )

    if ratio <= 0.05 and unique <= 50:
        return TaskDetection(
            "classification",
            "medium",
            f"Target has relatively few unique values ({unique}/{n})."
        )

    return TaskDetection(
        "regression",
        "high",
        f"Numeric target appears continuous with {unique} unique values."
    )


def drop_identifier_like_columns(df: pd.DataFrame, target: str):
    removed = []
    keep = []
    for col in df.columns:
        if col == target:
            keep.append(col)
            continue
        nunique = df[col].nunique(dropna=True)
        ratio = nunique / max(len(df), 1)
        name = col.lower()
        identifier_name = (
            name == "id" or name.endswith("_id") or name.startswith("id_")
            or "uuid" in name or "guid" in name
        )
        if identifier_name and ratio > 0.8:
            removed.append(col)
        else:
            keep.append(col)
    return df[keep].copy(), removed
