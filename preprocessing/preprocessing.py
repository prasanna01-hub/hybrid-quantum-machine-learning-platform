"""
preprocessing/preprocessing.py
------------------------------
All data-cleaning and feature-preparation logic.

Steps implemented:
    1. validation of the uploaded dataset
    2. missing-value handling (drop rows or impute)
    3. duplicate removal
    4. categorical encoding (LabelEncoder for object columns)
    5. constant-column removal (zero-variance columns add no information)
    6. numerical scaling (StandardScaler)
    7. train/test split
    8. feature selection / dimensionality reduction for the quantum model

Every function returns plain data + a small "report" dictionary that the UI
uses to show before / after information to the user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

from utils import helpers


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame, target_col: str) -> list[str]:
    """Check the dataset and return a list of problems (empty list = OK)."""
    problems: list[str] = []

    if target_col not in df.columns:
        problems.append(f"The target column '{target_col}' was not found in the dataset.")

    if df.shape[0] < 20:
        problems.append("The dataset is very small. Results may be unreliable.")

    if df[target_col].nunique(dropna=True) < 2:
        problems.append("The target column must contain at least two classes.")
    elif df[target_col].nunique(dropna=True) != 2:
        problems.append(
            "This prototype supports binary (two-class) classification. "
            "Please pick a target column with exactly two classes."
        )

    # Features must be usable: we only keep numeric features and
    # auto-encode the rarely occurring categorical ones.
    return problems


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    target_col: str,
    drop_duplicates: bool = True,
    missing_strategy: str = "median",
) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned_df, report) after handling missing values + duplicates."""
    before = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }

    cleaned = df.copy()

    # Drop fully empty rows / columns.
    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = cleaned.dropna(axis=1, how="all")

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Handle missing values only for numeric columns (features/scaling).
    if missing_strategy == "drop":
        cleaned = cleaned.dropna().reset_index(drop=True)
    elif missing_strategy in ("mean", "median"):
        fill_num = cleaned.select_dtypes(include=[np.number]).columns
        if missing_strategy == "mean":
            cleaned[fill_num] = cleaned[fill_num].fillna(cleaned[fill_num].mean())
        else:
            cleaned[fill_num] = cleaned[fill_num].fillna(cleaned[fill_num].median())

    after = {
        "rows": int(cleaned.shape[0]),
        "cols": int(cleaned.shape[1]),
        "missing_cells": int(cleaned.isna().sum().sum()),
        "duplicates": int(cleaned.duplicated().sum()),
        "missing_strategy": missing_strategy,
    }
    return cleaned, {"before": before, "after": after}


# ---------------------------------------------------------------------------
# Feature / label preparation
# ---------------------------------------------------------------------------

def prepare_features_labels(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str], dict]:
    """Split the dataset into (X, y) with encoded labels.

    Returns:
        X (DataFrame, numeric features only),
        y (encoded numpy array 0..k-1),
        feature_names,
        class_names (original labels),
        label_mapping {original -> code}
    """
    X = df.drop(columns=[target_col])
    y_series = df[target_col].reset_index(drop=True)

    # Encode non-numeric feature columns (e.g. 'sex: male/female').
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype(str)
        X[col] = helpers.encode_labels(X[col])[0]

    # Keep only numeric features; report the rest so the UI can mention it.
    all_features = list(X.columns)
    X = X.select_dtypes(include=[np.number])
    dropped_non_numeric = [c for c in all_features if c not in X.columns]

    # Remove constant (zero variance) columns - they cannot help a model.
    constant_cols = [c for c in X.columns if X[c].std() == 0 or X[c].nunique() <= 1]
    X = X.drop(columns=constant_cols)

    # Encode labels.
    y, class_names, mapping = helpers.encode_labels(y_series)

    report = {
        "dropped_non_numeric_features": dropped_non_numeric,
        "constant_features_removed": constant_cols,
        "num_features": int(X.shape[1]),
    }
    return X, y, list(X.columns), class_names, mapping, report


# ---------------------------------------------------------------------------
# Scaling + splitting
# ---------------------------------------------------------------------------

def scale_and_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> dict:
    """Scale features and split into train/test. Returns a dict of artifacts."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=X.columns, index=X.index
    )

    strat_y = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=strat_y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "test_size": test_size,
    }


# ---------------------------------------------------------------------------
# Feature selection / dimensionality reduction (for quantum encoding)
# ---------------------------------------------------------------------------

def build_quantum_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    method: str = "topk",
    n_features: int = 4,
    random_state: int = 42,
) -> dict:
    """Reduce features to a small number (2-6) for the quantum circuit.

    Two methods are offered:
        * 'topk' - SelectKBest(mutual_info_classif): keeps the K most
                   informative original features (easy to explain).
        * 'pca'  - PCA: creates K new features as linear combinations of
                   all original features (best coverage of variance).

    Returns a dict with the reduced train/test arrays and human-readable notes.
    """
    n_features = max(2, min(int(n_features), X_train.shape[1]))

    if method == "topk":
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        Xtr_q = selector.fit_transform(X_train, y_train)
        Xte_q = selector.transform(X_test)
        scores = pd.Series(selector.scores_, index=X_train.columns).sort_values(
            ascending=False
        )
        selected_cols = list(scores.index[:n_features])
        note = (
            f"Kept the top {n_features} original features by mutual information "
            "(how much each feature relates to the target)."
        )
        pca_obj = None
        explained_variance = None
    else:  # PCA
        pca_obj = PCA(n_components=n_features, random_state=random_state)
        Xtr_q = pca_obj.fit_transform(X_train)
        Xte_q = pca_obj.transform(X_test)
        selected_cols = [f"PC{i + 1}" for i in range(n_features)]
        explained_variance = pca_obj.explained_variance_ratio_.sum()
        note = (
            f"Used PCA to compress all features into {n_features} new "
            f"components. They keep about {explained_variance:.1%} of the "
            "information (variance) in the data."
        )
        scores = None

    return {
        "X_train": Xtr_q,
        "X_test": Xte_q,
        "selected_cols": selected_cols,
        "method": method,
        "note": note,
        "scores": scores,
        "pca": pca_obj,
        "explained_variance": explained_variance,
    }


# ---------------------------------------------------------------------------
# One-stop preprocessing pipeline (used by tests and the app)
# ---------------------------------------------------------------------------

def run_pipeline(
    df: pd.DataFrame,
    target_col: str,
    drop_duplicates: bool = True,
    missing_strategy: str = "median",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> dict:
    """Run the full cleaning -> encoding -> scaling -> split pipeline."""
    problems = validate_dataset(df, target_col)
    if problems:
        raise ValueError(" | ".join(problems))

    cleaned, clean_report = clean_data(
        df, target_col, drop_duplicates=drop_duplicates, missing_strategy=missing_strategy
    )

    X, y, feature_names, class_names, mapping, prep_report = prepare_features_labels(
        cleaned, target_col
    )

    artifacts = scale_and_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    artifacts.update(
        {
            "cleaned": cleaned,
            "class_names": class_names,
            "label_mapping": mapping,
            "y": y,
            "clean_report": clean_report,
            "prep_report": prep_report,
            "target_col": target_col,
            "random_state": random_state,
        }
    )
    return artifacts