"""
utils/helpers.py
----------------
Small reusable helper functions used across the whole application:

    * dataset loading / validation helpers
    * automatic target-column detection
    * categorical encoding helpers
    * timing / formatting helpers
    * safe parsing helpers (so user mistakes become friendly messages)

Everything here is kept simple on purpose so it is easy to read and explain.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Turn a number of seconds into a readable string like '4.21 s'."""
    if seconds < 60:
        return f"{seconds:.3f} s"
    return f"{seconds / 60:.2f} min"


def time_it(func, *args, **kwargs):
    """Run a function and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def load_csv(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV file into a pandas DataFrame.

    Raises a ValueError with a friendly message when the file is invalid,
    so the UI layer can show a clean error instead of a stack trace.
    """
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:  # noqa: BLE001 - we convert to a friendly message
        raise ValueError(
            "Could not read the uploaded file. Please make sure it is a valid "
            "CSV file (plain text, comma separated)."
        ) from exc

    if data.empty:
        raise ValueError("The uploaded CSV file contains no rows.")
    if data.shape[1] < 2:
        raise ValueError(
            "The CSV must contain at least one feature column and one target column."
        )
    return data


def parse_csv_text(text: str) -> pd.DataFrame:
    """Load a CSV from a raw string (used to embed the sample dataset)."""
    return pd.read_csv(io.StringIO(text))


# ---------------------------------------------------------------------------
# Target column detection
# ---------------------------------------------------------------------------

# Common names people use for the prediction/target column in medical datasets.
TARGET_HINTS = [
    "target", "outcome", "diagnosis", "class", "label", "disease",
    "diabetes", "cancer", "heart_disease", "result", "status", "y",
]


def suggest_target_column(df: pd.DataFrame) -> str | None:
    """Return the most likely target column name, or None if unsure."""
    cols = [str(c).strip().lower() for c in df.columns]

    for hint in TARGET_HINTS:
        if hint in cols:
            return str(df.columns[cols.index(hint)])

    # Guess: the last column is very often the class label in benchmark files.
    return str(df.columns[-1])


def encode_labels(y: pd.Series) -> tuple[np.ndarray, list[str], dict]:
    """Encode class labels to integers 0..k-1.

    Returns:
        (y_encoded, class_names, mapping) where mapping maps label -> integer.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    mapping = {label: int(code) for label, code in zip(le.classes_, le.transform(le.classes_))}
    return y_encoded, list(le.classes_), mapping


# ---------------------------------------------------------------------------
# Small safe helpers for the UI
# ---------------------------------------------------------------------------

def safe_float(value, default: float | None = None) -> float | None:
    """Convert a string/number to float, returning default when it fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_binary(df: pd.DataFrame, target_col: str) -> bool:
    """True when the target column has exactly two unique values."""
    return df[target_col].nunique(dropna=True) == 2