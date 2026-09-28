"""
visualization/plots.py
----------------------
Every chart used by the dashboard lives here as a small function that returns
a matplotlib Figure. Keeping them separate makes the app.py file clean and
each chart easy to explain.

Color scheme matches the app theme: teal = quantum, indigo/blue = classical.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: safe on Windows, works with Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Theme colors
TEAL = "#00D4AA"
INDIGO = "#7C5CFF"
BLUE = "#58A6FF"
CORAL = "#F47067"
YELLOW = "#F6C445"
GREEN = "#3FB950"

PALETTE = [TEAL, INDIGO, BLUE, CORAL, YELLOW, GREEN, "#DB61A2", "#8B949E"]

plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.titleweight"] = "bold"


def _base(figsize=(8, 4.5)):
    """Small helper that opens a figure and applies a clean grid style."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="none")
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color="#2a3654", alpha=0.35, lw=0.6)
    return fig, ax


# ---------------------------------------------------------------------------
# Dataset exploration charts
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, target_col: str) -> plt.Figure:
    """Bar + pie chart showing how many samples are in each class."""
    counts = df[target_col].value_counts().sort_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    fig.patch.set_facecolor("none")

    bars = ax1.bar(counts.index.astype(str), counts.values, color=PALETTE[: len(counts)], width=0.55)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val), ha="center", fontsize=10, fontweight="bold")
    ax1.set_title("Class distribution")
    ax1.set_ylabel("Number of samples")
    ax1.grid(axis="y", color="#2a3654", alpha=0.35, lw=0.6)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    ax2.pie(
        counts.values,
        labels=counts.index.astype(str),
        colors=PALETTE[: len(counts)],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "#E6EDF3"},
    )
    ax2.set_title("Class share")
    return fig


def plot_class_distribution_plotly(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Interactive Plotly chart of the class distribution (hover-friendly)."""
    counts = df[target_col].value_counts().sort_index().reset_index()
    counts.columns = [target_col, "count"]
    counts["share"] = (counts["count"] / counts["count"].sum() * 100).round(1)

    fig = px.bar(
        counts, x=target_col, y="count", color=target_col,
        text="count", color_discrete_sequence=PALETTE[: len(counts)],
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside", showlegend=False)
    fig.update_layout(
        xaxis_title="Class", yaxis_title="Number of samples",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#E6EDF3"),
        title="Interactive class distribution",
    )
    return fig


def plot_metric_comparison_plotly(comparison_df: pd.DataFrame, metric_labels: list[str]) -> go.Figure:
    """Interactive grouped bar chart for classical vs quantum metrics."""
    melted = pd.DataFrame(
        {
            "Metric": metric_labels * 2,
            "Score": list(comparison_df["classical"]) + list(comparison_df["quantum"]),
            "Model": ["Classical ML"] * len(metric_labels)
            + ["Hybrid Quantum ML"] * len(metric_labels),
        }
    )
    fig = px.bar(
        melted, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_map={"Classical ML": BLUE, "Hybrid Quantum ML": TEAL},
        text_auto=".2f",
    )
    fig.update_layout(
        yaxis_range=[0, 1.1], template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6EDF3"), title="Classical ML vs Hybrid Quantum ML (interactive)",
    )
    return fig


def plot_feature_distribution(X: pd.DataFrame, features: list[str], max_plots: int = 6) -> plt.Figure:
    """Histogram of each selected feature."""
    feats = features[:max_plots]
    n = len(feats)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows))
    fig.patch.set_facecolor("none")
    axes = np.atleast_1d(axes).ravel()

    for idx, feat in enumerate(feats):
        ax = axes[idx]
        ax.hist(X[feat].dropna(), bins=20, color=BLUE, edgecolor="none", alpha=0.9)
        ax.set_title(feat, fontsize=9)
        ax.grid(axis="y", color="#2a3654", alpha=0.3, lw=0.5)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap of all numeric columns."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor("none")
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdYlBu_r",
        center=0,
        annot=True if corr.shape[0] <= 10 else False,
        fmt=".2f",
        linewidths=0.4,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature correlation heatmap", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_pca_scatter(X2d: np.ndarray, y: np.ndarray, labels: list[str], col1: str, col2: str) -> plt.Figure:
    """2D scatter of the first two reduced features, coloured by class."""
    fig, ax = _base((7, 4.5))
    unique = np.unique(y)
    for klass, color in zip(unique, PALETTE[: len(unique)]):
        mask = y == klass
        ax.scatter(
            X2d[mask, 0], X2d[mask, 1], s=28, alpha=0.85,
            color=color, edgecolor="none", label=labels[int(klass)],
        )
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title("Reduced feature space (coloured by class)")
    ax.legend(frameon=False)
    return fig


# ---------------------------------------------------------------------------
# Model evaluation charts
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, class_names: list[str], title: str = "Confusion matrix") -> plt.Figure:
    """Heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    fig.patch.set_facecolor("none")

    max_val = cm.max()
    norm = cm / (cm.sum(axis=1, keepdims=True) if cm.sum(axis=1).max() > 0 else 1)
    sns.heatmap(
        norm,
        annot=cm,
        fmt="d",
        cmap=sns.light_palette(TEAL, as_cmap=True),
        cbar=False,
        xticklabels=[f"Pred {c}" for c in class_names],
        yticklabels=[f"True {c}" for c in class_names],
        linewidths=1, linecolor="#0b0f19", ax=ax, vmin=0, vmax=1,
    )
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_importance(importance_df: pd.DataFrame, title: str = "Feature importance") -> plt.Figure:
    """Horizontal bar chart of feature importance."""
    df = importance_df.sort_values("importance", ascending=True).head(10)
    fig, ax = _base((7.2, max(2.4, len(df) * 0.42)))
    ax.barh(df["feature"].astype(str), df["importance"], color=TEAL, height=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (accuracy drop when shuffled)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_importance_contribution(importance_df: pd.DataFrame, title: str = "Feature importance (top-K)") -> plt.Figure:
    """Bar chart that shows relative contribution % - clearer for demos."""
    df = importance_df.dropna().sort_values("importance", ascending=True).tail(8)
    total = df["importance"].sum()
    df = df.copy()
    df["pct"] = (df["importance"] / total * 100).round(1)
    fig, ax = _base((7.2, max(2.4, len(df) * 0.45)))
    colors = [TEAL if i == 1 else INDIGO for i in range(len(df), 0, -1)]
    ax.barh(df["feature"].astype(str), df["pct"], color=colors, height=0.6)
    for idx, (_, row) in enumerate(df.iterrows()):
        ax.text(row["pct"] + 0.8, idx, f"{row['pct']}%", va="center", fontsize=9, color="#E6EDF3")
    ax.invert_yaxis()
    ax.set_xlabel("Relative contribution (%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Classical vs Quantum comparison charts
# ---------------------------------------------------------------------------

def plot_metric_comparison(comparison_df: pd.DataFrame, metric_cols: list[str]) -> plt.Figure:
    """Grouped bar chart comparing classical vs quantum per metric."""
    fig, ax = _base((8, 4.2))
    x = np.arange(len(metric_cols))
    w = 0.36
    vals_cl = comparison_df["classical"].values
    vals_qm = comparison_df["quantum"].values

    ax.bar(x - w / 2, vals_cl, w, label="Classical ML", color=BLUE)
    ax.bar(x + w / 2, vals_qm, w, label="Hybrid Quantum ML", color=TEAL)

    for xi, v in zip(x - w / 2, vals_cl):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, vals_qm):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_cols, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Classical ML vs Hybrid Quantum ML")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_time_comparison(train_cl: float, pred_cl: float, train_qm: float, pred_qm: float) -> plt.Figure:
    """Bar chart comparing training and prediction time (seconds)."""
    fig, ax = _base((7, 3.8))
    categories = ["Training time", "Prediction time"]
    vals_cl = [train_cl, pred_cl]
    vals_qm = [train_qm, pred_qm]

    x = np.arange(len(categories))
    w = 0.36
    ax.bar(x - w / 2, vals_cl, w, label="Classical ML", color=BLUE)
    ax.bar(x + w / 2, vals_qm, w, label="Hybrid Quantum ML", color=TEAL)
    for xi, v in zip(x - w / 2, vals_cl):
        ax.text(xi, v + 0.01, f"{v:.2f}s", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, vals_qm):
        ax.text(xi, v + 0.01, f"{v:.2f}s", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Seconds")
    ax.set_title("Training / prediction time")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Quantum circuit drawing
# ---------------------------------------------------------------------------

def plot_quantum_circuit(circuit, n_qubits: int) -> plt.Figure:
    """Draw the quantum circuit used by the hybrid model."""
    try:
        from qiskit.visualization import circuit_drawer
        fig = circuit_drawer(circuit, output="mpl", style={"backgroundcolor": "#0b0f19", "textcolor": "#E6EDF3"})
        fig.set_size_inches(9, max(2.4, n_qubits * 0.8))
        return fig
    except Exception:  # noqa: BLE001 - fallback text draw
        fig, ax = _base((9, max(2.4, n_qubits * 0.8)))
        ax.text(0.5, 0.5, str(circuit), ha="center", va="center", family="monospace", fontsize=8, color="#E6EDF3")
        ax.axis("off")
        return fig