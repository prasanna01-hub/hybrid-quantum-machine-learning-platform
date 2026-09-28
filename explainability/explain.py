"""
explainability/explain.py
-------------------------
Simple Explainable AI (XAI) utilities.

We implement two explanation approaches:

    1. Permutation feature importance (model-agnostic, fast, reliable):
       - shuffle one feature at a time and measure how much the model's
         accuracy drops. A bigger drop = more important feature.

    2. SHAP values (used when a suitable explainer can be created):
       - classic method that tells us, for each prediction, how strongly each
         feature pushed the answer towards 'disease' or 'no disease'.

The results are turned into simple human-readable sentences such as:
    "Glucose has the strongest influence on the model prediction."

IMPORTANT: we only describe *which features mattered*, never medical claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Permutation importance (works for classical AND quantum models)
# ---------------------------------------------------------------------------

def permutation_importance(
    predict_fn,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute feature importance by shuffling each feature in turn."""
    rng = np.random.default_rng(random_state)
    baseline_pred = predict_fn(X_test)
    baseline_score = float((baseline_pred == y_test).mean())

    X = X_test.reset_index(drop=True)
    rows = []
    for feat in feature_names:
        scores = []
        original = X[feat].values.copy()
        for _ in range(n_repeats):
            X[feat] = rng.permutation(original)
            pred = predict_fn(X)
            scores.append(float((pred == y_test).mean()))
        X[feat] = original  # restore

        drop = baseline_score - np.mean(scores)
        rows.append({"feature": feat, "importance": drop})

    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    df["feature"] = df["feature"].astype(str)
    return df.reset_index(drop=True)


def importance_dataframe(importance: pd.DataFrame) -> pd.DataFrame:
    """Format the importance frame for display."""
    out = importance.copy()
    out["contribution"] = (out["importance"] / out["importance"].sum().clip(min=1e-9) * 100).round(1)
    out["contribution"] = out["contribution"].astype(str) + "%"
    return out.rename(columns={"importance": "importance (accuracy drop)"})


# ---------------------------------------------------------------------------
# SHAP (best effort - falls back gracefully)
# ---------------------------------------------------------------------------

def compute_shap_importances(model, X_train_sample: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame | None:
    """Return average absolute SHAP value per feature, or None on failure."""
    try:
        import shap  # imported lazily to keep startup fast

        if X_test.shape[0] < 5:
            return None

        if hasattr(model, "estimators_"):  # random forest / decision trees
            explainer = shap.TreeExplainer(model)
        elif str(type(model)).find("LinearRegression") >= 0 or str(type(model)).find("LogisticRegression") >= 0:
            explainer = shap.LinearExplainer(model, X_train_sample)
        else:
            # Generic model-agnostic explainer on a small background.
            explainer = shap.PermutationExplainer(
                model.predict, X_train_sample.sample(min(20, len(X_train_sample)), random_state=42)
            )

        shap_values = explainer.shap_values(X_test.head(50))
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]  # for multiclass take last class

        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
        out = pd.DataFrame({"feature": list(X_test.columns), "importance": mean_abs})
        return out.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:  # noqa: BLE001 - explainability must never crash the app
        return None


# ---------------------------------------------------------------------------
# Human-readable explanations
# ---------------------------------------------------------------------------

def explain_sentences(importance: pd.DataFrame, class_names: list[str], top_k: int = 3) -> list[str]:
    """Turn an importance table into simple sentences for non-experts."""
    if importance is None or importance.empty:
        return ["Not enough information to explain this model on this dataset."]

    top = importance.head(top_k)
    sentences = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        sentences.append(
            f"{row['feature']} had a strong influence on the model prediction "
            f"(rank #{rank})."
        )
    sentences.append(
        f"The model looks for patterns among these top features to separate "
        f"'{class_names[0]}' from '{class_names[1]}'. This is a research "
        "demonstration, not medical advice."
    )
    return sentences


def feature_direction_table(df: pd.DataFrame, pred_prob_target: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    """Show whether higher feature values push the prediction up or down.

    Computes a simple correlation between each feature and the predicted
    probability of the positive class.
    """
    rows = []
    for feat in feature_names:
        corr = df[feat].astype(float).corr(pred_prob_target.astype(float))
        if pd.isna(corr):
            corr = 0.0
        direction = (
            "Higher values increase predicted risk"
            if corr > 0.05
            else "Higher values decrease predicted risk" if corr < -0.05 else "Weak / mixed effect"
        )
        rows.append({"feature": feat, "correlation": round(float(corr), 3), "direction": direction})
    return pd.DataFrame(rows).sort_values("correlation", key=abs, ascending=False)