"""
models/classical_models.py
--------------------------
Classical machine-learning baseline models.

We provide three simple, well-known classifiers:

    * Logistic Regression - a linear model, fast and interpretable
    * Random Forest       - an ensemble of decision trees, robust
    * Support Vector Machine (SVM) - powerful non-linear classifier

Each model is trained only when the user clicks "Train", and the training is
cached in Streamlit's session state so it never runs twice for nothing.
"""

from __future__ import annotations

import time
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from utils import helpers


# Dictionary of available classical estimators.
CLASSIFIERS = {
    "Logistic Regression": lambda seed: LogisticRegression(
        max_iter=2000, random_state=seed
    ),
    "Random Forest": lambda seed: RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=seed
    ),
    "Support Vector Machine": lambda seed: SVC(
        probability=True, random_state=seed
    ),
}


def available_models() -> list[str]:
    """Names of the classical models the platform offers."""
    return list(CLASSIFIERS.keys())


def train_classical(model_name: str, X_train, y_train, random_state: int = 42) -> dict:
    """Train a classical model and return the fitted estimator + training time."""
    model = CLASSIFIERS[model_name](random_state)
    with warnings.catch_warnings():
        # scikit-learn >=1.9 deprecates SVC(probability=True) in favour of
        # CalibratedClassifierCV; it still works perfectly in current versions,
        # so we simply quiet the FutureWarning noise.
        warnings.filterwarnings("ignore", category=FutureWarning)
        _, elapsed = helpers.time_it(model.fit, X_train, y_train)
    return {"model": model, "training_time": elapsed, "model_name": model_name}


def predict_classical(pipeline: dict, X_test) -> dict:
    """Predict classes and (if possible) probabilities."""
    model = pipeline["model"]
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - start_pred
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    return {"y_pred": y_pred, "y_prob": y_prob, "prediction_time": pred_time}