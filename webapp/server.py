"""
webapp/server.py
-----------------
FastAPI backend + static hosting for the QDETEX web dashboard.

Reuses the real, tested modules from the project core:
    preprocessing.run_pipeline / build_quantum_features
    models.classical_models   (train_classical / predict_classical)
    models.quantum_model      (train_quantum / predict_quantum)
    evaluation.metrics        (classification_metrics)
    explainability.explain    (permutation_importance / explain_sentences)

Every number returned to the frontend is computed from the real model on the
uploaded data. Nothing is hardcoded.
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project modules importable when running from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from evaluation.metrics import classification_metrics
from explainability import explain
from models import classical_models, quantum_model
from preprocessing import preprocessing as pp
from utils import helpers

warnings.filterwarnings("ignore")

WEBAPP_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = PROJECT_ROOT / "data" / "alzheimer_dataset.csv"

app = FastAPI(title="QDETEX API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

app.mount("/lib", StaticFiles(directory=str(WEBAPP_DIR / "lib")), name="lib")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(WEBAPP_DIR / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "qdetex"}


@app.get("/api/sample_csv")
def sample_csv():
    """Return the built-in Alzheimer's sample dataset as CSV."""
    if not SAMPLE_CSV.exists():
        return JSONResponse({"ok": False, "error": "Sample dataset not found."}, status_code=404)
    return FileResponse(str(SAMPLE_CSV), media_type="text/csv", filename="alzheimer_dataset.csv")


# ---------------------------------------------------------------------------
# Analysis core (shared by /api/sample and /api/analyze)
# ---------------------------------------------------------------------------

_MODELS = {
    "Logistic Regression": "Logistic Regression",
    "Random Forest": "Random Forest",
    "Support Vector Machine": "Support Vector Machine",
}

QCLASSIFIERS = {
    "Logistic Regression": classical_models.CLASSIFIERS["Logistic Regression"],
    "Random Forest": classical_models.CLASSIFIERS["Random Forest"],
    "Support Vector Machine": classical_models.CLASSIFIERS["Support Vector Machine"],
}

_QUANTUM_MAX_ITER = 30
_QUANTUM_SUBSAMPLE = 100


def _cm_list(cm: np.ndarray) -> list:
    if cm.shape == (1, 1):
        return [[int(cm[0, 0])]]
    return [[int(v) for v in row] for row in cm]


def _clean_metric(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            out[k] = _cm_list(v)
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        else:
            out[k] = round(float(v), 4) if isinstance(v, (float, int)) else v
    return out


def _sample_row_for_prediction(
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: list[str],
    classical_pred,
    classical_prob,
    quantum_pred,
    quantum_prob,
) -> dict:
    i = 0
    actual = class_names[int(y_test[i])]
    predicted_c = class_names[int(classical_pred[i])]
    prob_c = float(classical_prob[i]) if classical_prob is not None else None
    out = {
        "sample_index": int(i),
        "actual_label": actual,
        "classical_label": predicted_c,
        "classical_probability": round(prob_c * 100, 2) if prob_c is not None else None,
    }
    if quantum_pred is not None:
        out["quantum_label"] = class_names[int(quantum_pred[i])]
        out["quantum_probability"] = round(float(quantum_prob[i]) * 100, 2) if quantum_prob is not None else None
    return out


def _risk_analysis(
    artifacts: dict,
    importance_rows: list,
    sample_index: int = 0,
    prob_class1: float | None = None,
) -> dict | None:
    """Explain a single patient's risk using the real trained model.

    The risk score is the model's predicted probability of the positive
    class on that held-out test row. For each influential feature we compare
    the patient's (scaled) value against the normal/reference sub-population
    (the target==0 group seen during training) and report how far and in which
    direction the value deviates, plus whether that direction increases or
    decreases the measured risk.
    """
    if prob_class1 is None or artifacts.get("X_test") is None or artifacts["X_test"].shape[0] == 0:
        return None

    try:
        X_train = artifacts["X_train"]
        y_train = artifacts["y_train"]
        X_test = artifacts["X_test"]
        class_names = artifacts["class_names"]

        patient = X_test.iloc[sample_index]
        ref = X_train[y_train == 0]
        if ref.shape[0] < 5:
            ref = X_train  # not enough reference rows -> use the whole training set

        ref_mean = ref.mean(skipna=True)
        ref_std = ref.std(skipna=True)

        X_arr = X_train.to_numpy() if hasattr(X_train, "to_numpy") else np.asarray(X_train)
        cols = list(X_train.columns)
        direction = {}
        for i, c in enumerate(cols):
            v = X_arr[:, i].astype(float)
            sd = float(np.nanstd(v))
            if sd <= 0:
                direction[c] = 0.0
                continue
            try:
                c_ = float(np.corrcoef(v, y_train)[0, 1])
                direction[c] = c_ if np.isfinite(c_) else 0.0
            except Exception:  # noqa: BLE001
                direction[c] = 0.0

        imp = {row["feature"]: float(row.get("contribution", 0.0)) for row in importance_rows}
        factors = []
        for c in cols:
            val = patient[c]
            if val is None or not np.isfinite(float(val)):
                continue
            mean = float(ref_mean.get(c, np.nan))
            std = float(ref_std.get(c, np.nan))
            if not np.isfinite(mean) or std <= 0:
                continue
            zscore = (float(val) - mean) / std
            if abs(zscore) < 0.5:
                continue  # not meaningfully different from the normal reference
            high = float(val) > mean
            risky_is_high = direction.get(c, 0.0) > 0.05
            increases = (high == risky_is_high) or (abs(direction.get(c, 0.0)) <= 0.05 and high)
            factors.append(
                {
                    "feature": str(c),
                    "value": round(float(val), 2),
                    "reference_mean": round(mean, 2),
                    "deviation": round(zscore, 2),
                    "direction": "above" if high else "below",
                    "increases_risk": bool(increases),
                    "contribution": round(imp.get(c, 0.0), 1),
                }
            )

        factors.sort(
            key=lambda f: (
                not f["increases_risk"],
                -abs(f["deviation"]) * max(abs(f["contribution"]), 0.1),
            )
        )
        factors = factors[:5]

        score = round(float(prob_class1), 1)
        level = "High" if score >= 70 else "Moderate" if score >= 40 else "Low"

        return {
            "score": score,
            "probability": score,
            "level_assessment": level,
            "class_name": class_names[1] if len(class_names) > 1 else class_names[0],
            "positive_label": str(class_names[1]) if len(class_names) > 1 else None,
            "factors": factors,
            "note": (
                "Risk score = model probability of the positive class ("
                + (str(class_names[1]) if len(class_names) > 1 else str(class_names[0]))
                + ") for the first held-out test patient. Factors show each feature's "
                "deviation from the normal-reference population and whether it pushes "
                "the measured risk up or down."
            ),
        }
    except Exception:  # noqa: BLE001 - risk analysis must never take the app down
        return None


def _analyze(
    df: pd.DataFrame,
    dataset_name: str,
    target: str | None,
    model_name: str,
    include_quantum: bool,
    quantum_features: int,
    quantum_maxiter: int,
) -> dict:
    if model_name not in _MODELS:
        model_name = "Logistic Regression"
    if df.shape[1] < 2:
        raise ValueError("The CSV must contain at least one feature column and one target column.")

    col = helpers.suggest_target_column(df) if not target or target not in df.columns else target
    config = {
        "col": col,
        "model_name": model_name,
        "include_quantum": include_quantum,
        "quantum_features": int(quantum_features),
        "quantum_maxiter": int(quantum_maxiter),
    }

    artifacts = pp.run_pipeline(df, col)

    X_train, X_test = artifacts["X_train"], artifacts["X_test"]
    y_train, y_test = artifacts["y_train"], artifacts["y_test"]
    feature_names = artifacts["feature_names"]
    class_names = artifacts["class_names"]

    # ---- Classical --------------------------------------------------------
    trained = classical_models.train_classical(
        config["model_name"], X_train, y_train, random_state=42
    )
    cp = classical_models.predict_classical(trained, X_test)
    c_metrics = classification_metrics(y_test, cp["y_pred"], cp["y_prob"])

    # ---- Quantum (hybrid VQC) ---------------------------------------------
    q_metrics, q_info, q_pred = None, None, None
    quantum_error = None
    if config["include_quantum"]:
        try:
            qfeat = pp.build_quantum_features(
                X_train, X_test, y_train, method="topk", n_features=config["quantum_features"]
            )
            Xtr_q = qfeat["X_train"]
            Xte_q = qfeat["X_test"]
            n = min(len(Xtr_q), _QUANTUM_SUBSAMPLE)
            Xtr_qs, ytr_qs = Xtr_q[:n], y_train[:n]

            q_trained = quantum_model.train_quantum(
                Xtr_qs, ytr_qs, maxiter=config["quantum_maxiter"], random_state=42
            )
            qp = quantum_model.predict_quantum(q_trained, Xte_q)
            q_metrics = classification_metrics(y_test, qp["y_pred"], qp["y_prob"])
            q_pred = qp
            q_info = {
                "n_qubits": int(q_trained["n_qubits"]) if "n_qubits" in q_trained else int(Xte_q.shape[1]),
                "n_parameters": int(q_trained.get("n_parameters", 0)),
                "maxiter": config["quantum_maxiter"],
                "samples_used": int(n),
                "selected_features": [str(c) for c in qfeat["selected_cols"]],
                "method": qfeat["method"],
            }
        except Exception as exc:  # noqa: BLE001 - quantum must never take the app down
            quantum_error = "Hybrid quantum training could not complete on this dataset."
            q_metrics, q_info, q_pred = None, None, None

    # ---- Sample prediction -------------------------------------------------
    sample_pred = _sample_row_for_prediction(
        X_test, y_test, class_names,
        cp["y_pred"], cp["y_prob"],
        q_pred["y_pred"] if q_pred else None,
        q_pred["y_prob"] if q_pred else None,
    )

    # ---- Feature importance (classical model) ------------------------------
    importance_rows = []
    try:
        imp = explain.permutation_importance(
            lambda x: trained["model"].predict(x),
            X_test,
            y_test,
            feature_names,
            n_repeats=3,
            random_state=42,
        )
        total = max(sum(abs(v) for v in imp["importance"].fillna(0)), 1e-9)
        for _, row in imp.head(5).iterrows():
            importance_rows.append(
                {
                    "feature": str(row["feature"]),
                    "importance": round(float(row["importance"]), 4),
                    "contribution": round(abs(float(row["importance"])) / total * 100, 1),
                }
            )
    except Exception:  # noqa: BLE001 - explainability is best-effort
        importance_rows = []

    # ---- Patient risk detection (why is this patient at risk?) ------------
    prob_class1 = None
    if cp["y_prob"] is not None and len(cp["y_prob"]) > 0:
        prob_class1 = float(cp["y_prob"][0]) * 100.0
    risk = _risk_analysis(artifacts, importance_rows, sample_index=0, prob_class1=prob_class1)

    # ---- Winner banner -----------------------------------------------------
    winner = None
    if q_metrics is not None:
        ca, qa = c_metrics["accuracy"], q_metrics["accuracy"]
        name = "Classical ML" if ca >= qa else "Hybrid Quantum ML"
        winner = {"name": name, "margin": round(abs(ca - qa) * 100, 2)}

    class_counts = {str(k): int(v) for k, v in df[col].value_counts().items()}

    return {
        "ok": True,
        "dataset": {
            "name": dataset_name,
            "disease": "Alzheimer's (Dementia)" if "alz" in dataset_name.lower() else "General biomedical",
            "rows": int(df.shape[0]),
            "features": int(df.shape[1] - 1),
            "target": str(col),
            "class_names": class_names,
            "class_counts": class_counts,
            "processed_rows": int(artifacts["cleaned"].shape[0]),
            "numeric_features": feature_names,
        },
        "classical": {
            "model": model_name,
            "metrics": _clean_metric(c_metrics),
            "training_time": round(float(trained["training_time"]), 3),
            "prediction_time": round(float(cp["prediction_time"]) * 1000, 2),
            "n_features": int(len(feature_names)),
            "test_samples": int(len(y_test)),
        },
        "quantum": None
        if q_metrics is None
        else {
            "model": "Variational Quantum Classifier (VQC)",
            "metrics": _clean_metric(q_metrics),
            "training_time": round(float(q_trained["training_time"]), 3),
            "prediction_time": round(float(q_pred["prediction_time"]) * 1000, 2),
            "info": q_info,
        },
        "quantum_error": quantum_error,
        "prediction": sample_pred,
        "risk": risk,
        "importance": importance_rows,
        "winner": winner,
        "messages": {
            "positive_class": class_names[1],
            "quantum_features_note": (
                f"Reduced to {config['quantum_features']} features with mutual-information selection."
                if q_info
                else None
            ),
        },
    }


def _analyze_with_status(**kwargs) -> dict:
    try:
        return _analyze(**kwargs)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "code": "validation"}
    except Exception:  # noqa: BLE001
        return {
            "ok": False,
            "error": "Unable to analyze the dataset. Please check the file and try again.",
            "code": "processing",
        }


@app.get("/api/sample")
def api_sample():
    if not SAMPLE_CSV.exists():
        return JSONResponse({"ok": False, "error": "Sample dataset not found."}, status_code=404)
    df = helpers.load_csv(str(SAMPLE_CSV))
    return _analyze_with_status(
        df=df,
        dataset_name="alzheimer_dataset.csv",
        target=None,
        model_name="Logistic Regression",
        include_quantum=True,
        quantum_features=4,
        quantum_maxiter=_QUANTUM_MAX_ITER,
    )


@app.post("/api/analyze")
async def api_analyze(
    file: UploadFile | None = File(default=None),
    target: str | None = Form(default=None),
    model: str = Form(default="Logistic Regression"),
    include_quantum: str = Form(default="true"),
    quantum_features: int = Form(default=4),
    quantum_maxiter: int = Form(default=_QUANTUM_MAX_ITER),
):
    if file is not None and file.filename:
        try:
            raw = await file.read()
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:  # noqa: BLE001
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Could not read the uploaded file. Please make sure it is a valid CSV file.",
                    "code": "file",
                },
                status_code=400,
            )
        if df.empty:
            return JSONResponse({"ok": False, "error": "The uploaded CSV file contains no rows.", "code": "empty"}, status_code=400)
        name = file.filename or "uploaded"
    else:
        if not SAMPLE_CSV.exists():
            return JSONResponse({"ok": False, "error": "No file uploaded and sample data is missing.", "code": "missing"}, status_code=400)
        df = helpers.load_csv(str(SAMPLE_CSV))
        name = "alzheimer_dataset.csv"

    return _analyze_with_status(
        df=df,
        dataset_name=name,
        target=(target or None),
        model_name=model,
        include_quantum=str(include_quantum).lower() in ("1", "true", "yes", "on"),
        quantum_features=quantum_features,
        quantum_maxiter=quantum_maxiter,
    )


# Serve the rest of the frontend (style.css, script.js, style-extras.css) from
# the webapp directory. Because this mount is registered last, the API routes
# above always take precedence.
app.mount("/", StaticFiles(directory=str(WEBAPP_DIR), html=True), name="static-root")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")