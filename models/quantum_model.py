"""
models/quantum_model.py
-----------------------
Hybrid Quantum-Classical model using Qiskit.

This is the most interesting part of the project, so we keep it as simple as
possible while still being a real variational quantum classifier (VQC).

How a VQC works (very simplified):
    1. The classical computer prepares a small number of features.
    2. A *feature map* encodes those numbers into quantum states.
    3. A *parameterized ansatz* (trainable quantum circuit) transforms the state.
    4. Quantum measurements produce numbers that we treat as "predictions".
    5. A classical optimizer (COBYLA) adjusts the circuit parameters so the
       predictions match the true labels - exactly like classical training.

Everything runs locally on Qiskit's Aer simulator (no real hardware needed).
"""

from __future__ import annotations

import time

import numpy as np

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_aer.primitives import SamplerV2

from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import SPSA

from utils import helpers


def build_circuit(n_features: int, feature_reps: int = 1, ansatz_reps: int = 2):
    """Create the full quantum circuit (feature map + ansatz) for display."""
    feature_map = zz_feature_map(feature_dimension=n_features, reps=feature_reps)
    ansatz = real_amplitudes(n_features, reps=ansatz_reps)
    circuit = feature_map.compose(ansatz)
    return feature_map, ansatz, circuit


def create_sampler(shots: int = 512) -> SamplerV2:
    """Create a fast local statevector simulator sampler.

    Uses the exact statevector under the hood, which makes training fast and
    stable on a normal laptop for up to a few qubits.
    """
    return SamplerV2(
        default_shots=shots,
        options={
            "backend_options": {"method": "statevector", "device": "CPU"},
        },
    )


def train_quantum(
    X_train,
    y_train,
    feature_reps: int = 1,
    ansatz_reps: int = 2,
    maxiter: int = 50,
    random_state: int = 42,
) -> dict:
    """Train the hybrid VQC model on the (reduced, scaled) features.

    Uses SPSA, the standard noise-tolerant optimizer for variational quantum
    circuits. COBYLA and SPSA are both available; SPSA converges better here.
    """
    n_features = X_train.shape[1]
    if n_features > 6:
        raise ValueError(
            f"Too many features ({n_features}) for the quantum circuit. "
            "Please reduce features to 2-6 before training the quantum model."
        )

    np.random.seed(random_state)

    # Build the two parts of the quantum circuit.
    feature_map = zz_feature_map(feature_dimension=n_features, reps=feature_reps)
    ansatz = real_amplitudes(n_features, reps=ansatz_reps)

    sampler = create_sampler()

    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        sampler=sampler,
        optimizer=SPSA(maxiter=maxiter),
        callback=None,
    )

    start = time.perf_counter()
    vqc.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    _, _, circuit = build_circuit(n_features, feature_reps, ansatz_reps)

    return {
        "model": vqc,
        "training_time": training_time,
        "circuit": circuit,
        "feature_map": feature_map,
        "ansatz": ansatz,
        "n_qubits": n_features,
        "n_parameters": len(ansatz.parameters),
    }


def predict_quantum(pipeline: dict, X_test) -> dict:
    """Predict classes and probabilities with the trained VQC."""
    model = pipeline["model"]
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - start
    y_prob = model.predict_proba(X_test)[:, 1]
    return {"y_pred": y_pred, "y_prob": y_prob, "prediction_time": pred_time}