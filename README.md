# Hybrid Quantum Machine Learning Platform for Early Disease Detection

A simple, beginner-friendly **Streamlit web platform** that compares **Classical Machine
Learning** with a **Hybrid Quantum-Classical Machine Learning** model (Variational Quantum
Classifier, VQC) for **early disease detection** on real biomedical datasets.

> :warning: **Disclaimer — this is a research / educational prototype, NOT a medical diagnosis system.**
> It runs on a local quantum **simulator** (no real hardware, no paid APIs) and must never be
> used to guide real clinical decisions.

---

## 1. Problem statement

Many diseases are significantly easier to treat when detected early, but early detection usually
requires analysing many subtle biomedical measurements together. Classical ML already helps here.
**Can a hybrid quantum-classical model - which encodes data into quantum states - compete with or
even outperform classical baselines?** This platform lets you answer that question experimentally,
on your own dataset, instead of assuming quantum ML is "automatically better".

## 2. Background

* **Classical ML** (scikit-learn) is mature and powerful. We use Logistic Regression, Random
  Forest and SVM as strong baselines.
* **Quantum computing** represents information in quantum states, which can encode non-trivial
  correlations between features. A **Variational Quantum Classifier (VQC)** is a small parameterized
  quantum circuit whose parameters are tuned by a classical optimizer.
* **Hybrid QML** = classical data preprocessing + quantum feature encoding + variational quantum
  circuit + classical optimizer. This is exactly what runs in this project on the **Qiskit Aer
  simulator**.

## 3. Objectives

1. Design a hybrid quantum-classical architecture.
2. Process real/benchmark biomedical datasets.
3. Perform feature preprocessing and selection.
4. Implement classical ML baselines.
5. Implement a hybrid (quantum-enhanced) model with Qiskit.
6. Compare classical vs quantum approaches **experimentally** (no hardcoded accuracy).
7. Evaluate accuracy, precision, recall/sensitivity, specificity, F1 and AUC.
8. Provide explainability (feature importance + SHAP where possible).
9. Keep the architecture modular and scalable.
10. Stay compatible with quantum simulators and future quantum hardware.

## 4. Proposed solution

A Streamlit dashboard that guides the user through the full ML workflow:

```
User
 │
 ▼
Upload Biomedical Dataset
 │
 ▼
Data Preprocessing  (cleaning, encoding, scaling, split)
 │
 ▼
Feature Engineering  (feature selection / PCA → 2-6 features)
 │
 ▼
Train Classical ML Model   +   Train Hybrid Quantum-classical VQC
 │
 ▼
Early Disease Prediction   +   Performance Comparison
 │
 ▼
Explainability  +  Quantum Circuit Viewer  +  Report
```

## 5. Architecture

```
hybrid_qml_disease_detection/
│
├── app.py                          # Streamlit entry point (run this)
├── requirements.txt
├── README.md
├── .streamlit/config.toml          # Dark healthcare + quantum theme
├── data/
│   └── sample_dataset.csv          # Pima Indians Diabetes (768 rows)
├── models/
│   ├── classical_models.py         # Logistic Regression, Random Forest, SVM
│   └── quantum_model.py            # VQC (Qiskit) on the local Aer simulator
├── preprocessing/
│   └── preprocessing.py            # cleaning, encoding, scaling, split, reduction
├── evaluation/
│   └── metrics.py                  # accuracy, precision, recall, spec., F1, AUC
├── explainability/
│   └── explain.py                  # permutation importance + SHAP + plain language
├── visualization/
│   └── plots.py                    # all matplotlib charts + circuit drawing
└── utils/
    └── helpers.py                  # dataset loading, target detection, formatting
```

## 6. Technologies used

| Layer | Technology |
|-------|-----------|
| Frontend / dashboard | Python + **Streamlit** |
| Data handling | Pandas, NumPy |
| Classical ML | Scikit-learn |
| Quantum ML | **Qiskit**, **Qiskit-Machine-Learning**, **Qiskit-Aer** (local simulator) |
| Charts | Matplotlib, Seaborn, Plotly |
| Explainability | SHAP + permutation feature importance |
| APIs | none (100% local, no paid APIs) |

## 7. Installation

Requires **Python 3.10 – 3.12**.

```bash
# 1) (recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2) install dependencies
pip install -r requirements.txt

# 3) verify Qiskit works
python -c "import qiskit, qiskit_aer, qiskit_machine_learning; print('Qiskit OK')"
```

## 8. How to run

```bash
streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

**Quick test without any upload:** open the app → the bundled sample diabetes dataset is already
loaded → *Dataset & Preprocessing* → click **Run preprocessing** → *Feature Engineering* →
**Build quantum feature set** → *Train & Compare Models* → train one classical model, then train
the quantum VQC → explore the prediction, explainability and circuit pages.

## 9. Dataset format

Any **biomedical classification CSV** with:

* **Feature columns** (numeric; small numbers of string columns are auto-encoded)
* **One target/class column** (exactly **two classes**; binary classification is supported)
* No hidden NaNs that completely break the data (missing values are auto-handled)

The platform works with e.g. diabetes, cancer, cardiovascular, and neurological datasets.
The **target column is configurable** — never hardcoded. Example row (bundled sample):

```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
```

## 10. Classical ML workflow

1. Data cleaned + encoded + scaled (StandardScaler).
2. 80/20 stratified train/test split (configurable).
3. User picks **Logistic Regression**, **Random Forest** or **SVM**.
4. Model trained on the processed data.
5. Test-set metrics computed and displayed with confusion matrix.

## 11. Quantum ML workflow (Hybrid VQC)

Why feature reduction? A quantum circuit uses **one qubit per feature**. Hundreds of raw features
would need far too many qubits for a classroom prototype, so we first reduce to **2–6 informative
features** (Top-K mutual information, or PCA).

Hyperparameters you can tune: feature-map repetitions, ansatz layers, optimizer iterations
(maxiter), and number of training samples (subsampling keeps the demo fast on laptops).

## 12. Evaluation metrics

| Metric | Meaning |
|--------|---------|
| Accuracy | fraction of correct predictions |
| Precision | of positive predictions, how many were right |
| Recall (Sensitivity) | of actual positives, how many we caught |
| Specificity | of actual negatives, how many we correctly ruled out |
| F1-score | harmonic mean of precision & recall |
| AUC | area under the ROC curve |

All values are computed from the **real predictions** of the trained models. Nothing is hardcoded —
the dashboard shows a dynamic comparison and an honest "which model won" verdict.

## 13. Explainability

* **Permutation feature importance** (works for classical and quantum models): shuffle one feature
  at a time and measure the accuracy drop.
* **SHAP** values when a suitable explainer exists (tree/linear/generic).
* Plain-language sentences: *"Glucose had a strong influence on the model prediction (rank #1)."*
* Input screens show which features were above/below the dataset average.

No medical claims are generated — only descriptions of behaviour.

## 14. Exception / error handling

* Missing target column → clear message with auto-detection
* Non-numeric data → auto-encoded or clearly reported
* Missing values → handled automatically (median/mean/drop)
* Too many features for the quantum circuit → blocked with a friendly note
* Invalid CSV → friendly error instead of a stack trace
* Qiskit compatibility problems → the implementation adapts to the installed Qiskit 2.x API
* Small datasets / training errors → user-friendly messages

## 15. Limitations

* Educational prototype — no clinical validation.
* Runs on a **simulator**, not real quantum hardware (noise, decoherence ignored).
* Quantum model uses only 2–6 features and a subsample for speed.
* Results are dataset-specific; classical ML can (and often does) outperform the VQC.
* Binary classification only in this version.

## 16. Future enhancements

* Run on near-term quantum hardware (e.g., IBM) when available.
* Multi-class support and larger biomedical datasets (genomics, imaging, EHR).
* Deeper ansatze, quantum kernels, error mitigation.
* More datasets + auto benchmark suite.

## 17. Expected deliverables

| Deliverable | Description |
|-------------|-------------|
| Working Platform | Streamlit-based hybrid QML disease detection platform |
| Dataset Module | CSV upload and sample dataset support |
| Data Processing | Cleaning, encoding, scaling and validation |
| Feature Engineering | Feature selection and dimensionality reduction |
| Classical Models | Logistic Regression, Random Forest and SVM |
| Quantum Model | Hybrid quantum-classical VQC using Qiskit simulator |
| Prediction Module | Disease/class prediction with probability where supported |
| Evaluation Module | Accuracy, Precision, Recall, F1, Sensitivity, Specificity |
| Comparison Module | Classical ML vs Hybrid QML comparison |
| Explainability | Feature importance / SHAP-based explanation |
| Visualization | Charts, confusion matrix and performance graphs |
| Quantum Visualization | Quantum circuit display |
| Documentation | README, installation and usage instructions |
| Future Scope | Near-term quantum hardware and larger biomedical datasets |

---

**Made for education and research.** Always verify with a real medical professional.