"""
app.py
------
Main Streamlit application for the

    "Hybrid Quantum Machine Learning Platform for Early Disease Detection"

Run it with:  streamlit run app.py

This file only *orchestrates* the UI. All the heavy logic lives in the
modules under preprocessing/, models/, evaluation/, explainability/,
visualization/ and utils/ so the whole project stays easy to read.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------------
from utils import helpers
from preprocessing import preprocessing as pp
from models import classical_models as cm
from models import quantum_model as qm
from evaluation import metrics as mtr
from explainability import explain as ex
from visualization import plots

# ---------------------------------------------------------------------------
# App configuration & theme helpers
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid QML • Early Disease Detection",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_dataset.csv"

NAV_PAGES = [
    "Overview & Dashboard",
    "Dataset & Preprocessing",
    "Feature Engineering",
    "Train & Compare Models",
    "Prediction",
    "Explainability & Quantum Circuit",
    "Model Information",
]


def _inject_style() -> None:
    """Very light CSS polish (rounded cards, hero gradient) for the dashboard."""
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(120deg, #0e1730 0%, #0f2438 55%, #12303a 100%);
            border: 1px solid #2a3654;
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 14px;
        }
        .hero h1 { color: #00D4AA; font-size: 1.9rem; font-weight: 700; margin: 0 0 6px 0; }
        .hero p  { color: #c6d0e0; margin: 2px 0; font-size: 0.98rem; }
        .hero .tag {
            display: inline-block; background: #00D4AA22; color:#00D4AA; border:1px solid #00D4AA55;
            border-radius: 999px; padding: 2px 12px; font-size: 0.75rem; margin-right: 6px;
        }
        .card {
            background: #141a2b; border: 1px solid #2a3654; border-radius: 12px;
            padding: 14px 16px; height: 100%;
        }
        .card .title { color: #58A6FF; font-weight: 600; font-size: 0.82rem; letter-spacing: .04em; }
        .card .body  { color: #E6EDF3; font-size: 0.9rem; line-height: 1.45; }
        .footer {
            color: #64748b; font-size: 0.78rem; margin-top: 30px; text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, tags: list[str]) -> None:
    """Render the gradient hero banner used at the top of key pages."""
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><div style="margin-top:8px">{tag_html}</div></div>',
        unsafe_allow_html=True,
    )


def disclaimer(message: str = "Research / educational prototype - NOT for medical diagnosis.") -> None:
    st.warning(message, icon=":material/verified_user:")


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
DERIVED_KEYS = [
    "prep", "quant", "classical_result", "quantum_result",
    "classical_metrics", "quantum_metrics", "comparison",
    "explain_state", "predict_state",
]


def reset_derived() -> None:
    """Clear everything that depends on the dataset/target choice."""
    for key in DERIVED_KEYS:
        st.session_state.pop(key, None)


def get_df() -> pd.DataFrame | None:
    return st.session_state.get("df")


# ---------------------------------------------------------------------------
# PAGE 1 - Overview & Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    hero(
        "Hybrid Quantum Machine Learning Platform",
        "Early disease detection by comparing Classical Machine Learning vs "
        "Hybrid Quantum-Classical Machine Learning on biomedical datasets.",
        ["Qiskit", "Streamlit", "Scikit-learn", "SHAP"],
    )

    col_badges = st.columns(3)
    col_badges[0].success(":material/check_circle: Fully local - no cloud / paid APIs")
    col_badges[1].info(":material/neurology: Works on the Qiskit Aer quantum simulator")
    col_badges[2].warning(":material/science: Research prototype - not clinical software")

    st.space("medium")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Classical models", "3", help="Logistic Regression, Random Forest, SVM")
    k2.metric("Quantum model", "VQC", help="Variational Quantum Classifier with Qiskit")

    # Dynamic accuracy cards (real values, ------- until models are trained).
    if st.session_state.get("classical_result"):
        k3.metric(
            "Classical accuracy",
            f"{st.session_state['classical_metrics']['accuracy']:.1%}",
            help=f"{st.session_state['classical_result']['model_name']} on hold-out test set",
        )
    else:
        k3.metric("Classical accuracy", "—", help="Train the classical model first")

    if st.session_state.get("quantum_result"):
        k4.metric(
            "Quantum accuracy",
            f"{st.session_state['quantum_metrics']['accuracy']:.1%}",
            help="Hybrid VQC on hold-out test set",
        )
    else:
        k4.metric("Quantum accuracy", "—", help="Train the quantum model first")

    st.space("medium")

    left, right = st.columns([3, 2])
    with left:
        st.subheader(":material/troubleshoot: Problem statement")
        st.write(
            "Many diseases are far easier to treat when detected early. Classic machine "
            "learning is powerful, but **quantum computers** can represent and explore "
            "feature combinations in a fundamentally different way. This platform lets you "
            "**compare both approaches side-by-side** on real biomedical data, so you can "
            "see (with actual measured results) whether a hybrid quantum-classical model can "
            "compete with, or even beat, a classical baseline."
        )

        st.subheader(":material/compare_arrows: Classical ML vs Quantum ML")
        c1, c2 = st.columns(2)
        c1.markdown(
            (
                "**Classical ML**  \n"
                "• Runs on ordinary computers (CPU)  \n"
                "• Mature, fast, well-understood  \n"
                "• Good baseline to compare against  \n"
                "• Used here: Logistic Regression, Random Forest, SVM"
            )
        )
        c2.markdown(
            (
                "**Hybrid Quantum ML**  \n"
                "• Encodes features into quantum states  \n"
                "• Learns a parameterized quantum circuit  \n"
                "• A classical optimizer does the training  \n"
                "• Used here: VQC on the Aer local simulator"
            )
        )

    with right:
        st.subheader(":material/account_tree: Workflow")
        steps = [
            ("Upload biomedical dataset", "dataset"),
            ("Preprocessing & validation", "auto_fix_high"),
            ("Feature selection / reduction", "filter_alt"),
            ("Train classical & quantum models", "model_training"),
            ("Compare performance honestly", "monitoring"),
            ("Explain the predictions", "lightbulb"),
        ]
        for text, icon in steps:
            st.markdown(f":material/{icon}: {text}")
        st.space("small")
        disclaimer()

    _build_quickstart_header()
    footer()


def _build_quickstart_header() -> None:
    st.space("medium")
    st.subheader(":material/rocket_launch: Quick start")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Load data**  \nGo to *Dataset & Preprocessing*, upload a CSV or use the bundled sample dataset, then pick the target column.")
    c2.markdown("**2 · Train models**  \nGo to *Train & Compare Models*, pick a classical model, train it, then train the hybrid quantum VQC.")
    c3.markdown("**3 · Explore**  \nCheck the comparison charts, make a prediction, and read the explainability sentences.")


def footer() -> None:
    st.markdown(
        '<div class="footer">Hybrid Quantum Machine Learning Platform for Early Disease Detection '
        "• Built with Streamlit + Qiskit • Educational project</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# PAGE 2 - Dataset & Preprocessing
# ---------------------------------------------------------------------------
def page_dataset() -> None:
    hero(
        "Dataset & Preprocessing",
        "Load a biomedical CSV (or use the bundled sample), inspect it, and let the "
        "platform prepare it for training.",
        ["CSV upload", "Auto-processing", "Validation"],
    )
    disclaimer()

    st.space("small")

    left, right = st.columns([2, 3])

    with left:
        st.subheader(":material/upload_file: Dataset source")
        source = st.radio(
            "Choose a source",
            ["Use sample dataset (diabetes)", "Upload my own CSV"],
            horizontal=True,
            key="ds_source",
            on_change=reset_derived,
        )

        if source == "Upload my own CSV":
            uploaded = st.file_uploader(
                "Drop your CSV file here",
                type=["csv"],
                key="ds_upload",
                on_change=reset_derived,
            )
            if uploaded is None:
                st.info("Waiting for a CSV file. While you wait, the sample dataset is shown.")
                df = _load_sample()
            else:
                try:
                    df = helpers.load_csv(uploaded)
                except ValueError as err:
                    st.error(str(err), icon=":material/error:")
                    st.stop()
        else:
            df = _load_sample()

    # Store the dataframe but only reset pipelines when it actually changes.
    if st.session_state.get("df_hash") != hash(pd.util.hash_pandas_object(df).sum()):
        st.session_state["df"] = df
        st.session_state["df_hash"] = hash(pd.util.hash_pandas_object(df).sum())
        reset_derived()

    df = st.session_state["df"]

    with right:
        st.subheader(":material/target: Target column")
        suggested = helpers.suggest_target_column(df)
        cols = list(df.columns)
        default_idx = cols.index(suggested) if suggested in cols else len(cols) - 1
        target_col = st.selectbox(
            "Which column contains the class label?",
            cols,
            index=default_idx,
            key="target_col",
            on_change=reset_derived,
        )

        problems = pp.validate_dataset(df, target_col)
        if problems:
            for p in problems:
                st.error(p, icon=":material/report:")
            st.stop()

        st.success(
            f"Target column: **{target_col}**  •  classes: {df[target_col].nunique()}  "
            f"•  {int(df.shape[0])} rows × {int(df.shape[1])} columns",
            icon=":material/check_circle:",
        )

    st.space("small")
    st.subheader(":material/table_view: Dataset overview")
    tab_overview, tab_stats, tab_classes = st.tabs(
        ["Preview & info", "Statistics", "Class distribution"]
    )

    with tab_overview:
        col_a, col_b = st.columns(2)
        col_a.metric("Rows", df.shape[0])
        col_b.metric("Columns", df.shape[1])
        st.markdown(f"**Feature names:** {', '.join(map(str, df.columns))}")
        missing = int(df.isna().sum().sum())
        dups = int(df.duplicated().sum())
        col_c, col_d = st.columns(2)
        col_c.metric("Missing cells", missing)
        col_d.metric("Duplicate rows", dups)
        st.dataframe(df.head(10), width="stretch", height=300)

    with tab_stats:
        st.dataframe(df.describe().T, width="stretch")

    with tab_classes:
        st.plotly_chart(plots.plot_class_distribution_plotly(df, target_col), width="stretch")
        st.pyplot(plots.plot_class_distribution(df, target_col))
        st.caption("Class distribution before any preprocessing.")

    st.space("small")

    # ------------------------- Preprocessing controls -----------------------
    st.subheader(":material/auto_fix: Preprocessing options")
    c1, c2, c3 = st.columns(3)
    missing_strategy = c1.radio(
        "Missing values",
        ["median", "mean", "drop"],
        help="Fill missing numeric values (median/mean) or drop rows containing them.",
        key="pp_missing",
        on_change=reset_derived,
    )
    drop_duplicates = c2.checkbox(
        "Remove duplicate rows", value=True, key="pp_dupes", on_change=reset_derived
    )
    stratify = c3.checkbox(
        "Stratified split (keep class balance)", value=True, key="pp_stratify", on_change=reset_derived
    )
    c_s, r_s = st.columns(2)
    test_size = c_s.slider("Test set size (%)", 10, 40, 20, key="pp_test", on_change=reset_derived) / 100
    seed = r_s.number_input("Random seed (reproducibility)", 0, 9999, 42, key="pp_seed", on_change=reset_derived)

    if st.button(":material/play_arrow: Run preprocessing", type="primary", width="stretch", key="btn_preprocess"):
        try:
            prep = pp.run_pipeline(
                df,
                target_col,
                drop_duplicates=drop_duplicates,
                missing_strategy=missing_strategy,
                test_size=test_size,
                random_state=int(seed),
                stratify=stratify,
            )
            st.session_state["prep"] = prep
            st.session_state.pop("quant", None)
            st.success("Preprocessing completed successfully!", icon=":material/check_circle:")
        except ValueError as err:
            st.error(str(err), icon=":material/error:")
            st.stop()

    prep = st.session_state.get("prep")
    if prep is None:
        st.info("Run preprocessing to continue. A report appears here.")
        return

    st.space("medium")
    st.subheader(":material/assignment: Processing report")
    b = prep["clean_report"]["before"]
    a = prep["clean_report"]["after"]
    cols = st.columns(4)
    cols[0].metric("Rows before → after", f"{b['rows']} → {a['rows']}")
    cols[1].metric("Duplicates removed", b["duplicates"])
    cols[2].metric("Missing cells after", a["missing_cells"])
    cols[3].metric("Ready features", len(prep["feature_names"]))

    with st.expander("What did the preprocessing do?"):
        st.markdown(
            f"- Handled missing values using **{missing_strategy}**.\n"
            f"- Removed {b['duplicates']} duplicate rows.\n"
            f"- Auto-encoded categorical features: {prep['prep_report']['dropped_non_numeric_features'] or 'none'}.\n"
            f"- Removed constant/zero-variance features: {prep['prep_report']['constant_features_removed'] or 'none'}.\n"
            f"- Scaled all numeric features with StandardScaler (mean 0, std 1).\n"
            f"- Split into train **{prep['X_train'].shape[0]}** / test **{prep['X_test'].shape[0]}** samples (stratified = {stratify}).",
        )
    st.caption(f"Class labels mapped as: {prep['label_mapping']}")


def _load_sample() -> pd.DataFrame:
    import pandas as pd

    return pd.read_csv(SAMPLE_CSV)


# ---------------------------------------------------------------------------
# PAGE 3 - Feature Engineering
# ---------------------------------------------------------------------------
def page_features() -> None:
    hero(
        "Feature Engineering for Quantum Encoding",
        "Quantum circuits work best with a small number of features. Here we select the "
        "most informative 2-6 features before passing data to the quantum circuit.",
        ["Feature selection", "PCA", "2-6 features"],
    )
    disclaimer()

    prep = st.session_state.get("prep")
    if prep is None:
        st.info(
            "You need a processed dataset first. Run **Dataset & Preprocessing** before "
            "using this page."
        )
        return

    st.markdown(
        ":material/lightbulb: **Why reduce features for the quantum model?**  \n"
        "In this prototype each feature needs its own qubit, and quantum circuits scale quickly "
        "in complexity with more qubits. So we compress hundreds (or dozens) of features into a "
        "few **2-6** highly relevant ones. This makes the VQC trainable on a normal laptop while "
        "still using real quantum circuits."
    )
    st.space("small")

    c1, c2, c3 = st.columns([1, 1, 1])
    method = c1.selectbox(
        "Reduction method",
        ["topk (most informative features)", "pca (principal components)"],
        key="feat_method",
        on_change=lambda: st.session_state.pop("quant", None),
    )
    max_q = min(6, len(prep["feature_names"]))
    n_quant = c2.slider(
        "Number of quantum features (qubits)",
        2, max_q, min(4, max_q),
        key="feat_n",
        on_change=lambda: st.session_state.pop("quant", None),
    )
    _ = c3.caption(f"Available features: {len(prep['feature_names'])}")

    if st.button(":material/build: Build quantum feature set", type="primary", width="stretch", key="btn_build_quant"):
        use_pca = method.startswith("pca")
        quant = pp.build_quantum_features(
            prep["X_train"], prep["X_test"], prep["y_train"],
            method="pca" if use_pca else "topk",
            n_features=int(n_quant),
            random_state=prep["random_state"],
        )
        st.session_state["quant"] = quant
        st.success("Quantum feature set ready!", icon=":material/check_circle:")

    quant = st.session_state.get("quant")
    if quant is None:
        st.info("Click **Build quantum feature set** to generate the reduced features.")
        return

    st.space("medium")
    st.subheader(":material/visibility: Reduced feature set")
    st.markdown(quant["note"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Selected features:** {', '.join(quant['selected_cols'])}")
        if quant["scores"] is not None:
            st.markdown("**Mutual-information scores (Top-K method):**")
            scores_df = quant["scores"].head(10).rename("score").to_frame()
            st.dataframe(scores_df, width="stretch")
    with col2:
        if quant["explained_variance"]:
            st.markdown(f"**Explained variance by PCA components:** {quant['explained_variance']:.1%}")

    # Scatter of reduced space
    st.subheader(":material/scatter_plot: Reduced feature space")
    labels = [str(c) for c in prep["class_names"]]
    if quant["method"] == "pca":
        fig = plots.plot_pca_scatter(
            np.asarray(quant["X_train"]), prep["y_train"], labels,
            quant["selected_cols"][0], quant["selected_cols"][1],
        )
        st.pyplot(fig)
    else:
        if quant["X_train"].shape[1] >= 2:
            fig = plots.plot_pca_scatter(
                np.asarray(quant["X_train"]), prep["y_train"], labels,
                quant["selected_cols"][0], quant["selected_cols"][1],
            )
            st.pyplot(fig)


# ---------------------------------------------------------------------------
# PAGE 4 - Train & Compare Models
# ---------------------------------------------------------------------------
def page_train() -> None:
    hero(
        "Train & Compare Models",
        "Train a classical ML baseline and a hybrid quantum-classical VQC on the same data, "
        "then compare their real measured performance.",
        ["Classical ML", "Hybrid VQC", "Honest comparison"],
    )
    disclaimer()

    prep = st.session_state.get("prep")
    if prep is None:
        st.info("Run **Dataset & Preprocessing** first.")
        return
    quant = st.session_state.get("quant")

    # ------------------------------------------------------------- Classical
    st.subheader(":material/window: 1 · Classical ML baseline")
    c1, c2, c3 = st.columns([2, 2, 1])
    model_name = c1.selectbox("Classical model", cm.available_models())
    c3.markdown("")  # spacing
    if st.button(":material/model_training: Train classical model", type="primary", width="stretch", key="btn_train_classical"):
        with st.spinner(f"Training {model_name}..."):
            try:
                res = cm.train_classical(model_name, prep["X_train"], prep["y_train"], prep["random_state"])
                preds = cm.predict_classical(res, prep["X_test"])
                metrics = mtr.classification_metrics(prep["y_test"], preds["y_pred"], preds["y_prob"])
                st.session_state["classical_result"] = {
                    **res, "preds": preds, "n_features": prep["X_train"].shape[1],
                }
                st.session_state["classical_metrics"] = metrics
                st.session_state["comparison"] = None
                st.success(f"{model_name} trained in {helpers.format_time(res['training_time'])}", icon=":material/check_circle:")
            except Exception as err:  # noqa: BLE001
                st.error(f"Classical training failed: {err}", icon=":material/error:")

    cr = st.session_state.get("classical_result")
    if cr:
        mets = st.session_state["classical_metrics"]
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{mets['accuracy']:.1%}")
        cols[1].metric("Precision", f"{mets['precision']:.1%}")
        cols[2].metric("Recall / Sensitivity", f"{mets['sensitivity']:.1%}")
        cols[3].metric("F1-score", f"{mets['f1']:.1%}")
        col4, col5 = st.columns(2)
        col4.metric("Specificity", f"{mets['specificity']:.1%}")
        col5.metric("Training time", helpers.format_time(cr["training_time"]))
        st.pyplot(
            plots.plot_confusion_matrix(
                mets["confusion_matrix"], prep["class_names"], title=f"Classical ({cr['model_name']})"
            )
        )

    st.space("medium")

    # -------------------------------------------------------------- Quantum
    st.subheader(":material/bolt: 2 · Hybrid Quantum-Classical model (VQC)")
    if quant is None:
        st.warning(
            "No quantum feature set yet. Build one on the **Feature Engineering** page, or "
            "use the defaults below.",
            icon=":material/warning:",
        )

    q1, q2, q3, q4 = st.columns([1, 1, 1, 1])
    maxiter = q1.number_input("Optimizer iterations (maxiter)", 10, 200, 50, key="vqc_maxiter")
    feats_reps = q2.selectbox("Feature-map repetitions", [1, 2], index=0, key="vqc_featreps")
    ans_reps = q3.selectbox("Ansatz layers", [1, 2], index=1, key="vqc_ansreps")
    # Subsample keeps the demo fast on laptops while still using real data.
    max_allow = min(len(prep["X_train"]), 150)
    q_sub = q4.number_input("Quantum training samples", 20, max_allow, min(80, max_allow), step=10, key="vqc_sub")

    qs_row = st.columns([2, 3])
    qs_row[1].caption(
        "For speed the VQC trains on a stratified subsample of the training data "
        "(a real approach when quantum circuits are expensive to evaluate)."
    )

    if st.button(":material/bolt: Train quantum model", type="primary", width="stretch", key="btn_train_quantum"):
        # Build a quantum feature set if the user did not.
        if quant is None:
            quant = pp.build_quantum_features(
                prep["X_train"], prep["X_test"], prep["y_train"],
                method="topk", n_features=min(4, len(prep["feature_names"])),
                random_state=prep["random_state"],
            )
            st.session_state["quant"] = quant

        # Stratified subsample for quantum training.
        from sklearn.model_selection import train_test_split as _t
        Xr, _, yr, _ = _t(
            quant["X_train"], prep["y_train"],
            train_size=int(q_sub), random_state=int(prep["random_state"]), stratify=prep["y_train"],
        )
        with st.spinner("Training the VQC on the local quantum simulator (this can take ~1 min)..."):
            try:
                qres = qm.train_quantum(
                    Xr, yr,
                    feature_reps=int(feats_reps),
                    ansatz_reps=int(ans_reps),
                    maxiter=int(maxiter),
                    random_state=int(prep["random_state"]),
                )
                preds = qm.predict_quantum(qres, quant["X_test"])
                metrics = mtr.classification_metrics(prep["y_test"], preds["y_pred"], preds["y_prob"])
                st.session_state["quantum_result"] = {**qres, "preds": preds}
                st.session_state["quantum_metrics"] = metrics
                st.session_state["comparison"] = None
                st.success(
                    f"VQC trained in {helpers.format_time(qres['training_time'])} "
                    f"({qres['n_qubits']} qubits, {qres['n_parameters']} trainable parameters)",
                    icon=":material/check_circle:",
                )
            except Exception as err:  # noqa: BLE001
                st.error(f"Quantum training failed: {err}", icon=":material/error:")

    qr = st.session_state.get("quantum_result")
    if qr:
        mets = st.session_state["quantum_metrics"]
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{mets['accuracy']:.1%}")
        cols[1].metric("Precision", f"{mets['precision']:.1%}")
        cols[2].metric("Recall / Sensitivity", f"{mets['sensitivity']:.1%}")
        cols[3].metric("F1-score", f"{mets['f1']:.1%}")
        col4, col5 = st.columns(2)
        col4.metric("Specificity", f"{mets['specificity']:.1%}")
        col5.metric("Training time", helpers.format_time(qr["training_time"]))

        st.markdown(
            f"**Circuit:** {qr['n_qubits']} qubits, {qr['n_parameters']} trainable parameters, "
            f"{feats_reps} feature repetitions, {ans_reps} ansatz layers."
        )
        st.pyplot(plots.plot_confusion_matrix(mets["confusion_matrix"], prep["class_names"], title="Hybrid Quantum VQC"))

    st.space("medium")

    # -------------------------------------------------------------- Comparison
    st.subheader(":material/compare_arrows: 3 · Classical vs Quantum comparison")
    _build_comparison(prep)


def _build_comparison(prep: dict) -> None:
    cr = st.session_state.get("classical_result")
    qr = st.session_state.get("quantum_result")
    if cr is None and qr is None:
        st.info("Train at least one model to see the comparison.")
        return

    rows = []
    if cr:
        rows.append(
            {
                "kind": "classical",
                "label": f"Classical ({cr['model_name']})",
                "metrics": st.session_state["classical_metrics"],
                "train_t": cr["training_time"],
                "pred_t": cr["preds"]["prediction_time"],
            }
        )
    if qr:
        rows.append(
            {
                "kind": "quantum",
                "label": "Hybrid Quantum VQC",
                "metrics": st.session_state["quantum_metrics"],
                "train_t": qr["training_time"],
                "pred_t": qr["preds"]["prediction_time"],
            }
        )

    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC"]

    # Build comparison frame for charts.
    comparison_df = pd.DataFrame(
        {
            "classical": [
                st.session_state.get("classical_metrics", {}).get(k, np.nan)
                for k in metric_keys
            ],
            "quantum": [
                st.session_state.get("quantum_metrics", {}).get(k, np.nan)
                for k in metric_keys
            ],
            "metric": metric_labels,
        }
    )
    st.session_state["comparison"] = comparison_df

    # Simple table
    display_rows = []
    for r in rows:
        m = r["metrics"]
        display_rows.append(
            {
                "Model": r["label"],
                "Accuracy": f"{m['accuracy']:.3f}",
                "Precision": f"{m['precision']:.3f}",
                "Recall (Sens.)": f"{m['recall']:.3f}",
                "Specificity": f"{m['specificity']:.3f}",
                "F1": f"{m['f1']:.3f}",
                "AUC": f"{m['auc']:.3f}",
                "Train time": helpers.format_time(r["train_t"]),
                "Predict time": helpers.format_time(r["pred_t"]),
            }
        )
    st.dataframe(pd.DataFrame(display_rows), width="stretch")

    if len(rows) == 2:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plots.plot_metric_comparison(comparison_df, metric_labels))
        with c2:
            st.pyplot(
                plots.plot_time_comparison(
                    train_cl=st.session_state["classical_result"]["training_time"],
                    pred_cl=st.session_state["classical_result"]["preds"]["prediction_time"],
                    train_qm=st.session_state["quantum_result"]["training_time"],
                    pred_qm=st.session_state["quantum_result"]["preds"]["prediction_time"],
                )
            )
        st.plotly_chart(plots.plot_metric_comparison_plotly(comparison_df, metric_labels), width="stretch")

        # Honest scientific verdict
        st.subheader(":material/leaderboard: Experimental result")
        acc_cl = st.session_state["classical_metrics"]["accuracy"]
        acc_qm = st.session_state["quantum_metrics"]["accuracy"]
        if acc_qm > acc_cl:
            st.success(
                f"On this dataset the **hybrid quantum model** reached {acc_qm:.1%} accuracy vs "
                f"{acc_cl:.1%} for classical ML. Results are dataset-specific - not a general claim.",
                icon=":material/emoji_objects:",
            )
        elif acc_cl > acc_qm:
            st.info(
                f"On this dataset **classical ML** reached {acc_cl:.1%} accuracy vs "
                f"{acc_qm:.1%} for the hybrid quantum model. This is a normal, scientifically "
                "honest outcome - hybrid QML does not always win.",
                icon=":material/emoji_objects:",
            )
        else:
            st.info(
                f"Both models reached the same accuracy ({acc_cl:.1%}) on this dataset.",
                icon=":material/emoji_objects:",
            )
    footer()


# ---------------------------------------------------------------------------
# PAGE 5 - Prediction
# ---------------------------------------------------------------------------
def page_prediction() -> None:
    hero(
        "Early Disease Detection - Prediction",
        "Enter feature values (or upload a test CSV) and get a prediction from the model of "
        "your choice, with confidence and the most influential features.",
        ["Prediction", "Confidence", "Explainability"],
    )
    disclaimer(
        "This prediction is for research/educational demonstration only and must "
        "not be used for medical diagnosis."
    )

    prep = st.session_state.get("prep")
    if prep is None:
        st.info("Run **Dataset & Preprocessing** first, then train models on **Train & Compare Models**.")
        return

    model_kind = st.radio(
        "Select the trained model",
        ["Classical ML", "Hybrid Quantum ML"],
        horizontal=True,
        key="pred_model_kind",
    )

    if model_kind == "Classical ML":
        if not st.session_state.get("classical_result"):
            st.warning("The classical model is not trained yet. Train it on **Train & Compare Models**.", icon=":material/warning:")
            return
        result = st.session_state["classical_result"]
    else:
        if not st.session_state.get("quantum_result"):
            st.warning("The quantum model is not trained yet. Train it on **Train & Compare Models**.", icon=":material/warning:")
            return
        if st.session_state.get("quant") is None:
            st.warning("Build the quantum feature set on **Feature Engineering** first.", icon=":material/warning:")
            return
        result = st.session_state["quantum_result"]

    # Context for predictions
    class_names = prep["class_names"]
    label_mapping = prep["label_mapping"]
    n_cls = len(class_names)

    tab_manual, tab_file = st.tabs([":material/tune: Enter values manually", ":material/upload_file: Upload a test CSV"])

    # -----------------------------------------------------------------
    with tab_manual:
        st.markdown("Move the sliders to describe a (simulated) patient record, then press **Predict**.")
        col_sliders = st.columns(3)
        values: dict[str, float] = {}

        # Use original (unscaled) ranges for intuitive sliders.
        orig = prep["cleaned"]
        for i, feat in enumerate(prep["feature_names"]):
            col = orig[feat]
            lo, hi = float(col.min()), float(col.max())
            mid = float(col.median())
            if hi == lo:
                hi = lo + 1.0
            with col_sliders[i % 3]:
                values[feat] = st.slider(
                    feat,
                    min_value=round(lo, 3),
                    max_value=round(hi, 3),
                    value=round(mid, 3),
                    key=f"slider_{feat}",
                )

        if st.button(":material/play_arrow: Predict", type="primary", key="btn_predict_manual"):
            preds, proba = _run_prediction(
                model_kind, pd.DataFrame([values], columns=prep["feature_names"]), prep
            )
            st.session_state["predict_state"] = {"kind": model_kind, "input": values, "preds": preds, "proba": proba}

    # -----------------------------------------------------------------
    with tab_file:
        st.markdown(
            "Upload a CSV containing the **same feature columns** (target column optional, it is ignored)."
        )
        test_file = st.file_uploader("Test CSV", type=["csv"], key="pred_upload")
        if test_file is not None:
            try:
                test_df = helpers.load_csv(test_file)
            except ValueError as err:
                st.error(str(err), icon=":material/error:")
                test_df = None
            if test_df is not None:
                missing_cols = [c for c in prep["feature_names"] if c not in test_df.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}", icon=":material/error:")
                else:
                    if st.button(":material/play_arrow: Predict batch", type="primary", key="btn_predict_file"):
                        try:
                            preds, proba = _run_prediction(
                                model_kind, test_df[prep["feature_names"]], prep
                            )
                            st.session_state["predict_state"] = {
                                "kind": model_kind, "input": None, "preds": preds,
                                "proba": proba, "df": test_df.reset_index(drop=True),
                            }
                        except Exception as err:  # noqa: BLE001
                            st.error(f"Prediction failed: {err}", icon=":material/error:")

    # -----------------------------------------------------------------
    st.space("small")
    pred_state = st.session_state.get("predict_state")
    if pred_state and pred_state.get("kind") == model_kind:
        preds = pred_state["preds"]
        proba = pred_state["proba"]

        if pred_state.get("input") is None:
            # Batch prediction table
            out = pred_state["df"].copy()
            out["Predicted class"] = [class_names[int(p)] for p in preds]
            if proba is not None:
                out["Confidence (positive class)"] = np.round(np.asarray(proba), 3)
            st.dataframe(out, width="stretch")
            st.markdown("*Summary of predictions:*")
            counts = pd.Series(
                [class_names[int(p)] for p in preds]
            ).value_counts().reset_index()
            counts.columns = ["Predicted class", "Count"]
            st.dataframe(counts, width="stretch")
            st.caption(f"Total test records: {len(preds)}")
        else:
            st.space("small")
            k1, k2 = st.columns(2)
            predicted_class = class_names[int(preds[0])]
            k1.metric("Predicted class", predicted_class)
            if proba is not None:
                conf = float(proba[0]) if n_cls == 2 else float(np.max(proba[0]))
                k2.metric("Confidence", f"{conf:.1%}")

            st.markdown("### Why this prediction?")
            _explain_single(pred_state["input"], predicted_class)

        disclaimer(
            "This prediction is for research/educational demonstration only and must "
            "not be used for medical diagnosis."
        )


def _run_prediction(model_kind: str, input_df: pd.DataFrame, prep: dict):
    """Apply scaling + reduction, then predict with the chosen model."""
    # Scale using the same scaler fitted on the training data.
    scaled = prep["scaler"].transform(input_df.astype(float))

    if model_kind == "Classical ML":
        result = st.session_state["classical_result"]
        preds = cm.predict_classical(result, scaled)
        y_pred, y_prob = preds["y_pred"], preds["y_prob"]
    else:
        # Quantum path
        quant = st.session_state["quant"]
        if quant["method"] == "pca":
            reduced = quant["pca"].transform(scaled)
        else:
            reduced = pd.DataFrame(scaled, columns=prep["feature_names"])[quant["selected_cols"]].values
        result = st.session_state["quantum_result"]
        preds = qm.predict_quantum(result, reduced)
        y_pred, y_prob = preds["y_pred"], preds["y_prob"]

    # VQC returns a 0-d array for a single sample - normalise to 1-d arrays.
    y_pred = np.atleast_1d(y_pred)
    y_prob = None if y_prob is None else np.atleast_1d(y_prob)
    return y_pred, y_prob


def _explain_single(input_values: dict, predicted_class: str) -> None:
    """Show which input features pushed the prediction (approximate)."""
    prep = st.session_state.get("prep")
    if prep is None:
        return
    df = prep["cleaned"]
    sentences = []
    for feat, val in input_values.items():
        mean = float(df[feat].mean())
        std = float(df[feat].std())
        if std == 0:
            continue
        z = (val - mean) / std
        impact = "strong" if abs(z) > 1.0 else "moderate" if abs(z) > 0.5 else "weak"
        direction = "above" if z > 0 else "below"
        sentences.append(
            f"{feat} was {impact} {direction} the dataset average "
            f"({val:.2f} vs mean {mean:.2f}), which influenced the prediction."
        )
    st.info(" ".join(sentences[:5]) or "No clear influential features could be identified.", icon=":material/lightbulb:")


# ---------------------------------------------------------------------------
# PAGE 6 - Explainability & Quantum Circuit
# ---------------------------------------------------------------------------
def page_explain() -> None:
    hero(
        "Explainability & Quantum Circuit",
        "See why the models predict what they predict - and inspect the quantum circuit "
        "that powers the hybrid model.",
        ["Permutation importance", "SHAP", "Circuit viewer"],
    )
    disclaimer()

    tab_exp, tab_circuit = st.tabs(
        [":material/lightbulb: Model explanations", ":material/memory: Quantum circuit"]
    )

    with tab_exp:
        _explainability_tab()

    with tab_circuit:
        _circuit_tab()


def _explainability_tab() -> None:
    prep = st.session_state.get("prep")
    cr = st.session_state.get("classical_result")
    qr = st.session_state.get("quantum_result")
    if prep is None or (cr is None and qr is None):
        st.info("Process the data and train at least one model first.")
        return

    st.markdown(
        "**How this works:** we shuffle one feature at a time and measure how much the "
        "model's accuracy drops. A big drop = an important feature. We also attempt a SHAP "
        "explanation when a suitable explainer exists for the model."
    )

    c1, c2 = st.columns([2, 1])
    which = c1.selectbox(
        "Explain which model?",
        ["Classical ML"] + (["Hybrid Quantum ML"] if qr else []),
        key="explain_which",
    )
    repeats = c2.number_input("Shuffle repeats", 2, 10, 4, key="explain_repeats")

    if st.button(":material/lightbulb: Generate explanation", type="primary", key="btn_explain"):
        if which.startswith("Classical"):
            feat_names = prep["feature_names"]
            X_test = prep["X_test"]
            predict_fn = st.session_state["classical_result"]["model"].predict
            kind = "classical"
        else:
            feat_names = st.session_state["quant"]["selected_cols"]
            X_test = pd.DataFrame(
                st.session_state["quant"]["X_test"], columns=feat_names
            )
            predict_fn = st.session_state["quantum_result"]["model"].predict
            kind = "quantum"

        with st.spinner("Computing permutation importance..."):
            imp = ex.permutation_importance(
                predict_fn, X_test, prep["y_test"],
                feature_names=feat_names, n_repeats=int(repeats),
                random_state=prep["random_state"],
            )
        shap_df = None
        if kind == "classical":
            with st.spinner("Attempting SHAP explanation (may take a moment)..."):
                sample = prep["X_train"].sample(min(30, len(prep["X_train"])), random_state=1)
                shap_df = ex.compute_shap_importances(
                    st.session_state["classical_result"]["model"], sample, X_test
                )

        st.session_state["explain_state"] = {"imp": imp, "shap": shap_df, "which": which}

    exp = st.session_state.get("explain_state")
    if not exp:
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Feature importance (accuracy drop)**")
        st.pyplot(plots.plot_importance(exp["imp"], title=f"Permutation importance - {exp['which']}"))
        st.markdown("**Explanation in plain language**")
        for s in ex.explain_sentences(exp["imp"], prep["class_names"]):
            st.markdown(f"- {s}")
    with col2:
        if exp.get("shap") is not None and not exp["shap"].empty:
            st.markdown("**SHAP mean absolute values**")
            st.pyplot(plots.plot_importance(exp["shap"], title="SHAP feature importance"))
        else:
            st.info("SHAP explanation not available for this model - using permutation importance instead.", icon=":material/info:")

    st.markdown("**Where do high/low values point?**")
    if exp["which"].startswith("Classical"):
        prob_col = st.session_state["classical_result"]["preds"]["y_prob"]
        pred_s = pd.Series(prob_col, index=prep["X_test"].index if hasattr(prep["X_test"], "index") else None)
        dir_df = ex.feature_direction_table(prep["X_test"], pred_s, prep["feature_names"])
        st.dataframe(dir_df.head(8), width="stretch")
        st.caption("Correlation between each feature and the predicted probability of the positive class.")


def _circuit_tab() -> None:
    qr = st.session_state.get("quantum_result")
    if qr is None:
        st.info("Train the quantum model to generate and display its circuit.")
        return

    nq = qr["n_qubits"]
    st.markdown(
        f"**Circuit summary**  \n"
        f"- Qubits: **{nq}** (one per selected feature)  \n"
        f"- Trainable parameters: **{qr['n_parameters']}**  \n"
        f"- Gates: feature-encoding rotations (RZ/H/CX from ZZFeatureMap) followed by "
        "entangling rotations (RX/RY + CX from RealAmplitudes)  \n"
        f"- Measurement: all qubits are measured at the end; the interpretation of the "
        "bitstrings is what the classical optimizer tunes."
    )

    st.pyplot(plots.plot_quantum_circuit(qr["circuit"], nq))

    with st.expander("What is this circuit actually doing? (simple language)"):
        st.markdown(
            """
            1. **Data encoding (ZZFeatureMap):** the reduced feature values are rotated into
               the phases of the qubits - this is how classical numbers become quantum states.
            2. **Learning (RealAmplitudes ansatz):** the circuit applies rotating and
               entangling gates whose *angles are trainable parameters*. The optimizer tweaks
               them during training, just like weights in a neural network.
            3. **Measurement:** each qubit is read out. The pattern of readouts (bitstring)
               is mapped to a class label.
            """
        )


# ---------------------------------------------------------------------------
# PAGE 7 - Model Information
# ---------------------------------------------------------------------------
def page_about() -> None:
    hero(
        "Model Information & Project Background",
        "Everything you need to understand and present this project.",
        ["Architecture", "Metrics", "Limitations"],
    )

    st.subheader(":material/category: Architecture")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        "**Classical ML**  \n"
        "- Runs on conventional computers  \n"
        "- Perfectly good baseline  \n"
        "- Handles preprocessing & features  "
    )
    c2.markdown(
        "**Quantum ML**  \n"
        "- Uses real quantum circuits  \n"
        "- Represents data as quantum states  \n"
        "- Tested here on the Aer **simulator**  "
    )
    c3.markdown(
        "**Hybrid approach**  \n"
        "- Classical: data preparation + optimization  \n"
        "- Quantum: encoding + variational circuit  \n"
        "- Best of both, feasible on today's hardware  "
    )

    st.subheader(":material/rule: Evaluation metrics used")
    st.markdown(
        "**Accuracy** = correct / total  •  **Precision** = TP / (TP+FP)  •  "
        "**Recall (Sensitivity)** = TP / (TP+FN)  •  **Specificity** = TN / (TN+FP)  •  "
        "**F1** = harmonic mean of precision & recall  •  **AUC** = area under the ROC curve."
    )

    st.subheader(":material/broken_image: Limitations")
    st.markdown(
        "- Educational prototype, **not** a medical device.\n"
        "- Runs on a quantum *simulator*, not real quantum hardware.\n"
        "- Quantum model uses only a few features and a subsample for speed.\n"
        "- Results depend strongly on the dataset - both models can win.\n"
        "- No clinical validation; predictions must not guide real decisions."
    )

    st.subheader(":material/rocket: Future enhancements")
    st.markdown(
        "- Run the VQC on near-term quantum hardware (IBM) when available.\n"
        "- Support multi-class and larger datasets.\n"
        "- Deeper ansatze, quantum kernel methods, and error mitigation.\n"
        "- Integration of genomics / imaging / EHR datasets."
    )

    # Deliverables table
    st.subheader(":material/inventory_2: Expected deliverables")
    deliverables = [
        ["Working Platform", "Streamlit-based hybrid QML disease detection platform"],
        ["Dataset Module", "CSV upload and sample dataset support"],
        ["Data Processing", "Cleaning, encoding, scaling and validation"],
        ["Feature Engineering", "Feature selection and dimensionality reduction"],
        ["Classical Models", "Logistic Regression, Random Forest and SVM"],
        ["Quantum Model", "Hybrid quantum-classical VQC using Qiskit simulator"],
        ["Prediction Module", "Disease/class prediction with probability where supported"],
        ["Evaluation Module", "Accuracy, Precision, Recall, F1, Sensitivity, Specificity"],
        ["Comparison Module", "Classical ML vs Hybrid QML comparison"],
        ["Explainability", "Feature importance / SHAP-based explanation"],
        ["Visualization", "Charts, confusion matrix and performance graphs"],
        ["Quantum Visualization", "Quantum circuit display"],
        ["Documentation", "README, installation and usage instructions"],
        ["Future Scope", "Near-term quantum hardware and larger biomedical datasets"],
    ]
    st.dataframe(
        pd.DataFrame(deliverables, columns=["Deliverable", "Description"]),
        width="stretch", hide_index=True,
    )
    footer()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
def sidebar() -> None:
    with st.sidebar:
        st.markdown("## :material/health_and_safety: Hybrid QML Platform")
        st.caption("Early disease detection • Classical vs Quantum ML")
        st.space("small")

        page = st.radio(
            "Menu",
            NAV_PAGES,
            label_visibility="collapsed",
            key="nav_page",
        )

        # Dataset status chip
        if st.session_state.get("df") is not None:
            st.badge("Dataset loaded", icon=":material/check:", color="green")
        else:
            st.badge("No dataset yet", icon=":material/pending:", color="orange")

        if st.session_state.get("quantum_result"):
            st.badge("Quantum model trained", icon=":material/bolt:", color="violet")
        elif st.session_state.get("classical_result"):
            st.badge("Classical model trained", icon=":material/model_training:", color="blue")

        st.space("small")
        if st.button(":material/refresh: Reset all results", width="stretch", key="btn_reset"):
            reset_derived()
            for k in ["df", "df_hash"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.space("medium")
        disclaimer(
            "Research / educational prototype. Not for medical diagnosis."
        )

    return page


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------
def main() -> None:
    _inject_style()
    page = sidebar()

    if page == NAV_PAGES[0]:
        page_dashboard()
    elif page == NAV_PAGES[1]:
        page_dataset()
    elif page == NAV_PAGES[2]:
        page_features()
    elif page == NAV_PAGES[3]:
        page_train()
    elif page == NAV_PAGES[4]:
        page_prediction()
    elif page == NAV_PAGES[5]:
        page_explain()
    else:
        page_about()


if __name__ == "__main__":
    main()