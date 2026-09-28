/* ============================================================
   QDETEX frontend — drives the real FastAPI backend.
   All numbers come from the server (/api/sample and /api/analyze).
   ============================================================ */

const sampleCSV = `age,bmi,blood_pressure,glucose,cholesterol,heart_rate,target
45,27.4,128,112,198,76,0
52,31.2,142,156,235,84,1
38,24.8,118,98,181,70,0
61,29.8,150,174,252,91,1
47,26.1,124,108,205,73,0
56,33.5,146,165,244,88,1
34,23.2,116,92,176,68,0
67,30.1,154,181,267,94,1
42,25.7,122,104,193,72,0
59,32.4,148,159,249,86,1
49,28.6,130,121,211,78,0
63,34.1,158,188,276,97,1
36,22.9,114,89,169,66,0
54,30.8,140,151,238,83,1
44,26.8,126,109,201,74,0
69,35.2,162,195,281,99,1
40,24.1,120,101,187,69,0
58,31.7,144,162,246,87,1
51,27.9,132,128,216,79,0
65,33.2,156,179,263,93,1
37,23.8,117,95,178,67,0
60,29.4,149,170,254,90,1
46,26.4,125,106,199,73,0
55,32.1,143,157,241,85,1
43,25.2,121,102,190,71,0
62,30.6,151,176,258,92,1
35,24.6,119,97,183,68,0
57,31.3,145,164,247,88,1
48,27.1,129,115,207,76,0
66,34.4,159,186,272,96,1
39,23.7,116,94,175,67,0
53,30.2,141,153,236,84,1
41,25.9,123,107,196,72,0
64,33.7,155,182,269,94,1
50,28.3,131,119,213,78,0
68,35.1,161,192,279,98,1
33,22.8,113,90,171,65,0
56,30.9,147,160,243,86,1
45,26.7,127,111,202,74,0
61,32.8,153,175,261,92,1
47,27.5,128,114,209,77,0
59,31.9,146,168,250,89,1
37,24.3,118,96,180,68,0
63,34,157,184,270,95,1
44,25.6,124,103,191,71,0
55,30.7,142,155,239,84,1
42,26,122,105,197,72,0
67,33.8,160,180,266,94,1
36,23.4,115,91,174,66,0
52,29.9,139,149,232,82,1
49,27.7,130,120,215,77,0
65,34.6,158,187,274,97,1
40,24.9,119,99,185,69,0
58,32.3,147,161,245,87,1
46,26.5,126,110,204,74,0
62,31.1,152,172,257,91,1
34,23.1,114,93,172,66,0
54,30.4,143,154,240,85,1
43,25.4,121,100,189,70,0
60,32,149,169,253,90,1`;

let charts = {};
let uploadedFile = null;

const state = {
  result: null,
  raw: Papa.parse(sampleCSV.trim(), { header: true, dynamicTyping: true }).data,
  target: null,
  source: "sample",
  fileName: "sample_dataset.csv",
  predMode: "classical",
};

function isAlzFocus() {
  const r = state.result;
  if (r && r.dataset) {
    return /alz/i.test(String(r.dataset.disease || "") + " " + String(r.dataset.name || ""));
  }
  return /alz/i.test(String(state.fileName || ""));
}
function prettyClass(v) {
  if (!isAlzFocus()) return String(v);
  const n = Number(v);
  if (Number.isNaN(n)) return String(v) + " (unknown)";
  return n === 1 ? "Alzheimer's (Dementia)" : n === 0 ? "No Alzheimer's (Normal)" : String(v);
}

/* ------------------------------------------------------------------ */
/* Navigation                                                          */
/* ------------------------------------------------------------------ */

function go(page) {
  document.querySelectorAll(".page").forEach((x) => x.classList.remove("active"));
  document.getElementById(page).classList.add("active");
  document.querySelectorAll(".nav").forEach((x) =>
    x.classList.toggle("active", x.dataset.page === page)
  );
  const labelMap = {
    results: "Results & Accuracy", dataset: "Dataset", preprocess: "Preprocessing",
    features: "Feature Engineering", classical: "Classical ML", quantum: "Hybrid Quantum ML",
    prediction: "Prediction", comparison: "Performance", explain: "Explainability",
    visuals: "Analytics", circuit: "Quantum Method",
  };
  document.getElementById("title").textContent =
    page === "dashboard"
      ? "QDETEX — Hybrid Quantum Machine Learning Platform"
      : "QDETEX • " + (labelMap[page] || page).toUpperCase();
  if (page === "dataset") renderDatasetPage();
  if (page === "features") renderFeatures();
  if (page === "prediction") renderPrediction();
  if (page === "visuals") renderVisuals();
  if (page === "explain") renderImportance();
}
document.querySelectorAll(".nav").forEach((x) => (x.onclick = () => go(x.dataset.page)));
window.go = go;

/* ------------------------------------------------------------------ */
/* Format helpers                                                      */
/* ------------------------------------------------------------------ */

const MISS = "\u2014";
const pct = (v) => (v == null || Number.isNaN(v) ? MISS : (Number(v) * 100).toFixed(1) + "%");
function fmtTimeSec(s) {
  if (s == null) return MISS;
  return s < 1 ? (s * 1000).toFixed(0) + " ms" : s.toFixed(2) + " s";
}
function numCols(rows, target) {
  const first = rows[0] || {};
  return Object.keys(first).filter(
    (k) => k !== target && rows.every((r) => typeof r[k] === "number" && !Number.isNaN(r[k]))
  );
}

/* ------------------------------------------------------------------ */
/* Loading overlay with staged steps                                   */
/* ------------------------------------------------------------------ */

function showLoader() {
  document.getElementById("loader").classList.remove("hidden");
  ["st1", "st2", "st3", "st4", "st5"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = "";
    el.textContent = el.textContent.replace(/^[✓✗] /, "");
  });
  window.__loaders = [];
  const plan = [["st1", 850], ["st2", 1600], ["st3", 2100], ["st5", 500]];
  plan.forEach(([id, dur]) =>
    window.__loaders.push(
      setTimeout(() => {
        const el = document.getElementById(id);
        if (el && el.id !== "st4") el.className = "done";
      }, dur)
    )
  );
  const st = document.getElementById("st4");
  if (st) {
    st.classList.add("active");
    st.textContent = "Running hybrid quantum-classical model (takes up to a minute)";
  }
}
function hideLoader() {
  document.getElementById("loader").classList.add("hidden");
  (window.__loaders || []).forEach(clearTimeout);
}

/* ------------------------------------------------------------------ */
/* API calls                                                           */
/* ------------------------------------------------------------------ */

async function postAnalyze(formData) {
  const res = await fetch("/api/analyze", { method: "POST", body: formData });
  let json = null;
  try { json = await res.json(); } catch (e) { json = null; }
  if (!res.ok) {
    throw new Error((json && json.error) || "The server could not complete the analysis.");
  }
  if (!json.ok) {
    throw new Error(json.error || "Unable to analyze the dataset. Please check the file and try again.");
  }
  return json;
}

function showErrorBox(text) {
  document.getElementById("uploadErrorText").textContent =
    text || "Unable to analyze the dataset. Please check the file and try again.";
  document.getElementById("uploadError").classList.remove("hidden");
}
function hideUploadError() {
  document.getElementById("uploadError").classList.add("hidden");
}

/* ------------------------------------------------------------------ */
/* Analysis actions                                                    */
/* ------------------------------------------------------------------ */

async function runAnalysis({ file, target, model, qFeatures }) {
  hideUploadError();
  showLoader();
  try {
    const fd = new FormData();
    if (file) fd.append("file", file, file.name);
    if (target) fd.append("target", target);
    fd.append("model", model || "Logistic Regression");
    fd.append("include_quantum", "true");
    fd.append("quantum_features", qFeatures || 4);
    const result = await postAnalyze(fd);
    const st = document.getElementById("st5");
    if (st) { st.className = "done"; st.textContent = "✓ Generating results"; }
    applyResult(result);
    hideLoader();
    go("results");
    window.scrollTo(0, 0);
  } catch (err) {
    hideLoader();
    showErrorBox(err.message);
    window.scrollTo(0, 0);
    go("dataset");
  }
}

function analyzeUpload() {
  if (!uploadedFile) {
    showErrorBox("Please choose a CSV file first (or click \u201CRun on Sample Dataset\u201D).");
    return;
  }
  runAnalysis({
    file: uploadedFile,
    target: state.target || undefined,
    model: document.getElementById("modelSelect").value,
    qFeatures: +document.getElementById("uploadFeatureCount").value,
  });
}

function analyzeSample() {
  const sel = document.getElementById("modelSelect");
  runAnalysis({
    file: null,
    target: undefined,
    model: sel ? sel.value : "Logistic Regression",
    qFeatures: 4,
  });
}

function reanalyze() {
  hideUploadError();
  showLoader();
  const model = document.getElementById("classicalModel").value;
  const fd = new FormData();
  if (state.source === "upload" && uploadedFile) {
    fd.append("file", uploadedFile, uploadedFile.name);
    fd.append("target", state.target || "");
  }
  fd.append("model", model);
  fd.append("include_quantum", "true");
  fd.append("quantum_features", "4");
  postAnalyze(fd)
    .then((result) => {
      applyResult(result);
      hideLoader();
      go("classical");
    })
    .catch((err) => {
      hideLoader();
      const box = document.getElementById("classicalError");
      box.textContent = err.message || "Re-training failed.";
      box.classList.remove("hidden");
    });
}
/* ------------------------------------------------------------------ */
/* Applying a real backend result to the whole UI                      */
/* ------------------------------------------------------------------ */

function applyResult(result) {
  state.result = result;
  state.source = uploadedFile ? "upload" : "sample";

  const c = result.classical || {};
  const q = result.quantum;
  const d = result.dataset || {};

  document.getElementById("dashDataset").textContent = d.name || "Dataset loaded";
  document.getElementById("fcRows").textContent = d.rows != null ? d.rows + " ROWS" : "READY";
  document.getElementById("fcModel").textContent = "LIVE";
  document.getElementById("snapRows").textContent = (d.rows != null ? d.rows : MISS) + " × " + (d.features != null ? d.features : MISS);
  document.getElementById("snapTarget").textContent = d.target || MISS;
  document.getElementById("snapFeatures").textContent = q && q.info ? q.info.n_qubits : "4";
  document.getElementById("snapStatus").textContent = "Complete";
  document.getElementById("wfData").textContent =
    (d.rows != null ? d.rows : MISS) + " rows · " + (d.features != null ? d.features : MISS) + " features";
  setDone("wfDataIco");
  document.getElementById("wfPrep").textContent = "Preprocessed · 80/20 split";
  setDone("wfPrepIco");
  document.getElementById("wfClass").textContent = "Trained • " + pct(c.metrics && c.metrics.accuracy);
  setDone("wfClassIco");
  if (q && q.metrics) {
    document.getElementById("wfQuantum").textContent = "Complete • " + pct(q.metrics.accuracy);
    setDone("wfQuantumIco");
  } else {
    document.getElementById("wfQuantum").textContent = result.quantum_error || "Not available";
  }
  document.getElementById("wfResults").textContent = "Complete — see results";
  setDone("wfResultsIco");

  document.getElementById("dashClassAcc").textContent = pct(c.metrics && c.metrics.accuracy);
  document.getElementById("dashClassModel").textContent = c.model || "Accuracy";
  document.getElementById("dashQuantumAcc").textContent = pct(q && q.metrics && q.metrics.accuracy);
  if (result.winner && result.winner.name) {
    document.getElementById("winnerText").textContent =
      result.winner.name + " leads this experiment by " + result.winner.margin.toFixed(1) + " percentage points.";
    document.getElementById("winnerBadge").textContent = result.winner.name.toUpperCase();
  } else {
    document.getElementById("winnerText").textContent =
      "Classical model measured " + pct(c.metrics && c.metrics.accuracy) + " accuracy.";
    document.getElementById("winnerBadge").textContent = "CLASSICAL";
  }

  renderResults();
  renderRisk();
  renderDatasetPage();
  renderPreprocess();
  renderClassical();
  renderQuantum();
  renderPrediction();
  renderComparison();
  renderImportance();
  renderFeatures();
}

function setDone(id) {
  const el = document.getElementById(id);
  if (el) { el.textContent = "\u2713"; el.style.color = "#40e2a0"; }
}

/* ------------------------------------------------------------------ */
/* RESULTS / ACCURACY PAGE                                             */
/* ------------------------------------------------------------------ */

const CIRC = 2 * Math.PI * 84;

function renderResults() {
  const r = state.result;
  if (!r) return;
  const c = r.classical, q = r.quantum, d = r.dataset || {};
  const accC = c.metrics ? c.metrics.accuracy : null;
  const accQ = q && q.metrics ? q.metrics.accuracy : null;
  const acc = accC != null && accQ != null ? Math.max(accC, accQ) : accC != null ? accC : accQ;

  document.getElementById("resultsStatus").textContent = "\u2713 ANALYSIS COMPLETE";
  const modelPart = (c.model || "model") + " and the hybrid VQC on " +
    (d.name || "the dataset") + " — " + (d.rows != null ? d.rows : MISS) + " samples.";
  document.getElementById("resultsSubtitle").textContent =
    (isAlzFocus() ? "Early Alzheimer's detection — metrics from the " : "Real metrics from the ") + modelPart;

  const g = document.getElementById("gaugeValue");
  g.style.strokeDasharray = CIRC;
  const targetOffset = acc != null ? CIRC * (1 - Math.min(Math.max(acc, 0), 1)) : CIRC;
  g.style.strokeDashoffset = CIRC;
  setTimeout(() => { g.style.strokeDashoffset = targetOffset; }, 120);
  document.getElementById("gaugeAccuracy").textContent = pct(acc);
  document.getElementById("gaugeState").textContent =
    acc == null ? "Awaiting run"
    : accC != null && accQ != null ? (acc === accC ? "Best overall — Classical ML" : "Best overall — Hybrid QML")
    : acc >= 0.7 ? "Strong result" : acc >= 0.5 ? "Moderate result" : "Weak result";

  document.getElementById("resultsHeadline").textContent =
    (isAlzFocus() ? "Alzheimer's detection — " : "") +
    (q && q.metrics
      ? "Classical scored " + pct(c.metrics && c.metrics.accuracy) + " · Hybrid Quantum scored " + pct(q.metrics.accuracy)
      : "Classical model scored " + pct(c.metrics && c.metrics.accuracy) + ".");
  document.getElementById("resultsText").textContent =
    (isAlzFocus() ? "Accuracy below is measured on the Alzheimer's sample (class 1 = Alzheimer's/Dementia, class 0 = no Alzheimer's). " : "") +
    "Both models were trained and evaluated on your data with an 80/20 stratified split — no values are hardcoded. " +
    (r.winner
      ? "In this run " + r.winner.name + " performed better, and the winner can change across datasets."
      : "Use the tabs to inspect every metric, the confusion matrix and the per-feature influence.");

  document.getElementById("riDisease").textContent = d.disease && /alz/i.test(String(d.disease)) ? "Alzheimer's (Dementia)" : (d.disease || "General biomedical");
  document.getElementById("riDataset").textContent = d.name || MISS;
  document.getElementById("riModel").textContent = c.model || MISS;
  document.getElementById("riStatus").textContent = "Complete";

  buildMetricCards();
  buildBigMatrix();
  buildPredictionResult();

  document.getElementById("diName").textContent = d.name || MISS;
  document.getElementById("diRows").textContent = d.rows != null ? d.rows : MISS;
  document.getElementById("diFeat").textContent = d.features != null ? d.features : MISS;
  document.getElementById("diModel").textContent = c.model || MISS;
  document.getElementById("diModelDetail").textContent =
    "Hybrid approach: classical " + (c.model || "") + " baseline + variational quantum circuit" +
    (q && q.info ? " with " + q.info.n_qubits + " qubits." : ".");
  document.getElementById("diTime").textContent = fmtTimeSec(c.training_time);
  document.getElementById("diCTime").textContent = fmtTimeSec(c.training_time);
  document.getElementById("diQTime").textContent = q ? fmtTimeSec(q.training_time) : MISS;
  document.getElementById("diStatus").textContent = "Complete";
  document.getElementById("diStatusDetail").textContent =
    (d.rows != null ? d.rows : MISS) + " samples · " + (d.features != null ? d.features : MISS) +
    " features · binary classification · 80/20 split · standard scaling.";
}

function buildMetricCards() {
  const r = state.result;
  if (!r) return;
  const c = r.classical, q = r.quantum;
  const cm = c.metrics, qm = q && q.metrics;
  const names = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall (Sensitivity)", "recall"], ["F1-score", "f1"], ["Specificity", "specificity"]];
  let html = "";
  names.forEach(([label, key]) => {
    const cv = cm ? cm[key] : null;
    const qv = qm ? qm[key] : null;
    let tag = "";
    if (cv != null && qv == null) tag = "CLASS";
    else if (cv != null && qv != null) tag = "C / Q";
    if (cv != null && qv != null && qv > cv) tag = "QML-BEST";
    html +=
      '<div class="metric-card"><small>' + label + "</small><strong>" + pct(cv) +
      "</strong><b class='up'>" + tag + "</b>" +
      (qv != null ? '<span style="display:block;font-size:9px;color:#9b8cff">QML ' + pct(qv) + "</span>" : "") +
      "</div>";
  });
  html += '<div class="metric-card"><small>Classical Training</small><strong>' +
    fmtTimeSec(c.training_time) + "</strong><b class='flat'>MEASURED</b></div>";
  if (q) html += '<div class="metric-card"><small>Quantum Training</small><strong>' +
    fmtTimeSec(q.training_time) + "</strong><b class='up'>VQC</b></div>";
  document.getElementById("metricCards").innerHTML = html;
}

function short(name) {
  const s = String(name);
  return s.length > 16 ? s.slice(0, 14) + "\u2026" : s;
}

function buildBigMatrix() {
  const r = state.result;
  if (!r) return;
  const c = r.classical, q = r.quantum;
  const cm = c.metrics ? c.metrics.confusion_matrix : null;
  const qcm = q && q.metrics ? q.metrics.confusion_matrix : null;
  const d = r.dataset || {};
  const pos = d.class_names ? (isAlzFocus() ? prettyClass(d.class_names[1]) : short(d.class_names[1])) : "Positive";
  const neg = d.class_names ? (isAlzFocus() ? prettyClass(d.class_names[0]) : short(d.class_names[0])) : "Negative";
  const box = document.getElementById("resultsMatrix");
  if (!cm) { box.innerHTML = "<p>No confusion matrix available for this run.</p>"; return; }
  const tn = cm[0][0], fp = cm[0][1] != null ? cm[0][1] : 0, fn = cm[1][0], tp = cm[1][1] != null ? cm[1][1] : 0;
  box.innerHTML =
    '<div class="cell tn">TN — TRUE NEGATIVE <small>predicted ' + neg + ", correct</small><b>" + tn + "</b></div>" +
    '<div class="cell fp">FP — FALSE POSITIVE <small>predicted ' + pos + " wrongly</small><b>" + fp + "</b></div>" +
    '<div class="cell fn">FN — FALSE NEGATIVE <small>predicted ' + neg + " wrongly</small><b>" + fn + "</b></div>" +
    '<div class="cell tp">TP — TRUE POSITIVE <small>predicted ' + pos + ", correct</small><b>" + tp + "</b></div>";
  document.getElementById("cmNote").textContent =
    "Confusion matrix of the " + (c.model || "classical") + " model on the held-out test set." +
    (qcm ? " QML matrix: TN " + qcm[0][0] + " · FP " + (qcm[0][1] || 0) + " · FN " + qcm[1][0] + " · TP " + (qcm[1][1] || 0) + "." : "");
}

function buildPredictionResult() {
  const r = state.result;
  if (!r) return;
  const p = r.prediction;
  const box = document.getElementById("resultsPrediction");
  if (!p) { box.innerHTML = "<p>No sample prediction returned.</p>"; return; }
  const label = p.classical_label;
  const prob = p.classical_probability != null ? p.classical_probability : null;
  const actual = p.actual_label;
  const match = label === actual;
  const chips = [];
  if (p.quantum_label !== undefined) {
    chips.push(
      '<div class="pred-box"><small>QUANTUM PREDICTION</small><strong>' + prettyClass(p.quantum_label) +
      "</strong><span class='prob'>" + (p.quantum_probability != null ? p.quantum_probability.toFixed(1) + "%" : MISS) +
      " confidence</span></div>"
    );
  }
  box.innerHTML =
    '<div class="prediction-hero">' +
    '<div class="pred-box"><small>PREDICTED CLASS</small><strong>' + prettyClass(label) +
    "</strong><span class='prob'>" + (prob != null ? prob.toFixed(1) + "%" : MISS) +
    ' confidence</span><div class="pred-match" style="background:' + (match ? "#123a2a" : "#3a2412") + ";color:" + (match ? "#42dda0" : "#e0b05a") + '">' +
    (match ? "✓ MATCHES ACTUAL" : "✗ DIFFERS FROM ACTUAL") + "</div></div>" +
    '<div class="pred-box"><small>ACTUAL TEST LABEL</small><strong>' + prettyClass(actual) + "</strong>" +
    "<span class='prob'>Held-out test sample</span><div class='pred-match'>Real model output</div></div>" +
    chips.join("") + "</div>" +
    '<p style="margin-top:10px;font-size:10px;color:#6f86a4">Prediction for the first held-out test row, produced by the trained ' +
    (r.classical ? r.classical.model : "classical") + " model — not a medical diagnosis" +
    (isAlzFocus() ? " of Alzheimer's disease." : ".") + "</p>";
}
/* ------------------------------------------------------------------ */
/* DATASET PAGE + upload handling                                      */
/* ------------------------------------------------------------------ */

const PALETTE = ["#42dda0", "#9b8cff", "#16cde7", "#e0b05a", "#6ea8ff", "#ff8fb3"];

function chartDefaults() {
  if (window.Chart) {
    Chart.defaults.color = "#b7c6d9";
    Chart.defaults.borderColor = "#1a2c46";
  }
}

function renderDatasetPage() {
  const rows = state.raw || [];
  const keys = rows.length ? Object.keys(rows[0]) : [];
  if (!state.target) state.target = helpers_lastCol(keys);
  document.getElementById("rows").textContent = rows.length;
  document.getElementById("cols").textContent = keys.length;
  document.getElementById("targetLabel").textContent = state.target || MISS;
  const s = document.getElementById("targetSelect");
  if (s) {
    s.innerHTML = keys.map((k) => '<option value="' + k + '"' + (k === state.target ? " selected" : "") + ">" + k + "</option>").join("");
    s.onchange = () => { state.target = s.value; renderDatasetPage(); };
  }
  document.getElementById("dataTable").innerHTML =
    "<thead><tr>" + keys.map((k) => "<th>" + k + "</th>").join("") + "</tr></thead><tbody>" +
    rows.slice(0, 10).map((r) => "<tr>" + keys.map((k) => "<td>" + (r[k] == null ? "" : r[k]) + "</td>").join("") + "</tr>").join("") +
    "</tbody>";
  drawDoughnut("classChart", rows, state.target);
}
function helpers_lastCol(keys) { return keys.length ? keys[keys.length - 1] : null; }

function classCounts(rows, target) {
  const c = {};
  rows.forEach((r) => { const k = String(r[target]); c[k] = (c[k] || 0) + 1; });
  return c;
}
function drawDoughnut(canvasId, rows, target) {
  const c = classCounts(rows, target);
  const labels = Object.keys(c);
  if (charts[canvasId]) charts[canvasId].destroy();
  const el = document.getElementById(canvasId);
  if (!el || !window.Chart) return;
  charts[canvasId] = new Chart(el, {
    type: "doughnut",
    data: { labels, datasets: [{ data: labels.map((k) => c[k]), backgroundColor: PALETTE.slice(0, Math.max(labels.length, 2)) }] },
    options: { plugins: { legend: { labels: { color: "#b7c6d9" } } } },
  });
}

function bindUpload() {
  const dz = document.getElementById("dropzone");
  const fi = document.getElementById("fileInput");
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("over");
    if (e.dataTransfer.files.length) acceptFile(e.dataTransfer.files[0]);
  });
  fi.addEventListener("change", () => { if (fi.files.length) acceptFile(fi.files[0]); });
}
function acceptFile(file) {
  hideUploadError();
  const name = (file.name || "").toLowerCase();
  if (!name.endsWith(".csv")) {
    showErrorBox("Invalid file format. Please upload a .csv file.");
    return;
  }
  uploadedFile = file;
  state.fileName = file.name;
  document.getElementById("dzTitle").textContent = file.name;
  document.getElementById("dzSub").textContent = "CSV selected — ready to analyze";
  const badge = document.getElementById("fileBadge");
  badge.classList.remove("hidden");
  badge.textContent = "\u2713 " + file.name + " loaded";
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const parsed = Papa.parse(String(e.target.result), { header: true, dynamicTyping: true, skipEmptyLines: true });
      const rows = parsed.data || [];
      if (!rows.length) { showErrorBox("The uploaded CSV appears to be empty."); return; }
      state.raw = rows;
      state.target = null;
      renderDatasetPage();
      renderPreprocess();
    } catch (err) {
      showErrorBox("Could not parse the CSV file. Please check its format.");
    }
  };
  reader.readAsText(file);
}

function downloadSample() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([sampleCSV], { type: "text/csv" }));
  a.download = "qdetex_sample_biomedical.csv";
  a.click();
}

/* ------------------------------------------------------------------ */
/* PREPROCESSING                                                       */
/* ------------------------------------------------------------------ */

function renderPreprocess() {
  const rows = state.raw || [];
  let miss = 0, dups = 0;
  const seen = new Set();
  rows.forEach((r) => {
    const sig = JSON.stringify(r);
    if (seen.has(sig)) dups++; else seen.add(sig);
    Object.keys(r).forEach((k) => { const v = r[k]; if (v === "" || v === null || v === undefined || Number.isNaN(Number(v)) && typeof v === "number" && Number.isNaN(v)) miss++; });
  });
  document.getElementById("missingCount").textContent = miss;
  document.getElementById("duplicateCount").textContent = dups;
  document.getElementById("encodingState").textContent = "ON";
  document.getElementById("scalingState").textContent = "STANDARD";
  const r = state.result;
  const log = document.getElementById("preprocessLog");
  if (r) {
    log.textContent =
      "Analysis complete: " + r.dataset.rows + " rows × " + r.dataset.features + " features validated. " +
      "Missing cells: " + miss + ". Duplicates: " + dups + ". Target: '" + r.dataset.target + "'. " +
      "Pipeline: impute → encode → standard-scale → 80/20 stratified split → train → test.";
  } else {
    log.textContent =
      "Validated " + rows.length + " rows × " + (rows.length ? Object.keys(rows[0]).length : 0) +
      " columns. Missing cells: " + miss + ". Duplicates: " + dups + ". Run the analysis to execute the full pipeline and see the model results.";
  }
}

/* ------------------------------------------------------------------ */
/* FEATURE ENGINEERING                                                 */
/* ------------------------------------------------------------------ */

function renderFeatures() {
  const r = state.result;
  const note = document.getElementById("featureNote");
  const list = document.getElementById("featureList");
  const fs = document.getElementById("featureCount");
  if (fs) fs.onchange = () => { document.getElementById("uploadFeatureCount").value = fs.value; note.textContent = "Will apply to the next analysis run."; };
  if (r && r.quantum && r.quantum.info && r.quantum.info.selected_features && r.quantum.info.selected_features.length) {
    const feat = r.quantum.info.selected_features;
    document.getElementById("snapFeatures").textContent = feat.length;
    list.innerHTML = feat.map((f, i) => '<span><b>Q' + (i + 1) + "</b><em>" + f + "</em></span>").join("");
    note.textContent = r.messages && r.messages.quantum_features_note ? r.messages.quantum_features_note : "Kept by mutual-information selection for the VQC.";
    if (!r.quantum.info.method || r.quantum.info.method === "topk") {
      note.textContent += " These are real, selected columns from your data.";
    }
    drawFeatureScatter(feat);
  } else {
    list.innerHTML = "<p class='cm-note'>Run an analysis to see which features the quantum model actually uses.</p>";
    const num = numCols(state.raw, state.target).slice(0, 2);
    if (num.length >= 2) drawScatterState(num[0], num[1]);
  }
}
function drawFeatureScatter(feat) {
  const num = numCols(state.raw, state.target);
  const x = feat[0], y = feat[1] || feat[0];
  drawScatter(x, y);
}
function drawScatterState(x, y) { drawScatter(x, y); }
function drawScatter(x, y, canvasId) {
  const id = canvasId || "pcaChart";
  const rows = state.raw || [];
  const t = state.target;
  const counts = classCounts(rows, t);
  const names = Object.keys(counts);
  const sets = [];
  names.forEach((n, i) => {
    sets.push({
      label: "Class " + n,
      data: rows.filter((r) => r[x] != null && r[y] != null && String(r[t]) === n).map((r) => ({ x: +r[x], y: +r[y] })),
      backgroundColor: PALETTE[i] || PALETTE[0],
    });
  });
  if (charts[id]) charts[id].destroy();
  if (!window.Chart) return;
  charts[id] = new Chart(document.getElementById(id), {
    type: "scatter",
    data: { datasets: sets },
    options: { scales: { x: { title: { display: true, text: x } }, y: { title: { display: true, text: y } } }, plugins: { legend: { labels: { color: "#b7c6d9" } } } },
  });
}

/* ------------------------------------------------------------------ */
/* CLASSICAL PAGE                                                      */
/* ------------------------------------------------------------------ */

function renderClassical() {
  const box = document.getElementById("classicalMetrics");
  const r = state.result;
  if (!r) {
    box.innerHTML = '<div class="metric"><small>ACCURACY</small><strong>—</strong><span>Run the analysis</span></div>';
    document.getElementById("classicalNote").textContent = "No model has been trained yet.";
    document.getElementById("classicalMatrix").innerHTML = "<p>Awaiting results.</p>";
    return;
  }
  const m = r.classical.metrics || {};
  const defs = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1-score", "f1"], ["Specificity", "specificity"]];
  box.innerHTML = defs.map(([l, k]) =>
    '<div class="metric"><small>' + l + "</small><strong>" + pct(m[k]) + "</strong><span>Measured result</span></div>"
  ).join("") +
    '<div class="metric"><small>Training Time</small><strong>' + fmtTimeSec(r.classical.training_time) + "</strong><span>" + r.classical.n_features + " features</span></div>";
  const cm = m.confusion_matrix;
  document.getElementById("classicalMatrix").innerHTML =
    cm
      ? '<div class="tm">TN <b>' + cm[0][0] + "</b></div><div>FP <b>" + (cm[0][1] || 0) + "</b></div><div>FN <b>" + cm[1][0] + "</b></div><div>TP <b>" + (cm[1][1] || 0) + "</b></div>"
      : "<p>Awaiting results.</p>";
  document.getElementById("classicalNote").textContent =
    r.classical.model + " trained on " + r.classical.n_features + " features with a " +
    r.classical.test_samples + "-sample test set. AUC: " + pct(m.auc) + ". Numbers come from real predictions.";
}

/* ------------------------------------------------------------------ */
/* QUANTUM PAGE                                                        */
/* ------------------------------------------------------------------ */

function renderQuantum() {
  const r = state.result;
  const res = document.getElementById("quantumResult");
  const errBox = document.getElementById("quantumErrorBox");
  const qm = document.getElementById("quantumMetrics");
  if (!r) {
    res.textContent = "Waiting for quantum experiment…";
    errBox.classList.add("hidden");
    return;
  }
  if (!r.quantum || !r.quantum.metrics) {
    res.textContent = r.quantum_error || "The hybrid quantum model could not be trained on this dataset.";
    errBox.classList.remove("hidden");
    errBox.textContent = "Quantum step skipped: " + (r.quantum_error || "unknown error") + " The classical results remain valid.";
    document.getElementById("qQubits").textContent = "—";
    qm.innerHTML = "<p>No quantum metrics available.</p>";
    return;
  }
  const q = r.quantum;
  document.getElementById("qQubits").textContent = q.info ? q.info.n_qubits : "4";
  errBox.classList.add("hidden");
  res.textContent =
    "Quantum experiment complete. " + (q.info ? q.info.samples_used : "—") + " training samples, " +
    (q.info ? q.info.maxiter : "—") + " SPSA iterations, " + (q.info ? q.info.n_qubits : "—") + " qubits. " +
    "Measured accuracy: " + pct(q.metrics.accuracy) + " · F1: " + pct(q.metrics.f1) + " · AUC: " + pct(q.metrics.auc);
  const defs = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1-score", "f1"], ["AUC", "auc"]];
  qm.innerHTML = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">' +
    defs.map(([l, k]) => '<div class="process quantum-card"><span>' + l + '</span><strong style="font-size:16px">' + pct(q.metrics[k]) + "</strong></div>").join("") +
    '<div class="process quantum-card"><span>TRAIN TIME</span><strong style="font-size:16px">' + fmtTimeSec(q.training_time) + "</strong></div>" +
    "</div>";
}

/* ------------------------------------------------------------------ */
/* PREDICTION PAGE                                                     */
/* ------------------------------------------------------------------ */

function renderPrediction() {
  const sel = document.getElementById("predictionModel");
  const r = state.result;
  if (!r || !r.prediction) {
    document.getElementById("predictionClass").textContent = MISS;
    document.getElementById("predictionText").textContent = "Run the analysis to see a real prediction.";
    document.getElementById("actualLabel").textContent = MISS;
    document.getElementById("predictionDetails").textContent = "No prediction yet.";
    return;
  }
  const p = r.prediction;
  const mode = sel ? sel.value : "Classical ML";
  const isQuantum = mode.indexOf("Quantum") >= 0 && p.quantum_label !== undefined;
  const label = isQuantum ? p.quantum_label : p.classical_label;
  const prob = isQuantum ? p.quantum_probability : p.classical_probability;
  const modelName = isQuantum ? "Hybrid Quantum ML (VQC)" : r.classical.model;
  document.getElementById("predictionClass").textContent = prettyClass(label);
  document.getElementById("actualLabel").textContent = prettyClass(p.actual_label);
  const w = prob != null ? Math.min(prob, 100) : 0;
  document.getElementById("confidenceBar").style.width = w + "%";
  document.getElementById("predictionText").textContent =
    modelName + " predicted " + prettyClass(label) + " with " + (prob != null ? prob.toFixed(1) + "%" : "unknown") + " confidence.";
  document.getElementById("predictionDetails").textContent =
    "Model: " + modelName + ".\nActual label of the first held-out test row: " + prettyClass(p.actual_label) +
    ".\nPredicted: " + prettyClass(label) + ". Probability of the positive class: " + (prob != null ? prob.toFixed(1) + "%" : MISS) +
    ".\nThis is a real model output on test data — not a diagnosis" + (isAlzFocus() ? " of Alzheimer's disease." : ".");
  const pc = document.getElementById("predRiskChip");
  if (pc) {
    const rz = r.risk;
    if (!rz || rz.score == null) { pc.textContent = "RISK —"; pc.style.borderColor = "#2b4266"; }
    else {
      pc.textContent = rz.level_assessment + " RISK · " + Math.round(rz.score) + "%";
      pc.style.borderColor = rz.level_assessment === "High" ? "#e87ba0" : rz.level_assessment === "Moderate" ? "#e0b05a" : "#42dda0";
      pc.style.color = rz.level_assessment === "High" ? "#e87ba0" : rz.level_assessment === "Moderate" ? "#e0b05a" : "#42dda0";
    }
  }
}

function renderRisk() {
  const r = state.result;
  const box = document.getElementById("riskFactors");
  const chip = document.getElementById("riskLevel");
  const scoreEl = document.getElementById("riskScore");
  const sum = document.getElementById("riskSummary");
  if (!box || !r || !r.risk) {
    if (chip) chip.textContent = "—";
    if (scoreEl) scoreEl.textContent = "—";
    return;
  }
  const rz = r.risk;
  const col = rz.level_assessment === "High" ? "#e87ba0" : rz.level_assessment === "Moderate" ? "#e0b05a" : "#42dda0";
  if (chip) {
    chip.textContent = rz.level_assessment + " RISK";
    chip.style.borderColor = col;
    chip.style.color = col;
    chip.style.background = "none";
  }
  if (scoreEl) {
    scoreEl.textContent = rz.score != null ? Math.round(rz.score) : MISS;
    scoreEl.style.background = "linear-gradient(120deg," + col + ",#16cde7)";
    scoreEl.style.webkitBackgroundClip = "text";
    scoreEl.style.backgroundClip = "text";
    scoreEl.style.color = "transparent";
  }
  document.getElementById("riskPositiveLabel").textContent = prettyClass(rz.positive_label);
  document.getElementById("riskRefClass").textContent = isAlzFocus() ? "Normal" : "non-case";
  if (sum) sum.textContent = rz.note || "";
  const dirTxt = { above: "above", below: "below" };
  box.innerHTML = (!rz.factors || !rz.factors.length)
    ? '<p class="cm-note">No strongly deviating features for this patient — the score is set by the overall model on this test row.</p>'
    : rz.factors.map((f, i) => {
        const inc = f.increases_risk;
        const c = inc ? "#e87ba0" : "#42dda0";
        const arrow = inc ? "↑" : "↓";
        const action = inc ? "raises" : "lowers";
        const off = f.deviation > 0 ? "+" : "";
        const bar = Math.min(100, Math.abs(f.deviation) * 15);
        return '<div class="rf"><div class="rf-top"><b>' + (i + 1) + ". " + f.feature + "</b><span>" + dirTxt[f.direction] +
          ' the normal reference (' + off + f.deviation + "σ)</span></div>" +
          '<div class="rf-bar"><i style="width:' + bar + '%;background:' + c + '"></i></div>' +
          "<em class=\"" + (inc ? "bad" : "good") + '">' + arrow + " " + action + " risk · " + f.contribution + "% of model influence</em></div>";
      }).join("");
}
document.addEventListener("DOMContentLoaded", () => {
  const sel = document.getElementById("predictionModel");
  if (sel) sel.onchange = renderPrediction;
  const fs = document.getElementById("featureCount");
  if (fs) fs.onchange = renderFeatures;
});

/* ------------------------------------------------------------------ */
/* COMPARISON PAGE                                                     */
/* ------------------------------------------------------------------ */

function renderComparison() {
  const r = state.result;
  const tbody = document.querySelector("#compareTable tbody");
  if (!r || !r.classical) {
    tbody.innerHTML = '<tr><td colspan="3">Run the analysis to populate actual, measured results.</td></tr>';
    document.getElementById("comparisonWinner").textContent = "Awaiting experiment";
    document.getElementById("comparisonDelta").textContent = MISS;
    return;
  }
  const c = r.classical, q = r.quantum;
  const rows = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1-score", "f1"], ["Specificity", "specificity"], ["AUC", "auc"]];
  tbody.innerHTML = rows.map(([l, k]) =>
    "<tr><td>" + l + "</td><td>" + pct(c.metrics[k]) + "</td><td>" + (q && q.metrics ? pct(q.metrics[k]) : "—") + "</td></tr>"
  ).join("") +
    "<tr><td>Training time</td><td>" + fmtTimeSec(c.training_time) + "</td><td>" + (q ? fmtTimeSec(q.training_time) : "—") + "</td></tr>";
  if (r.winner && r.winner.name) {
    document.getElementById("comparisonWinner").textContent = r.winner.name + " performed higher in this run";
    document.getElementById("comparisonDelta").textContent = r.winner.margin.toFixed(1) + " pp";
  } else {
    document.getElementById("comparisonWinner").textContent = "Classical model only in this run";
    document.getElementById("comparisonDelta").textContent = MISS;
  }
  if (charts.compare) charts.compare.destroy();
  const el = document.getElementById("compareChart");
  if (!el || !window.Chart) return;
  const labels = ["Accuracy", "Precision", "Recall", "F1", "Specificity", "AUC"];
  charts.compare = new Chart(el, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Classical ML", data: labels.map((l) => (c.metrics[{"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1": "f1", "Specificity": "specificity", "AUC": "auc"}[l]] || 0) * 100), backgroundColor: "#6ea8ff" },
        { label: "Hybrid QML", data: labels.map((l) => (q && q.metrics ? (q.metrics[{"Accuracy": "accuracy", "Precision": "precision", "Recall": "recall", "F1": "f1", "Specificity": "specificity", "AUC": "auc"}[l]] || 0) : 0) * 100), backgroundColor: "#9b8cff" },
      ],
    },
    options: { scales: { y: { beginAtZero: true, max: 100 } } },
  });
}

/* ------------------------------------------------------------------ */
/* EXPLAINABILITY                                                      */
/* ------------------------------------------------------------------ */

function renderImportance() {
  const r = state.result;
  const el = document.getElementById("importanceChart");
  const ex = document.getElementById("explanation");
  if (!r || !r.importance.length) {
    ex.innerHTML = "<p>Run an analysis to compute real permutation feature importance.</p>";
    if (charts.imp) charts.imp.destroy();
    return;
  }
  const items = r.importance.slice(0, 5).reverse();
  if (charts.imp) charts.imp.destroy();
  if (el && window.Chart) {
    charts.imp = new Chart(el, {
      type: "bar",
      data: {
        labels: items.map((i) => i.feature),
        datasets: [{ label: "Relative influence (%)", data: items.map((i) => i.contribution), backgroundColor: "#735cff" }],
      },
      options: { indexAxis: "y", plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } },
    });
  }
  ex.innerHTML =
    "<p class='cm-note'>Ranked by permutation importance (drop in accuracy when the feature is shuffled).</p>" +
    items.reverse().map((i, idx) =>
      "<p style='margin-top:10px'><b>" + i.feature + "</b> — rank #" + (idx + 1) + " · " + i.contribution.toFixed(1) + "% relative influence on the model.</p>"
    ).join("") +
    "<p class='cm-note' style='margin-top:12px'>Feature importance describes this experiment's model behavior, not medical causation.</p>";
}

/* ------------------------------------------------------------------ */
/* ANALYTICS PAGE                                                      */
/* ------------------------------------------------------------------ */

function renderVisuals() {
  drawDoughnut("classChart2", state.raw, state.target);
  const rows = state.raw || [];
  const num = numCols(rows, state.target).slice(0, 3);
  if (charts.dist) charts.dist.destroy();
  const el = document.getElementById("distChart");
  if (el && window.Chart) {
    const sample = rows.slice(0, 20);
    charts.dist = new Chart(el, {
      type: "line",
      data: {
        labels: sample.map((_, i) => "row " + (i + 1)),
        datasets: num.map((f, i) => ({ label: f, data: sample.map((r) => r[f]), borderColor: PALETTE[i], tension: 0.35, pointRadius: 2 })),
      },
      options: { plugins: { legend: { labels: { color: "#b7c6d9" } } }, scales: { y: { grid: { color: "#16263d" } } } },
    });
  }
  const box = document.getElementById("heatmap");
  const corrMat = computeCorrelations(rows, num.length ? num : numCols(rows, state.target).slice(0, 4));
  box.style.gridTemplateColumns = "repeat(" + corrMat.length + ",1fr)";
  let html = "";
  for (let i = 0; i < corrMat.length; i++) {
    for (let j = 0; j < corrMat.length; j++) {
      const v = corrMat[i][j];
      html += '<div style="--v:' + ((v + 1) / 2) + '" title="' + corrMat.names[i] + " / " + corrMat.names[j] + '">' + v.toFixed(2) + "</div>";
    }
  }
  box.innerHTML = html;
  box.dataset.tip = corrMat.names.join(",");
}
function computeCorrelations(rows, names) {
  const n = names.length;
  const mat = [];
  const cols = names.map((nm) => rows.map((r) => (r[nm] == null ? 0 : +r[nm])));
  mat.names = names;
  for (let i = 0; i < n; i++) {
    mat.push([]);
    for (let j = 0; j < n; j++) {
      if (i === j) { mat[i].push(1); continue; }
      const a = cols[i], b = cols[j];
      const ma = mean(a), mb = mean(b);
      let num = 0, da = 0, db = 0;
      for (let k = 0; k < a.length; k++) {
        const x = a[k] - ma, y = b[k] - mb;
        num += x * y; da += x * x; db += y * y;
      }
      mat[i].push(da === 0 || db === 0 ? 0 : num / Math.sqrt(da * db));
    }
  }
  return mat;
}
function mean(a) { return a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0; }

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */

chartDefaults();
async function loadPreviewFromServer() {
  try {
    const res = await fetch("/api/sample_csv");
    if (!res.ok) return;
    const txt = await res.text();
    const rows = Papa.parse(txt, { header: true, dynamicTyping: true, skipEmptyLines: true }).data;
    if (rows.length) {
      state.raw = rows;
      state.target = null;
      state.fileName = "alzheimer_dataset.csv";
      renderDatasetPage();
      renderPreprocess();
    }
  } catch (e) { /* keep the embedded preview sample */ }
}
(function initial() {
  if (window.Chart) {
    const g = document.getElementById("gaugeValue");
    if (g) { g.style.strokeDasharray = CIRC; g.style.strokeDashoffset = CIRC; }
  }
  bindUpload();
  state.target = helpers_lastCol(Object.keys(state.raw[0] || {}));
  renderDatasetPage();
  loadPreviewFromServer();
  const sampleBtn = document.getElementById("sampleBtn");
  if (sampleBtn) sampleBtn.onclick = () => analyzeSample();
  const reRunBtn = document.getElementById("reRunBtn");
  if (reRunBtn) reRunBtn.onclick = () => analyzeSample();
  const analyzeBtn = document.getElementById("analyzeBtn");
  if (analyzeBtn) analyzeBtn.onclick = () => analyzeUpload();
  const classicalBtn = document.querySelector('[data-buttont="classical"]');
  if (classicalBtn) classicalBtn.onclick = reanalyze;
})();

window.reanalyzeClassical = reanalyze;
window.reanalyzeQuantum = reanalyze;
window.analyzeUpload = analyzeUpload;
window.analyzeSample = analyzeSample;
window.renderImportance = renderImportance;
window.downloadSample = downloadSample;