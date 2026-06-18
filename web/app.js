const state = {
  status: null,
  sampleFeatures: null,
  sampleDirty: false,
  sampleRequest: null,
  generatorOptions: null,
  health: null,
  artifacts: null,
  outputNameTouched: false,
};

let metricsSort = { key: "rows", dir: "desc" };
let metricsData = [];


const viewMeta = {
  overview: "Overview",
  predictor: "Predict Flow",
  batch: "Use Dataset",
  generator: "Create Data",
  metrics: "Results",
  about: "About",
  explain: "Why Result",
  drift: "Data Drift",
};

const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString();
}

function formatBytes(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  const units = ["B", "KB", "MB", "GB"];
  let size = Number(value);
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(size >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function formatDate(value) {
  if (!value) return "--";
  return new Date(value).toLocaleString();
}

function toast(message, type = "info") {
  const icons = { ok: "\u2713", error: "\u2715", info: "\u25cf" };
  const node = document.createElement("div");
  node.className = `toast toast-${type}`;
  node.innerHTML = `<span class="toast-icon">${icons[type] || icons.info}</span><span>${escapeHtml(message)}</span>`;
  document.body.appendChild(node);
  setTimeout(() => {
    node.style.animation = "toastOut 0.28s ease forwards";
    setTimeout(() => node.remove(), 300);
  }, 3400);
}

function animateValue(element, displayText, duration = 900) {
  if (!element) return;
  const isPercent = displayText.includes("%");
  const cleanText = displayText.replace(/[,%]/g, "");
  const numericValue = parseFloat(cleanText);
  if (displayText === "--" || Number.isNaN(numericValue)) {
    element.textContent = displayText;
    return;
  }
  const start = performance.now();
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const current = numericValue * easeOut(progress);
    element.textContent = isPercent
      ? `${current.toFixed(2)}%`
      : Math.round(current).toLocaleString();
    if (progress < 1) requestAnimationFrame(step);
    else element.textContent = displayText;
  }
  requestAnimationFrame(step);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the status text.
    }
    throw new Error(detail);
  }
  return response.json();
}

function activeViewFromHash() {
  const id = window.location.hash.replace("#", "") || "overview";
  return viewMeta[id] ? id : "overview";
}

function activateView(id = activeViewFromHash()) {
  const activeId = viewMeta[id] ? id : "overview";
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active-view", view.id === activeId);
  });
  document.querySelectorAll("[data-view-link]").forEach((link) => {
    link.classList.toggle("active", link.dataset.viewLink === activeId);
  });
  $("#pageTitle").textContent = viewMeta[activeId];
}

function renderStatus(payload) {
  state.status = payload;
  const threshold = payload.training?.decision_threshold ?? 0.5;
  $("#modelStatus").textContent = `${payload.model} ready`;
  $("#runSummary").textContent = `${payload.features.total} features - threshold ${Number(
    threshold,
  ).toFixed(4)} - ${formatNumber(payload.data.rows_after_dedup)} validated rows`;

  animateValue($("#metricF1"), formatPercent(payload.test.f1));
  animateValue($("#metricPrecision"), formatPercent(payload.test.precision));
  animateValue($("#metricRecall"), formatPercent(payload.test.recall));
  animateValue($("#metricAuc"), formatPercent(payload.test.roc_auc));
  animateValue($("#metricRows"), formatNumber(payload.data.train_rows));
  document.querySelectorAll(".metric").forEach((m) => m.classList.remove("loading"));

  if (payload.model_modified_at) {
    const d = new Date(payload.model_modified_at);
    $("#modelModifiedAt").textContent = `Trained ${d.toLocaleDateString()} ${d.toLocaleTimeString()}`;
  }

  $("#metricStrategy").textContent = payload.training?.strategy || "lightgbm_baseline";
  $("#metricThreshold").textContent = Number(threshold).toFixed(4);
  $("#metricFeatures").textContent = `${payload.features.total} total, ${payload.features.numeric} numeric, ${payload.features.categorical} categorical`;

  renderMatrix(payload.test.confusion_matrix);
  renderDatasetBars(payload.data.dataset_counts);
  renderFeatureList(payload.training?.feature_importance?.top_original || []);
  renderCategoryMetrics(payload.test.attack_category_metrics || []);
}

function healthRow(item) {
  const statusClass = item.ok ? "pass" : item.required ? "fail" : "warn";
  return `
    <div class="health-item">
      <span class="check-dot ${statusClass}"></span>
      <div>
        <strong>${escapeHtml(item.label)}</strong>
        <span>${escapeHtml(item.path)}${item.size_bytes ? ` - ${formatBytes(item.size_bytes)}` : ""}</span>
      </div>
    </div>
  `;
}

function renderHealth(payload) {
  state.health = payload;
  const ready = payload.status === "ready";
  $("#healthStatus").textContent = ready ? "Ready for Docker run" : "Setup needs review";
  $("#healthUpdated").textContent = `Updated ${new Date().toLocaleTimeString()}`;

  const rows = payload.checks?.map(healthRow).join("") || `<div class="empty-row">No health checks returned</div>`;
  $("#readinessList").innerHTML = rows;
}

function reportLinks(reports) {
  if (!reports?.length) return `<div class="empty-row">No report files found</div>`;
  return reports
    .map(
      (report) => `
        <a class="report-link" href="${report.download_url}">
          <span>${escapeHtml(report.name)}</span>
          <strong>${formatBytes(report.size_bytes)}</strong>
        </a>
      `,
    )
    .join("");
}

function renderArtifacts(payload) {
  state.artifacts = payload;
  const syntheticFiles = payload.synthetic || [];
  const parquetFiles = syntheticFiles.filter((item) => item.name.endsWith(".parquet"));
  const datasets = syntheticFiles.filter((item) => item.name.endsWith(".parquet") || item.name.endsWith(".csv"));
  const latest = parquetFiles[0] || datasets[0];
  $("#latestArtifact").textContent = latest ? latest.name : "--";
  $("#latestRows").textContent = latest?.rows ? formatNumber(latest.rows) : "--";
  $("#reportStatus").textContent = payload.reports?.length ? `${payload.reports.length} files` : "Missing";
  const exactMatches = latest?.quality?.exact_real_row_matches;
  $("#privacyStatus").textContent =
    exactMatches === 0 ? "0 exact matches" : exactMatches ? `${formatNumber(exactMatches)} matches` : "Review";

  $("#recentArtifactsTable").innerHTML = datasets.length
    ? datasets
        .slice(0, 8)
        .map(
          (item) => `
            <tr>
              <td>${escapeHtml(item.name)}<br /><span class="muted-small">${formatDate(item.modified_at)}</span></td>
              <td>${item.rows ? formatNumber(item.rows) : "--"}</td>
              <td>${formatBytes(item.size_bytes)}</td>
              <td><a href="${item.download_url}">Download</a></td>
            </tr>
          `,
        )
        .join("")
    : `<tr><td colspan="4" class="empty-row">No generated dataset files found</td></tr>`;

  const links = reportLinks(payload.reports || []);
  $("#reportLinks").innerHTML = links;
}

function renderFeatureList(features) {
  if (!features.length) {
    $("#featureList").innerHTML = `<div class="empty-row">No feature importance saved yet</div>`;
    return;
  }
  const maxGain = Math.max(...features.map((item) => Number(item.gain) || 0), 1);
  $("#featureList").innerHTML = features
    .slice(0, 10)
    .map((item) => {
      const width = Math.max(4, ((Number(item.gain) || 0) / maxGain) * 100);
      return `
        <div class="feature-row">
          <span>${escapeHtml(item.feature)}</span>
          <strong>${width.toFixed(0)}%</strong>
        </div>
      `;
    })
    .join("");
}

function renderMatrix(matrix) {
  const labels = ["True normal", "False attack", "False normal", "True attack"];
  const classes = ["tn", "fp", "fn", "tp"];
  const values = matrix ? [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]] : [];
  $("#matrix").innerHTML = labels
    .map(
      (label, index) => `
      <div class="matrix-cell ${classes[index]}">
        <span>${label}</span>
        <strong>${formatNumber(values[index])}</strong>
      </div>
    `,
    )
    .join("");
}

function renderDatasetBars(counts) {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  const max = entries.length ? entries[0][1] : 1;
  $("#datasetBars").innerHTML = entries
    .map(([name, count]) => {
      const width = Math.max(3, (count / max) * 100);
      return `
        <div class="bar-row">
          <span>${escapeHtml(name)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <strong>${formatNumber(count)}</strong>
        </div>
      `;
    })
    .join("");
}

function rateColorClass(rate, isAttack) {
  if (rate === null || rate === undefined) return "";
  if (!isAttack) return rate <= 0.02 ? "rate-ok" : rate <= 0.06 ? "rate-warn" : "rate-danger";
  return rate >= 0.95 ? "rate-ok" : rate >= 0.80 ? "rate-warn" : "rate-danger";
}

function probColorClass(prob, isAttack) {
  if (prob === null || prob === undefined) return "";
  if (!isAttack) return prob <= 0.03 ? "rate-ok" : prob <= 0.08 ? "rate-warn" : "rate-danger";
  return prob >= 0.90 ? "rate-ok" : prob >= 0.70 ? "rate-warn" : "rate-danger";
}

function rateValue(row) {
  return row.label === 1 ? (row.detection_rate ?? 0) : 1 - (row.false_positive_rate ?? 0);
}

function renderCategoryMetricsTable() {
  if (!metricsData.length) return;

  const sorted = [...metricsData].sort((a, b) => {
    const key = metricsSort.key;
    const dir = metricsSort.dir === "asc" ? 1 : -1;
    if (key === "attack_category") {
      return dir * a.attack_category.toLowerCase().localeCompare(b.attack_category.toLowerCase());
    }
    if (key === "rate") return dir * (rateValue(a) - rateValue(b));
    if (key === "mean_attack_probability") return dir * ((a.mean_attack_probability ?? 0) - (b.mean_attack_probability ?? 0));
    return dir * ((a.rows ?? 0) - (b.rows ?? 0));
  });

  $("#categoryMetricsTable").innerHTML = sorted
    .slice(0, 40)
    .map((row) => {
      const isAttack = row.label === 1;
      const rate = isAttack ? (row.detection_rate ?? null) : (row.false_positive_rate ?? null);
      const rateLabel = isAttack ? "Detection" : "False positive";
      const rColor = rateColorClass(rate, isAttack);
      const pColor = probColorClass(row.mean_attack_probability, isAttack);
      const pct = rate !== null ? Math.min(100, Math.max(0, rate * 100)) : 0;
      const typeChip = isAttack
        ? `<span class="cat-chip cat-attack">Attack</span>`
        : `<span class="cat-chip cat-normal">Normal</span>`;
      const isConcern = isAttack && rate !== null && rate < 0.75;

      return `
        <tr${isConcern ? ' class="row-concern"' : ''}>
          <td>
            <div class="cat-name-cell">
              ${typeChip}
              <span>${escapeHtml(row.attack_category)}</span>
            </div>
          </td>
          <td>${formatNumber(row.rows)}</td>
          <td><span class="${pColor}">${formatPercent(row.mean_attack_probability)}</span></td>
          <td>
            <div class="rate-cell">
              <span class="rate-label ${rColor}">${rateLabel} ${formatPercent(rate)}</span>
              <div class="rate-bar-wrap"><div class="rate-bar-fill ${rColor}" style="width:${pct.toFixed(1)}%"></div></div>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");

  document.querySelectorAll(".metrics-sort-th").forEach((th) => {
    const arrow = th.querySelector(".sort-arrow");
    if (!arrow) return;
    const key = th.dataset.sortKey;
    if (key === metricsSort.key) {
      arrow.textContent = metricsSort.dir === "asc" ? " \u2191" : " \u2193";
    } else {
      arrow.textContent = " \u2195";
    }
  });
}

function renderCategoryMetrics(rows) {
  metricsData = rows;
  if (!rows.length) {
    $("#categoryMetricsTable").innerHTML = `<tr><td colspan="4" class="empty-row">No category metrics saved yet</td></tr>`;
    return;
  }
  renderCategoryMetricsTable();
}

function renderCheckboxGrid(gridId, name, items, options = {}) {
  const grid = $(gridId);
  if (!items || !items.length) {
    grid.innerHTML = `<div class="empty-row">No filters available</div>`;
    return;
  }

  grid.innerHTML = items
    .map((item) => {
      const label = item.label ?? item.value ?? item.name;
      const value = item.value ?? item.name;
      const type = item.label === 0 || item.value === 0 ? "normal" : item.label === 1 || item.value === 1 ? "attack" : "";
      const checked = options.checked === false ? "" : "checked";
      return `
        <label class="category-option ${type}">
          <input type="checkbox" name="${name}" value="${escapeHtml(value)}" ${checked} data-label="${escapeHtml(label)}" />
          <span>
            <strong>${escapeHtml(item.name)}</strong>
            <span>${formatNumber(item.rows)} training rows</span>
          </span>
        </label>
      `;
    })
    .join("");
}

function renderCategoryFilters(payload) {
  state.generatorOptions = payload;
  renderCheckboxGrid("#labelGrid", "label_filter", payload.labels);
  renderCheckboxGrid("#categoryGrid", "category", payload.categories);
  renderCheckboxGrid("#sourceGrid", "dataset_source", payload.dataset_sources);
  renderCheckboxGrid("#environmentGrid", "environment_type", payload.environment_types);
  renderCheckboxGrid("#subcategoryGrid", "subcategory", payload.subcategories);
  updateSuggestedOutputName(false);
}

async function loadGeneratorOptions() {
  try {
    const payload = await api("/api/generator-options");
    renderCategoryFilters(payload);
  } catch (error) {
    $("#categoryGrid").innerHTML = `<div class="empty-row">${escapeHtml(error.message)}</div>`;
  }
}

function formFeatures() {
  const data = new FormData($("#predictionForm"));
  const features = {};
  for (const [key, value] of data.entries()) {
    if (value === "") continue;
    if (key === "protocol" || key === "service") {
      features[key] = value;
    } else {
      const parsed = Number(value);
      if (!Number.isNaN(parsed)) features[key] = parsed;
    }
  }
  return features;
}

function checkedValues(name) {
  return [...document.querySelectorAll(`input[name="${name}"]:checked`)].map((input) => input.value);
}

function slugPart(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 28);
}

function selectedLabelName() {
  const values = checkedValues("label_filter").sort();
  if (values.length === 1 && values[0] === "0") return "normal";
  if (values.length === 1 && values[0] === "1") return "attack";
  return "mixed";
}

function suggestDatasetName() {
  const date = new Date().toISOString().slice(0, 10).replaceAll("-", "");
  const labels = selectedLabelName();
  const sources = checkedValues("dataset_source");
  const categories = checkedValues("category").filter((value) => value.toLowerCase() !== "normal");
  const sourcePart = sources.length === 1 ? slugPart(sources[0]) : `${sources.length || "all"}sources`;
  const categoryPart = categories.length === 1 ? slugPart(categories[0]) : `${categories.length || "all"}families`;
  return ["cybersynth", labels, categoryPart, sourcePart, date].filter(Boolean).join("_");
}

function updateSuggestedOutputName(force = false) {
  const input = $("#outputNameInput");
  if (!input) return;
  if (force || !state.outputNameTouched || !input.value.trim()) {
    input.value = suggestDatasetName();
    state.outputNameTouched = false;
  }
}

function generatorPayload() {
  const data = new FormData($("#generatorForm"));
  const outputName = String(data.get("output_name") || "").trim() || suggestDatasetName();
  return {
    rows: Number(data.get("rows")),
    mode: data.get("mode"),
    minimum_per_category: Number(data.get("minimum_per_category")),
    noise_scale: Number(data.get("noise_scale")),
    clip_quantile: Number(data.get("clip_quantile")),
    random_state: Number(data.get("random_state")),
    output_name: outputName,
    drop_exact_matches: data.get("drop_exact_matches") === "on",
    labels: checkedValues("label_filter").map((value) => Number(value)),
    categories: checkedValues("category"),
    subcategories: checkedValues("subcategory"),
    dataset_sources: checkedValues("dataset_source"),
    environment_types: checkedValues("environment_type"),
  };
}

function fillForm(features) {
  for (const [key, value] of Object.entries(features)) {
    const input = $(`[name="${key}"]`);
    if (!input) continue;
    input.value = value;
  }
}

function setPrediction(result) {
  const probability = Number(result.attack_probability);
  const degrees = Math.max(0, Math.min(1, probability)) * 360;
  const color =
    result.risk === "Critical" || result.risk === "High"
      ? "#d1495b"
      : result.risk === "Review"
        ? "#b7791f"
        : "#10875c";

  const dial = $("#probabilityDial");
  dial.style.background = `conic-gradient(${color} ${degrees}deg, #e8eef4 ${degrees}deg)`;
  dial.classList.toggle("risk-high", result.risk === "Critical" || result.risk === "High");

  $("#probabilityValue").textContent = formatPercent(probability);
  $("#normalProbability").textContent = formatPercent(result.normal_probability);
  $("#attackProbability").textContent = formatPercent(probability);
  $("#riskLevel").textContent = result.risk;
  $("#verdictText").textContent = result.verdict;
  $("#resultTimestamp").textContent = new Date().toLocaleTimeString();

  const chip = $("#verdictChip");
  chip.textContent = result.risk;
  chip.className = `risk-chip ${result.risk.toLowerCase()}`;
}

async function loadStatus() {
  try {
    renderStatus(await api("/api/status"));
  } catch (error) {
    $("#modelStatus").textContent = "Model unavailable";
    document.querySelectorAll(".metric").forEach((m) => m.classList.remove("loading"));
    toast(error.message, "error");
  }
}

async function reloadModel() {
  const button = $("#reloadModelButton");
  if (button) button.disabled = true;
  try {
    await api("/api/reload-model", { method: "POST" });
    toast("Model cache cleared - reloading...", "ok");
    setTimeout(refreshDashboard, 600);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    if (button) button.disabled = false;
  }
}

async function loadHealth() {
  try {
    renderHealth(await api("/api/health"));
  } catch (error) {
    $("#healthStatus").textContent = "Health check failed";
    $("#readinessList").innerHTML = `<div class="empty-row">${escapeHtml(error.message)}</div>`;
  }
}

async function loadArtifacts() {
  try {
    renderArtifacts(await api("/api/artifacts"));
  } catch (error) {
    $("#recentArtifactsTable").innerHTML = `<tr><td colspan="4" class="empty-row">${escapeHtml(error.message)}</td></tr>`;
    $("#reportLinks").innerHTML = `<div class="empty-row">${escapeHtml(error.message)}</div>`;
  }
}

async function refreshDashboard() {
  await Promise.all([loadStatus(), loadHealth(), loadArtifacts()]);
}

async function loadSample(kind) {
  const request = api(`/api/sample?kind=${kind}`);
  state.sampleRequest = request;
  try {
    const payload = await request;
    state.sampleFeatures = payload.features;
    state.sampleDirty = false;
    fillForm(payload.features);
    toast(`${payload.truth.dataset_source} - ${payload.truth.attack_category}`, "ok");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    if (state.sampleRequest === request) state.sampleRequest = null;
  }
}

async function submitPrediction(event) {
  event.preventDefault();
  try {
    if (state.sampleRequest) await state.sampleRequest;
    const visibleFeatures = formFeatures();
    const features =
      state.sampleFeatures && !state.sampleDirty
        ? state.sampleFeatures
        : {
            ...(state.sampleFeatures || {}),
            ...visibleFeatures,
          };
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    setPrediction(result);
  } catch (error) {
    toast(error.message);
  }
}

function riskChip(risk) {
  return `<span class="risk-chip ${risk.toLowerCase()}">${escapeHtml(risk)}</span>`;
}

function renderBatch(payload) {
  if (payload.validation) renderUploadValidation(payload.validation);
  $("#batchRows").textContent = formatNumber(payload.rows_scored);
  $("#batchAttack").textContent = formatNumber(payload.attack_count);
  $("#batchMean").textContent = formatPercent(payload.mean_attack_probability);

  if (!payload.results.length) {
    $("#batchTable").innerHTML = `<tr><td colspan="4" class="empty-row">No batch results</td></tr>`;
    return;
  }

  $("#batchTable").innerHTML = payload.results
    .map(
      (row) => `
        <tr>
          <td>${row.row}</td>
          <td>${escapeHtml(row.verdict)}</td>
          <td>${riskChip(row.risk)}</td>
          <td>${formatPercent(row.attack_probability)}</td>
        </tr>
      `,
    )
    .join("");

  if (payload.truncated) {
    toast(`Scored first ${formatNumber(payload.rows_scored)} rows`);
  }
}

function readinessText(value) {
  if (value === "ready") return "Ready";
  if (value === "review") return "Review";
  return "Not ready";
}

function renderUploadValidation(payload) {
  $("#uploadStatus").textContent = readinessText(payload.readiness);
  $("#uploadRows").textContent = formatNumber(payload.rows);
  $("#uploadCoverage").textContent = formatPercent(payload.feature_coverage);
  $("#uploadMissing").textContent = formatNumber(payload.missing_feature_count);
  $("#uploadRecommendations").innerHTML = (payload.recommendations || [])
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join("");

  const columns = payload.preview_columns || [];
  if (!columns.length || !payload.preview_rows?.length) {
    $("#uploadPreviewHead").innerHTML = `<tr><th>Preview</th></tr>`;
    $("#uploadPreview").innerHTML = `<tr><td class="empty-row">No upload preview</td></tr>`;
    return;
  }

  $("#uploadPreviewHead").innerHTML = `<tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
  $("#uploadPreview").innerHTML = payload.preview_rows
    .map(
      (row) => `
        <tr>
          ${columns.map((column) => `<td>${escapeHtml(row[column] ?? "")}</td>`).join("")}
        </tr>
      `,
    )
    .join("");
}

function updateSelectedCsvName() {
  const file = $("#csvFile").files[0];
  $("#uploadFileName").textContent = file ? file.name : "No file selected";
}

async function validateCsv() {
  const file = $("#csvFile").files[0];
  if (!file) {
    toast("Choose a CSV file");
    return;
  }

  const body = new FormData();
  body.append("file", file);
  const button = $("#validateCsvButton");
  button.disabled = true;
  button.textContent = "Validating...";

  try {
    const payload = await api("/api/validate-csv", {
      method: "POST",
      body,
    });
    renderUploadValidation(payload);
    toast(`Validated ${payload.filename}`, "ok");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.innerHTML = `<svg viewBox="0 0 24 24"><path d="m5 12 4 4L19 6" /></svg>Validate`;
  }
}

async function submitBatch(event) {
  event.preventDefault();
  const file = $("#csvFile").files[0];
  if (!file) {
    toast("Choose a CSV file");
    return;
  }

  const body = new FormData();
  body.append("file", file);

  try {
    const payload = await api("/api/predict-csv", {
      method: "POST",
      body,
    });
    renderBatch(payload);
  } catch (error) {
    toast(error.message);
  }
}

function setCheckboxes(name, predicate) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((input) => {
    input.checked = predicate(input);
  });
  updateSuggestedOutputName(false);
}

function setGeneratorBusy(isBusy) {
  const button = $("#generateDatasetButton");
  button.disabled = isBusy;
  if (isBusy) {
    button.textContent = "Generating...";
    $("#generatorStatus").textContent = "Running";
  }
}

function enableDownload(anchor, url) {
  anchor.href = url;
  anchor.classList.remove("disabled");
  anchor.setAttribute("aria-disabled", "false");
}

function renderGeneratorQuality(summary) {
  const quality = summary.quality || {};
  $("#qualitySchema").textContent = quality.schema_match === true ? "Matched" : "Review";
  $("#qualityMissing").textContent = formatNumber(quality.missing_cells);
  $("#qualityDuplicates").textContent = formatNumber(quality.duplicate_rows);
  $("#qualityExact").textContent = formatNumber(quality.exact_real_row_matches);
}

async function generateDataset() {
  const payload = generatorPayload();
  if (!payload.labels.length) {
    toast("Select at least one label");
    return;
  }
  if (!payload.categories.length) {
    toast("Select at least one category");
    return;
  }

  setGeneratorBusy(true);
  try {
    const result = await api("/api/generate-synthetic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("#generatorStatus").textContent = "Complete";
    $("#generatedRows").textContent = formatNumber(result.summary.rows_generated);
    $("#generatedFile").textContent = result.output_file;
    renderGeneratorQuality(result.summary);
    enableDownload($("#downloadSynthetic"), result.download_url);
    enableDownload($("#downloadSummary"), result.summary_url);
    loadArtifacts();
    toast(`Generated ${formatNumber(result.summary.rows_generated)} synthetic rows`, "ok");
  } catch (error) {
    $("#generatorStatus").textContent = "Failed";
    if (error.message && error.message.includes("already running")) {
      toast("Generation already running - please wait for it to finish", "info");
    } else {
      toast(error.message, "error");
    }
  } finally {
    $("#generateDatasetButton").disabled = false;
    $("#generateDatasetButton").innerHTML = `
      <svg viewBox="0 0 24 24"><path d="M12 3v12m0 0 5-5m-5 5-5-5M5 21h14" /></svg>
      Generate dataset
    `;
  }
}

function bindEvents() {
  window.addEventListener("hashchange", () => activateView());
  document.querySelectorAll("[data-view-link]").forEach((link) => {
    link.addEventListener("click", () => activateView(link.dataset.viewLink));
  });
  $("#refreshButton").addEventListener("click", refreshDashboard);
  $("#sampleNormal").addEventListener("click", () => loadSample("normal"));
  $("#sampleAttack").addEventListener("click", () => loadSample("attack"));
  $("#noiseScale").addEventListener("input", (event) => {
    $("#noiseValue").textContent = Number(event.target.value).toFixed(2);
  });
  $("#selectLabelsAll").addEventListener("click", () => setCheckboxes("label_filter", () => true));
  $("#selectLabelsAttack").addEventListener("click", () =>
    setCheckboxes("label_filter", (input) => input.value === "1"),
  );
  $("#selectAllCategories").addEventListener("click", () => setCheckboxes("category", () => true));
  $("#selectAttackCategories").addEventListener("click", () =>
    setCheckboxes("category", (input) => input.dataset.label !== "0"),
  );
  $("#clearCategories").addEventListener("click", () => setCheckboxes("category", () => false));
  $("#selectAllSources").addEventListener("click", () => setCheckboxes("dataset_source", () => true));
  $("#clearSources").addEventListener("click", () => setCheckboxes("dataset_source", () => false));
  $("#selectAllEnvironments").addEventListener("click", () => setCheckboxes("environment_type", () => true));
  $("#clearEnvironments").addEventListener("click", () => setCheckboxes("environment_type", () => false));
  $("#selectAllSubcategories").addEventListener("click", () => setCheckboxes("subcategory", () => true));
  $("#clearSubcategories").addEventListener("click", () => setCheckboxes("subcategory", () => false));
  $("#generateDatasetButton").addEventListener("click", generateDataset);
  $("#validateCsvButton").addEventListener("click", validateCsv);
  $("#csvFile").addEventListener("change", updateSelectedCsvName);
  $("#suggestNameButton").addEventListener("click", () => updateSuggestedOutputName(true));
  $("#outputNameInput").addEventListener("input", () => {
    state.outputNameTouched = true;
  });
  $("#generatorForm").addEventListener("change", (event) => {
    if (event.target.name !== "output_name") updateSuggestedOutputName(false);
  });
  $("#generatorForm").addEventListener("input", (event) => {
    if (event.target.name !== "output_name") updateSuggestedOutputName(false);
  });
  $("#predictionForm").addEventListener("input", () => {
    if (state.sampleFeatures) state.sampleDirty = true;
  });
  $("#predictionForm").addEventListener("submit", submitPrediction);
  $("#batchForm").addEventListener("submit", submitBatch);
  document.querySelectorAll(".metrics-sort-th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sortKey;
      if (metricsSort.key === key) {
        metricsSort.dir = metricsSort.dir === "asc" ? "desc" : "asc";
      } else {
        metricsSort = { key, dir: key === "attack_category" ? "asc" : "desc" };
      }
      renderCategoryMetricsTable();
    });
  });
  const reloadBtn = $("#reloadModelButton");
  if (reloadBtn) reloadBtn.addEventListener("click", reloadModel);

  //  Explain tab 
  $("#explainSampleNormal").addEventListener("click", async () => {
    try {
      const payload = await api("/api/sample?kind=normal");
      fillExplainForm(payload.features);
    } catch (e) { toast(e.message, "error"); }
  });
  $("#explainSampleAttack").addEventListener("click", async () => {
    try {
      const payload = await api("/api/sample?kind=attack");
      fillExplainForm(payload.features);
    } catch (e) { toast(e.message, "error"); }
  });
  $("#explainForm").addEventListener("submit", submitExplain);

  //  Drift tab 
  $("#driftForm").addEventListener("submit", submitDrift);
}

//  Explain helpers 

function fillExplainForm(features) {
  const form = $("#explainForm");
  for (const [key, value] of Object.entries(features)) {
    const input = form.querySelector(`[name="${key}"]`);
    if (input) input.value = value;
  }
}

function renderSHAPBars(containerId, attributions, colorClass) {
  const container = $(containerId);
  if (!attributions.length) {
    container.innerHTML = `<div class="empty-row">None</div>`;
    return;
  }
  const maxAbs = Math.max(...attributions.map((a) => Math.abs(a.shap_value)), 0.001);
  container.innerHTML = attributions
    .map((a) => {
      const pct = Math.max(4, (Math.abs(a.shap_value) / maxAbs) * 100);
      const sign = a.shap_value > 0 ? "+" : "";
      return `
        <div class="feature-row">
          <span>${escapeHtml(a.feature)}</span>
          <div class="bar-track" style="flex:1;margin:0 8px">
            <div class="bar-fill ${colorClass}" style="width:${pct.toFixed(1)}%"></div>
          </div>
          <strong>${sign}${a.shap_value.toFixed(4)}</strong>
        </div>
      `;
    })
    .join("");
}

async function submitExplain(event) {
  event.preventDefault();
  const data = new FormData($("#explainForm"));
  const features = {};
  for (const [key, value] of data.entries()) {
    if (value === "") continue;
    if (key === "protocol" || key === "service") { features[key] = value; }
    else { const n = Number(value); if (!Number.isNaN(n)) features[key] = n; }
  }
  const btn = $("#explainForm").querySelector("button[type=submit]");
  btn.disabled = true;
  btn.textContent = "Explaining...";
  try {
    const result = await api("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features }),
    });
    const exp = result.explanation;
    $("#explainProb").textContent = formatPercent(result.attack_probability);
    $("#explainBase").textContent = formatPercent(exp.base_value);
    $("#explainRisk").textContent = result.risk;
    $("#explainTimestamp").textContent = new Date().toLocaleTimeString();
    const chip = $("#explainVerdictChip");
    chip.textContent = result.risk;
    chip.className = `risk-chip ${result.risk.toLowerCase()}`;
    $("#explainVerdictText").textContent = result.verdict;
    renderSHAPBars("#explainAttackBars", exp.top_attack_drivers, "bar-fill-danger");
    renderSHAPBars("#explainNormalBars", exp.top_normal_drivers.map(a => ({...a, shap_value: -a.shap_value})), "bar-fill-ok");
    toast(`Explained: ${result.verdict} (${formatPercent(result.attack_probability)})`, result.verdict === "Attack" ? "error" : "ok");
  } catch (error) {
    if (error.message.includes("SHAP not installed")) {
      toast("Install SHAP first: pip install shap", "error");
    } else {
      toast(error.message, "error");
    }
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="m5 12 5 5L20 7" /></svg> Explain prediction`;
  }
}

//  Drift helpers 

function renderDriftFeatureList(containerId, features, emptyMsg) {
  const container = $(containerId);
  if (!features.length) {
    container.innerHTML = `<div class="empty-row">${emptyMsg}</div>`;
    return;
  }
  const maxPsi = Math.max(...features.map((f) => f.psi || 0), 0.001);
  container.innerHTML = features
    .map((f) => {
      const pct = Math.max(4, ((f.psi || 0) / maxPsi) * 100);
      const sevClass = f.severity === "high" ? "rate-danger" : f.severity === "moderate" ? "rate-warn" : "rate-ok";
      return `
        <div class="feature-row">
          <span>${escapeHtml(f.feature)}</span>
          <div class="bar-track" style="flex:1;margin:0 8px">
            <div class="bar-fill ${sevClass}" style="width:${pct.toFixed(1)}%"></div>
          </div>
          <strong class="${sevClass}">${f.psi != null ? f.psi.toFixed(4) : "--"}</strong>
        </div>
      `;
    })
    .join("");
}

async function submitDrift(event) {
  event.preventDefault();
  const file = $("#driftFile").files[0];
  if (!file) { toast("Choose a CSV file"); return; }
  const body = new FormData();
  body.append("file", file);
  const btn = $("#driftForm").querySelector("button[type=submit]");
  btn.disabled = true;
  btn.textContent = "Analyzing...";
  try {
    const result = await api("/api/drift", { method: "POST", body });
    const sevClass = result.overall_severity === "high" ? "rate-danger" : result.overall_severity === "moderate" ? "rate-warn" : "rate-ok";
    $("#driftSeverity").className = sevClass;
    $("#driftSeverity").textContent = result.overall_severity.charAt(0).toUpperCase() + result.overall_severity.slice(1);
    $("#driftFeatureCount").textContent = formatNumber(result.features_analysed);
    $("#driftDriftedCount").textContent = formatNumber(result.drifted_features);
    $("#driftRetrain").textContent = result.retraining_recommended ? "Yes" : "No";
    $("#driftRetrain").className = result.retraining_recommended ? "rate-danger" : "rate-ok";
    renderDriftFeatureList("#driftHighTable", result.high_drift_features, "No high-drift features");
    renderDriftFeatureList("#driftModerateTable", result.moderate_drift_features, "No moderate-drift features");
    toast(`Drift: ${result.overall_severity} - ${result.drifted_features} features drifted`, result.retraining_recommended ? "error" : "ok");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24"><path d="M3 18l4-8 4 4 4-6 4 4" /></svg> Analyze drift`;
  }
}

document.querySelectorAll(".metric").forEach((m) => m.classList.add("loading"));
activateView();
bindEvents();
refreshDashboard();
loadGeneratorOptions();
