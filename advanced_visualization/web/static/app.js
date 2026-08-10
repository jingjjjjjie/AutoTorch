import { getSchema, getSources } from "./api.js";
import { bindAnalysis, configureAnalysis, loadAnalysis, selectAnalysisSource } from "./analysis.js";
import { initializeCategoricalFilters, renderCategoricalFilters } from "./categorical-filters.js";
import { bindComparison, configureComparison, loadComparison } from "./comparison.js";
import { byId } from "./dom.js";
import { bindImageViewer } from "./image-viewer.js";
import { bindLiveInference, configureLiveInference } from "./live-inference.js";
import { bindProjection, clearProjection, loadProjection } from "./projection.js";
import { bindReview, loadReview } from "./review.js";
import { setOptions, state } from "./state.js";

const setBusy = busy => byId("loading").classList.toggle("hidden", !busy);
const showError = message => { byId("error-banner").textContent = message; byId("error-banner").classList.toggle("hidden", !message); };


function applyReviewPreset(schema) {
  const preset = schema.review_preset || {};
  if (!Object.keys(preset).length) return;

  const selectValue = (id, value) => {
    const element = byId(id);
    if (value != null && [...element.options].some(option => option.value === value)) {
      element.value = value;
    }
  };
  selectValue("image-column", preset.image_column);
  selectValue("prediction-column", preset.prediction_column);
  selectValue("truth-column", preset.truth_column);
  selectValue("subclass-column", preset.subclass_column);
  selectValue("truth-rows", preset.truth_rows);
  selectValue("failure-mode", preset.failure_mode);
  selectValue("projection-method", preset.feature_projection_method);
  selectValue("color-column", preset.feature_color_column);
  selectValue("feature-image-column", preset.feature_image_column);

  if (preset.feature_max_rows != null) {
    byId("max-rows").value = String(preset.feature_max_rows);
  }
  if (preset.feature_scale != null) {
    byId("scale-features").checked = Boolean(preset.feature_scale);
  }
  byId("projection-method").dispatchEvent(new Event("change"));

  if (preset.threshold != null) {
    byId("threshold").value = String(preset.threshold);
    byId("threshold-value").value = Number(preset.threshold).toFixed(2);
  }
  if (preset.image_mode) {
    state.imageMode = preset.image_mode;
    for (const button of byId("image-mode").querySelectorAll("button")) {
      button.classList.toggle("active", button.dataset.value === preset.image_mode);
    }
  }

  for (const [column, rule] of Object.entries(preset.categorical_filters || {})) {
    const available = schema.categories[column];
    if (!available) continue;
    if (!state.activeFilterColumns.includes(column)) state.activeFilterColumns.push(column);
    const excluded = new Set((rule.exclude || []).map(String));
    const included = rule.include ? new Set(rule.include.map(String)) : null;
    state.filterSelections.set(column, new Set(available.filter(value =>
      !excluded.has(String(value)) && (!included || included.has(String(value)))
    )));
  }
  renderCategoricalFilters();
}


function populateControls(schema, preserveFilters = false) {
  const defaults = schema.defaults;
  const imageLabels = Object.fromEntries(schema.image_columns.map(column => {
    const availability = schema.image_availability[column] || 0;
    return [column, availability > 0 ? column : `${column} (files unavailable)`];
  }));
  setOptions(byId("item-column"), schema.columns, defaults.item_id_column, false);
  setOptions(byId("image-column"), schema.image_columns, defaults.image_column, true, imageLabels);
  setOptions(byId("gradcam-column"), schema.gradcam_columns, schema.gradcam_columns[0] || "", true, imageLabels);
  if (schema.prepared_gradcam_methods.length && !schema.gradcam_columns.length) {
    byId("gradcam-column").options[0].textContent = "Auto-detect prepared CAM";
  }
  setOptions(
    byId("gradcam-layer"),
    schema.prepared_gradcam_layers || [],
    schema.default_gradcam_layer || "",
    false,
    schema.prepared_gradcam_layer_labels || {},
  );
  const hasIndividualCam = Boolean(
    (schema.gradcam_columns || []).length
    || (schema.prepared_gradcam_layers || []).length
  );
  const individualCamModes = [
    "gradcam",
    "both",
    "original_genuine",
    "side_by_side",
    "original_both",
  ];
  for (const value of individualCamModes) {
    byId("image-mode")
      .querySelector(`[data-value="${value}"]`)
      .classList.toggle("hidden", !hasIndividualCam);
  }
  for (const id of ["gradcam-column", "gradcam-method", "gradcam-layer", "gradcam-target"]) {
    byId(id).closest("label").classList.toggle("hidden", !hasIndividualCam);
  }
  const montageButton = byId("image-mode").querySelector('[data-value="montage"]');
  montageButton.classList.toggle("hidden", !schema.gradcam_montage_column);
  const invalidMode = (
    (!schema.gradcam_montage_column && state.imageMode === "montage")
    || (!hasIndividualCam && individualCamModes.includes(state.imageMode))
  );
  if (invalidMode) {
    state.imageMode = schema.gradcam_montage_column ? "montage" : "original";
    for (const button of byId("image-mode").querySelectorAll("button")) {
      button.classList.toggle("active", button.dataset.value === state.imageMode);
    }
  }
  setOptions(byId("subclass-column"), schema.categorical_columns, defaults.subclass_column);
  setOptions(byId("truth-column"), schema.columns, defaults.truth_column);
  setOptions(byId("prediction-column"), schema.numeric_columns, defaults.prediction_column);
  setOptions(byId("color-column"), schema.categorical_columns, defaults.subclass_column);
  setOptions(
    byId("feature-color-score"),
    schema.numeric_columns,
    defaults.prediction_column,
  );
  setOptions(byId("feature-image-column"), schema.image_columns, defaults.image_column, true, imageLabels);
  setOptions(byId("feature-gradcam-column"), schema.gradcam_columns, schema.gradcam_columns[0] || "", true, imageLabels);
  if (schema.prepared_gradcam_methods.length && !schema.gradcam_columns.length) {
    byId("feature-gradcam-column").options[0].textContent = "Auto-detect prepared CAM";
  }
  setOptions(
    byId("feature-gradcam-layer"),
    schema.prepared_gradcam_layers || [],
    schema.default_gradcam_layer || "",
    false,
    schema.prepared_gradcam_layer_labels || {},
  );
  byId("feature-count").textContent = schema.feature_columns.length
    ? `${schema.feature_columns.length.toLocaleString()} feature columns detected.`
    : "No numeric feature columns were detected.";
  byId("feature-controls").querySelector("button.primary").disabled = !schema.feature_columns.length;
  initializeCategoricalFilters(schema, preserveFilters);
  applyReviewPreset(schema);
}

async function selectSource(sourceId) {
  setBusy(true);
  try {
    const previousSource = state.source;
    const nextSource = state.sources.find(source => source.id === sourceId);
    const preserveFilters = Boolean(
      previousSource?.model_key && previousSource.model_key === nextSource?.model_key
    );
    state.source = nextSource;
    state.schema = await getSchema(sourceId);
    state.page = 1;
    clearProjection();
    populateControls(state.schema, preserveFilters);
    byId("source-status").textContent = `${state.source.label} | ${state.schema.source.rows.toLocaleString()} rows`;
    if (!byId("review-view").classList.contains("hidden")) {
      await loadReview(setBusy, showError);
    }
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

function activeView() {
  return document.querySelector(".tab.active")?.dataset.view || "review";
}

function activateView(view, load = true) {
  const supported = new Set(["review", "features", "comparison", "analysis", "live"]);
  const selected = supported.has(view) ? view : "review";
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.view === selected);
  }
  for (const name of supported) {
    byId(`${name === "features" ? "feature" : name}-controls`).classList.toggle("hidden", name !== selected);
    byId(`${name}-view`).classList.toggle("hidden", name !== selected);
  }
  const isLive = selected === "live";
  byId("source-field").classList.toggle("hidden", isLive);
  document.querySelector(".dataset-filters").classList.toggle("hidden", isLive);
  byId("source-status").textContent = isLive
    ? "Live checkpoint inference"
    : `${state.source.label} | ${state.schema.source.rows.toLocaleString()} rows`;
  if (selected === "features" && state.schema && state.source && !state.schema.feature_columns.length) {
    const featureSource = state.sources.find(source =>
      source.model_key === state.source.model_key && source.label.endsWith(" - features")
    );
    if (featureSource) {
      byId("source-select").value = featureSource.id;
      selectSource(featureSource.id);
      return;
    }
  }
  if (!load) return;
  if (selected === "features" && state.schema?.review_preset?.feature_autorun) {
    loadProjection(setBusy, showError);
  }
  if (selected === "review") loadReview(setBusy, showError);
  if (selected === "comparison") loadComparison();
  if (selected === "analysis") {
    selectAnalysisSource(state.source.id)
      .then(loadAnalysis)
      .catch(error => showError(error.message));
  }
}

function bindNavigation() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => activateView(tab.dataset.view));
  }
}

function bindSidebar() {
  const sidebar = byId("sidebar");
  const shell = document.querySelector(".shell");
  const button = byId("filter-toggle");
  const media = window.matchMedia("(max-width: 720px)");
  const isMobile = () => media.matches;
  const setOpen = open => {
    if (isMobile()) {
      sidebar.classList.toggle("mobile-open", open);
      shell.classList.remove("sidebar-collapsed");
      button.textContent = open ? "Close sidebar" : "Open sidebar";
    } else {
      shell.classList.toggle("sidebar-collapsed", !open);
      sidebar.classList.remove("mobile-open");
      button.textContent = open ? "Hide sidebar" : "Show sidebar";
      localStorage.setItem("autotorch-sidebar-open", String(open));
    }
    button.setAttribute("aria-expanded", String(open));
  };
  setOpen(isMobile() ? false : localStorage.getItem("autotorch-sidebar-open") !== "false");
  byId("filter-toggle").addEventListener("click", () => {
    const open = isMobile()
      ? !sidebar.classList.contains("mobile-open")
      : shell.classList.contains("sidebar-collapsed");
    setOpen(open);
  });
  media.addEventListener("change", () => {
    setOpen(isMobile() ? false : localStorage.getItem("autotorch-sidebar-open") !== "false");
  });
  for (const form of [
    byId("review-controls"),
    byId("feature-controls"),
    byId("comparison-controls"),
    byId("analysis-controls"),
    byId("live-controls"),
  ]) {
    form.addEventListener("submit", () => {
      if (isMobile()) setOpen(false);
    });
  }
}

function applyUrlState(parameters) {
  const mappings = {
    item: "item-column",
    truth: "truth-column",
    prediction: "prediction-column",
    image: "image-column",
    gradcam: "gradcam-column",
    target: "gradcam-target",
  };
  for (const [parameter, id] of Object.entries(mappings)) {
    const value = parameters.get(parameter);
    const select = byId(id);
    if (value != null && [...select.options].some(option => option.value === value)) select.value = value;
  }
  if (parameters.has("threshold")) {
    const threshold = Number(parameters.get("threshold"));
    if (Number.isFinite(threshold) && threshold >= 0 && threshold <= 1) {
      byId("threshold").value = String(threshold);
      byId("threshold-value").value = threshold.toFixed(2);
    }
  }
}

async function start() {
  bindReview(setBusy, showError);
  bindProjection(setBusy, showError);
  bindComparison(setBusy, showError);
  bindAnalysis(setBusy, showError);
  bindImageViewer();
  bindLiveInference(setBusy, showError);
  bindNavigation();
  bindSidebar();
  byId("add-filter-column").addEventListener("change", event => {
    if (!event.target.value) return;
    state.activeFilterColumns.push(event.target.value);
    renderCategoricalFilters();
  });
  byId("reset-filters").addEventListener("click", () => initializeCategoricalFilters(state.schema, false));
  byId("source-select").addEventListener("change", event => selectSource(event.target.value));
  setBusy(true);
  try {
    const parameters = new URLSearchParams(window.location.search);
    state.sources = await getSources();
    setOptions(byId("source-select"), state.sources.map(source => source.id), "", false);
    [...byId("source-select").options].forEach((option, index) => { option.textContent = state.sources[index].label; });
    if (!state.sources.length) throw new Error("No configured CSV files are currently available.");
    const requestedSource = parameters.get("source");
    const initialSource = state.sources.find(source => source.id === requestedSource) || state.sources[0];
    byId("source-select").value = initialSource.id;
    await selectSource(initialSource.id);
    await Promise.all([
      configureComparison(state.sources, initialSource.id),
      configureAnalysis(state.sources, initialSource.id),
      configureLiveInference(),
    ]);
    applyUrlState(parameters);
    const requestedView = parameters.get("view") || "review";
    if (requestedView !== "review" || parameters.size) activateView(requestedView);
  } catch (error) {
    showError(error.message);
    byId("source-status").textContent = "No source loaded";
  } finally {
    setBusy(false);
  }
}

start();
