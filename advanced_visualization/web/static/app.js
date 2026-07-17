import { getSchema, getSources } from "./api.js";
import { bindProjection } from "./projection.js";
import { bindReview, loadReview } from "./review.js";
import { setOptions, state } from "./state.js";

const byId = id => document.getElementById(id);
const setBusy = busy => byId("loading").classList.toggle("hidden", !busy);
const showError = message => { byId("error-banner").textContent = message; byId("error-banner").classList.toggle("hidden", !message); };

function updateFilterValues() {
  const column = byId("filter-column").value;
  setOptions(byId("filter-values"), state.schema.categories[column] || [], "", false);
  for (const option of byId("filter-values").options) option.selected = true;
}

function populateControls(schema) {
  const defaults = schema.defaults;
  const imageLabels = Object.fromEntries(schema.image_columns.map(column => {
    const availability = schema.image_availability[column] || 0;
    return [column, availability > 0 ? column : `${column} (files unavailable)`];
  }));
  setOptions(byId("item-column"), schema.columns, defaults.item_id_column, false);
  setOptions(byId("image-column"), schema.image_columns, defaults.image_column, true, imageLabels);
  setOptions(byId("gradcam-column"), schema.gradcam_columns, schema.gradcam_columns[0] || "", true, imageLabels);
  setOptions(byId("subclass-column"), schema.categorical_columns, defaults.subclass_column);
  setOptions(byId("truth-column"), schema.columns, defaults.truth_column);
  setOptions(byId("prediction-column"), schema.numeric_columns, defaults.prediction_column);
  setOptions(byId("filter-column"), Object.keys(schema.categories), "");
  setOptions(byId("color-column"), schema.categorical_columns, defaults.subclass_column);
  setOptions(byId("feature-image-column"), schema.image_columns, defaults.image_column, true, imageLabels);
  byId("feature-count").textContent = schema.feature_columns.length
    ? `${schema.feature_columns.length.toLocaleString()} feature columns detected.`
    : "No numeric feature columns were detected.";
  byId("feature-controls").querySelector("button.primary").disabled = !schema.feature_columns.length;
  updateFilterValues();
}

async function selectSource(sourceId) {
  setBusy(true);
  try {
    state.source = state.sources.find(source => source.id === sourceId);
    state.schema = await getSchema(sourceId);
    state.page = 1;
    state.projection = null;
    populateControls(state.schema);
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

function bindNavigation() {
  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => {
      for (const item of document.querySelectorAll(".tab")) item.classList.toggle("active", item === tab);
      const review = tab.dataset.view === "review";
      byId("review-controls").classList.toggle("hidden", !review);
      byId("feature-controls").classList.toggle("hidden", review);
      byId("review-view").classList.toggle("hidden", !review);
      byId("features-view").classList.toggle("hidden", review);
      if (!review && state.schema && state.source && !state.schema.feature_columns.length) {
        const featureSource = state.sources.find(source =>
          source.model_key === state.source.model_key && source.label.endsWith(" - features")
        );
        if (featureSource) {
          byId("source-select").value = featureSource.id;
          selectSource(featureSource.id);
        }
      }
    });
  }
}

function bindMobileFilters() {
  const sidebar = byId("sidebar");
  byId("filter-toggle").addEventListener("click", () => {
    const open = sidebar.classList.toggle("mobile-open");
    byId("filter-toggle").textContent = open ? "Close" : "Filters";
  });
  for (const form of [byId("review-controls"), byId("feature-controls")]) {
    form.addEventListener("submit", () => {
      if (window.matchMedia("(max-width: 720px)").matches) {
        sidebar.classList.remove("mobile-open");
        byId("filter-toggle").textContent = "Filters";
      }
    });
  }
}

async function start() {
  bindReview(setBusy, showError);
  bindProjection(setBusy, showError);
  bindNavigation();
  bindMobileFilters();
  byId("filter-column").addEventListener("change", updateFilterValues);
  byId("source-select").addEventListener("change", event => selectSource(event.target.value));
  byId("close-dialog").addEventListener("click", () => byId("image-dialog").close());
  setBusy(true);
  try {
    state.sources = await getSources();
    setOptions(byId("source-select"), state.sources.map(source => source.id), "", false);
    [...byId("source-select").options].forEach((option, index) => { option.textContent = state.sources[index].label; });
    if (!state.sources.length) throw new Error("No configured CSV files are currently available.");
    await selectSource(state.sources[0].id);
  } catch (error) {
    showError(error.message);
    byId("source-status").textContent = "No source loaded";
  } finally {
    setBusy(false);
  }
}

start();
