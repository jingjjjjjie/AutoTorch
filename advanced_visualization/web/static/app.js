import { getSchema, getSources } from "./api.js";
import { bindProjection } from "./projection.js";
import { bindReview, loadReview } from "./review.js";
import { setOptions, state } from "./state.js";

const byId = id => document.getElementById(id);
const setBusy = busy => byId("loading").classList.toggle("hidden", !busy);
const showError = message => { byId("error-banner").textContent = message; byId("error-banner").classList.toggle("hidden", !message); };

function initializeCategoricalFilters(schema, preserve = false) {
  const previousActive = state.activeFilterColumns;
  const previousSelections = state.filterSelections;
  const defaults = preserve
    ? previousActive.filter(column => column in schema.categories)
    : schema.default_filter_columns.length
      ? schema.default_filter_columns
      : [schema.defaults.subclass_column].filter(column => column in schema.categories);
  state.activeFilterColumns = [...new Set(defaults)];
  state.filterSelections = new Map(Object.entries(schema.categories).map(([column, values]) => {
    if (!preserve || !previousSelections.has(column)) return [column, new Set(values)];
    const previous = previousSelections.get(column);
    const compatible = values.filter(value => previous.has(value));
    return [column, new Set(compatible.length || previous.size === 0 ? compatible : values)];
  }));
  renderCategoricalFilters();
}

function updateFilterCount() {
  let count = 0;
  for (const column of state.activeFilterColumns) {
    if ((state.filterSelections.get(column)?.size || 0) !== state.schema.categories[column].length) count += 1;
  }
  byId("active-filter-count").textContent = String(count);
}

function filterGroup(column, index) {
  const values = state.schema.categories[column];
  const selected = state.filterSelections.get(column) || new Set(values);
  const details = document.createElement("details");
  details.className = "filter-group";
  details.open = index === 0;
  const summary = document.createElement("summary");
  const label = document.createElement("span");
  label.textContent = column;
  const count = document.createElement("span");
  count.className = "filter-selection-count";
  count.textContent = `${selected.size}/${values.length}`;
  summary.append(label, count);
  details.append(summary);

  const body = document.createElement("div");
  body.className = "filter-group-body";
  if (values.length > 10) {
    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "Find value";
    search.addEventListener("input", () => {
      const query = search.value.trim().toLowerCase();
      for (const option of body.querySelectorAll(".filter-option")) {
        option.classList.toggle("hidden", !option.textContent.toLowerCase().includes(query));
      }
    });
    body.append(search);
  }
  const actions = document.createElement("div");
  actions.className = "filter-actions";
  for (const [text, valuesToSelect] of [["All", values], ["None", []]]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = text;
    button.addEventListener("click", () => {
      state.filterSelections.set(column, new Set(valuesToSelect));
      renderCategoricalFilters();
    });
    actions.append(button);
  }
  const remove = document.createElement("button");
  remove.type = "button";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => {
    state.activeFilterColumns = state.activeFilterColumns.filter(item => item !== column);
    renderCategoricalFilters();
  });
  actions.append(remove);
  body.append(actions);

  const options = document.createElement("div");
  options.className = "filter-options";
  for (const value of values) {
    const option = document.createElement("label");
    option.className = "filter-option";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = selected.has(value);
    checkbox.addEventListener("change", () => {
      const next = new Set(state.filterSelections.get(column) || []);
      checkbox.checked ? next.add(value) : next.delete(value);
      state.filterSelections.set(column, next);
      count.textContent = `${next.size}/${values.length}`;
      updateFilterCount();
    });
    const text = document.createElement("span");
    text.textContent = value;
    option.append(checkbox, text);
    options.append(option);
  }
  body.append(options);
  details.append(body);
  return details;
}

function renderCategoricalFilters() {
  const container = byId("categorical-filters");
  container.replaceChildren(...state.activeFilterColumns.map(filterGroup));
  const available = Object.keys(state.schema.categories).filter(
    column => !state.activeFilterColumns.includes(column)
  );
  setOptions(byId("add-filter-column"), available, "", true);
  byId("add-filter-column").options[0].textContent = available.length ? "Choose column" : "All filters added";
  byId("add-filter-column").disabled = !available.length;
  updateFilterCount();
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
  setOptions(byId("subclass-column"), schema.categorical_columns, defaults.subclass_column);
  setOptions(byId("truth-column"), schema.columns, defaults.truth_column);
  setOptions(byId("prediction-column"), schema.numeric_columns, defaults.prediction_column);
  setOptions(byId("color-column"), schema.categorical_columns, defaults.subclass_column);
  setOptions(byId("feature-image-column"), schema.image_columns, defaults.image_column, true, imageLabels);
  setOptions(byId("feature-gradcam-column"), schema.gradcam_columns, schema.gradcam_columns[0] || "", true, imageLabels);
  if (schema.prepared_gradcam_methods.length && !schema.gradcam_columns.length) {
    byId("feature-gradcam-column").options[0].textContent = "Auto-detect prepared CAM";
  }
  byId("feature-count").textContent = schema.feature_columns.length
    ? `${schema.feature_columns.length.toLocaleString()} feature columns detected.`
    : "No numeric feature columns were detected.";
  byId("feature-controls").querySelector("button.primary").disabled = !schema.feature_columns.length;
  initializeCategoricalFilters(schema, preserveFilters);
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
    state.projection = null;
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
      button.textContent = open ? "Close" : "Filters";
    } else {
      shell.classList.toggle("sidebar-collapsed", !open);
      sidebar.classList.remove("mobile-open");
      button.textContent = open ? "Hide filters" : "Show filters";
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
  for (const form of [byId("review-controls"), byId("feature-controls")]) {
    form.addEventListener("submit", () => {
      if (isMobile()) setOpen(false);
    });
  }
}

async function start() {
  bindReview(setBusy, showError);
  bindProjection(setBusy, showError);
  bindNavigation();
  bindSidebar();
  byId("add-filter-column").addEventListener("change", event => {
    if (!event.target.value) return;
    state.activeFilterColumns.push(event.target.value);
    renderCategoricalFilters();
  });
  byId("reset-filters").addEventListener("click", () => initializeCategoricalFilters(state.schema, false));
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
