import { byId } from "./dom.js";
import { setOptions, state } from "./state.js";
import { refreshSubclassLimits } from "./subclass-limits.js";

export function initializeCategoricalFilters(schema, preserve = false) {
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
      refreshSubclassLimits();
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

export function renderCategoricalFilters() {
  const container = byId("categorical-filters");
  container.replaceChildren(...state.activeFilterColumns.map(filterGroup));
  const available = Object.keys(state.schema.categories).filter(
    column => !state.activeFilterColumns.includes(column)
  );
  setOptions(byId("add-filter-column"), available, "", true);
  byId("add-filter-column").options[0].textContent = available.length ? "Choose column" : "All filters added";
  byId("add-filter-column").disabled = !available.length;
  updateFilterCount();
  refreshSubclassLimits();
}

