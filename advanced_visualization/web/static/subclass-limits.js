import { categoricalFilters, state } from "./state.js";

const byId = id => document.getElementById(id);
const overridesByScope = new Map();
const availabilityByContext = new Map();
let bound = false;
let notifyChange = () => {};

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function maximumRows() {
  const value = Number(byId("max-rows").value);
  return clamp(Number.isFinite(value) ? Math.round(value) : 5000, 3, 50000);
}

function masterLimit() {
  return clamp(Number(byId("class-limit").value) || maximumRows(), 1, maximumRows());
}

function selectedLabels() {
  const column = byId("color-column").value;
  const labels = state.schema?.categories[column] || [];
  if (!state.activeFilterColumns.includes(column)) return labels;
  const selected = state.filterSelections.get(column) || new Set(labels);
  return labels.filter(label => selected.has(label));
}

function currentOverrides() {
  const key = JSON.stringify([state.source?.id || "", byId("color-column").value]);
  if (!overridesByScope.has(key)) overridesByScope.set(key, new Map());
  return overridesByScope.get(key);
}

function availabilityKey() {
  return JSON.stringify([
    state.source?.id || "",
    byId("color-column").value,
    categoricalFilters(),
  ]);
}

function currentAvailability() {
  return availabilityByContext.get(availabilityKey()) || null;
}

function updateMasterOutput() {
  byId("class-limit-value").value = masterLimit().toLocaleString();
}

function applyMasterToVisibleLimits() {
  for (const row of byId("subclass-limit-list").querySelectorAll(".subclass-limit")) {
    const range = row.querySelector('input[type="range"]');
    const value = Math.min(masterLimit(), Number(range.max));
    range.value = String(value);
    row.querySelector('input[type="number"]').value = String(value);
  }
}

function emptyLimitMessage(message) {
  const empty = document.createElement("div");
  empty.className = "subclass-limit-empty";
  empty.textContent = message;
  return empty;
}

export function refreshSubclassLimits() {
  const panel = byId("subclass-limits-panel");
  const list = byId("subclass-limit-list");
  const column = byId("color-column").value;
  const labels = selectedLabels();
  const overrides = currentOverrides();
  const availability = currentAvailability();
  const maximum = maximumRows();
  const inherited = masterLimit();

  panel.classList.toggle("hidden", !column || !state.schema);
  byId("subclass-limit-count").textContent = `${labels.length} selected`;
  list.replaceChildren();
  if (!column || !state.schema) return;
  if (!labels.length) {
    const knownLabels = state.schema.categories[column];
    list.append(emptyLimitMessage(
      knownLabels ? "No subclasses selected." : "Individual limits are unavailable for this column."
    ));
    return;
  }

  for (const label of labels) {
    const row = document.createElement("label");
    row.className = "subclass-limit";
    const heading = document.createElement("span");
    heading.className = "subclass-limit-heading";
    const name = document.createElement("span");
    name.className = "subclass-limit-name";
    name.textContent = label;
    name.title = label;
    const number = document.createElement("input");
    number.type = "number";
    number.className = "subclass-limit-number";
    number.min = "1";
    number.step = "1";
    number.setAttribute("aria-label", `Exact maximum rows for ${label}`);
    const count = availability
      ? availability.get(label) || { available: 0, displayed: 0 }
      : null;
    const rowMaximum = Math.max(1, Math.min(maximum, count?.available ?? maximum));
    number.max = String(rowMaximum);
    const stored = overrides.get(label);
    const value = clamp(stored ?? inherited, 1, rowMaximum);
    number.value = String(value);
    heading.append(name, number);

    const input = document.createElement("input");
    input.type = "range";
    input.min = "1";
    input.max = String(rowMaximum);
    input.step = "1";
    input.value = String(value);
    input.setAttribute("aria-label", `Maximum rows for ${label}`);
    const setLimit = nextValue => {
      if (nextValue === masterLimit()) overrides.delete(label);
      else overrides.set(label, nextValue);
      input.value = String(nextValue);
      number.value = String(nextValue);
      notifyChange();
    };
    input.addEventListener("input", () => setLimit(Number(input.value)));
    number.addEventListener("change", () => {
      const requested = Number(number.value);
      const nextValue = clamp(Number.isFinite(requested) ? Math.round(requested) : value, 1, rowMaximum);
      setLimit(nextValue);
    });
    row.append(heading, input);
    if (count) {
      const metadata = document.createElement("span");
      metadata.className = "subclass-limit-meta";
      metadata.textContent = `${count.displayed.toLocaleString()} projected / ${count.available.toLocaleString()} available`;
      row.append(metadata);
    }
    if (count?.available === 0) {
      input.disabled = true;
      number.disabled = true;
    }
    list.append(row);
  }
}

function syncMasterLimit() {
  const input = byId("class-limit");
  const previousMaximum = Number(input.max) || 5000;
  const nextMaximum = maximumRows();
  const wasUnlimited = Number(input.value) >= previousMaximum;
  input.max = String(nextMaximum);
  input.value = String(wasUnlimited ? nextMaximum : Math.min(Number(input.value), nextMaximum));
  updateMasterOutput();
  refreshSubclassLimits();
}

export function projectionClassLimits() {
  const column = byId("color-column").value;
  const maximum = maximumRows();
  const shared = masterLimit();
  const overrides = currentOverrides();
  const availability = currentAvailability();
  const entries = [];
  if (column) {
    for (const label of selectedLabels()) {
      if (!overrides.has(label)) continue;
      const available = availability
        ? availability.get(label)?.available ?? 0
        : undefined;
      const labelMaximum = Math.max(1, Math.min(maximum, available ?? maximum));
      const value = clamp(overrides.get(label), 1, labelMaximum);
      if (value !== shared) entries.push([label, value]);
    }
  }
  return {
    max_rows_per_class: column && shared < maximum ? shared : null,
    max_rows_by_class: Object.fromEntries(entries),
  };
}

export function updateSubclassAvailability(classCounts) {
  availabilityByContext.set(
    availabilityKey(),
    new Map((classCounts || []).map(count => [count.label, count])),
  );
  refreshSubclassLimits();
}

export function bindSubclassLimits(onChange) {
  if (bound) return;
  bound = true;
  notifyChange = onChange || (() => {});
  byId("max-rows").addEventListener("input", () => {
    syncMasterLimit();
    notifyChange();
  });
  byId("class-limit").addEventListener("input", () => {
    currentOverrides().clear();
    updateMasterOutput();
    applyMasterToVisibleLimits();
    notifyChange();
  });
  byId("color-column").addEventListener("change", refreshSubclassLimits);
  syncMasterLimit();
}
