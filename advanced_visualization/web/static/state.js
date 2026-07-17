export const state = {
  sources: [],
  source: null,
  schema: null,
  page: 1,
  pages: 1,
  imageMode: "original",
  projection: null,
  filterSelections: new Map(),
  activeFilterColumns: [],
};

export function setOptions(select, values, preferred = "", allowNone = true, labels = {}) {
  select.replaceChildren();
  if (allowNone) select.add(new Option("None", ""));
  for (const value of values) select.add(new Option(labels[value] || value, value));
  select.value = values.includes(preferred) ? preferred : (allowNone ? "" : values[0] || "");
}

export function categoricalFilters() {
  const filters = {};
  for (const column of state.activeFilterColumns) {
    const allValues = state.schema.categories[column] || [];
    const selected = state.filterSelections.get(column) || new Set(allValues);
    if (selected.size !== allValues.length) filters[column] = [...selected];
  }
  return filters;
}
