export const state = {
  source: null,
  schema: null,
  page: 1,
  pages: 1,
  imageMode: "original",
  projection: null,
};

export function setOptions(select, values, preferred = "", allowNone = true) {
  select.replaceChildren();
  if (allowNone) select.add(new Option("None", ""));
  for (const value of values) select.add(new Option(value, value));
  select.value = values.includes(preferred) ? preferred : (allowNone ? "" : values[0] || "");
}

export function selectedValues(select) {
  return [...select.selectedOptions].map(option => option.value);
}

