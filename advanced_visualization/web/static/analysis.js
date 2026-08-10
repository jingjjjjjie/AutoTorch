// Notebook-equivalent original/crop/merged analysis view.

import { getAnalysis, getSchema } from "./api.js";
import {
  bindGalleryImages,
  byId,
  escapeText,
  formatPercent,
  formatScore,
  imagePane,
  metric,
} from "./dom.js";
import { openImageViewer } from "./image-viewer.js";
import { categoricalFilters, setOptions, state } from "./state.js";

const NOTEBOOK_ORI = "Ex8point2res1024largerbs_square_exp_pred_ori";
const NOTEBOOK_CROP = "exp8point4_pred_crop";
const NOTEBOOK_EXCLUSIONS = [
  "Aug_2025_myID_datacollection",
  "Feb_2026_mixed_RoutineAnnotation",
];

let schema = null;
let page = 1;
let pages = 1;
let drilldown = null;
let setBusyCallback = () => {};
let showErrorCallback = () => {};

function sourceSelectOptions(select, sources, preferred) {
  setOptions(select, sources.map(source => source.id), preferred, false);
  [...select.options].forEach((option, index) => { option.textContent = sources[index].label; });
}

function findColumn(columns, exact, pattern) {
  return columns.find(column => column === exact)
    || columns.find(column => pattern.test(column))
    || "";
}

function populateControls(nextSchema) {
  const columns = nextSchema.columns;
  setOptions(byId("analysis-item"), columns, nextSchema.defaults.item_id_column, false);
  setOptions(byId("analysis-truth"), columns, nextSchema.defaults.truth_column || "label", false);
  setOptions(
    byId("analysis-ori-prediction"),
    nextSchema.numeric_columns,
    findColumn(nextSchema.numeric_columns, NOTEBOOK_ORI, /(?:pred_?ori|ori.*pred)/i),
    false,
  );
  setOptions(
    byId("analysis-crop-prediction"),
    nextSchema.numeric_columns,
    findColumn(nextSchema.numeric_columns, NOTEBOOK_CROP, /(?:pred_?crop|crop.*pred)/i),
    false,
  );
  setOptions(
    byId("analysis-original-image"),
    nextSchema.image_columns,
    findColumn(nextSchema.image_columns, "absolute_ori_path", /(?:absolute.*ori|ori.*path)/i)
      || nextSchema.defaults.image_column,
    true,
  );
  setOptions(
    byId("analysis-crop-image"),
    nextSchema.image_columns,
    findColumn(nextSchema.image_columns, "absolute_crop_path", /(?:absolute.*crop|crop.*path)/i)
      || findColumn(nextSchema.image_columns, "absolute_ocr_path", /(?:absolute.*ocr|ocr.*path)/i),
    true,
  );
  setOptions(
    byId("analysis-subclass"),
    columns,
    findColumn(columns, "Recapture_Subclass", /subclass|group/i),
    true,
  );
  setOptions(
    byId("analysis-identity"),
    columns,
    findColumn(columns, "Data_Identity", /identity|month|source/i),
    true,
  );
  setOptions(
    byId("analysis-quality"),
    columns,
    findColumn(columns, "Quality_Issue", /quality/i),
    true,
  );
  populateIdentityExclusions();
}

function populateIdentityExclusions() {
  const select = byId("analysis-excluded-identities");
  const identityColumn = byId("analysis-identity").value;
  const values = schema?.categories?.[identityColumn] || [];
  setOptions(select, values, "", false);
  for (const option of select.options) option.selected = NOTEBOOK_EXCLUSIONS.includes(option.value);
  select.disabled = !values.length;
}

async function loadSchema(sourceId) {
  schema = await getSchema(sourceId);
  drilldown = null;
  populateControls(schema);
}

export async function configureAnalysis(sources, primarySourceId) {
  sourceSelectOptions(byId("analysis-source"), sources, primarySourceId);
  await selectAnalysisSource(primarySourceId);
}

export async function selectAnalysisSource(sourceId) {
  const select = byId("analysis-source");
  if (![...select.options].some(option => option.value === sourceId)) return;
  select.value = sourceId;
  await loadSchema(sourceId);
}

function selectedValues(select) {
  return [...select.selectedOptions].map(option => option.value);
}

function payload() {
  const sourceId = byId("analysis-source").value;
  const filters = sourceId === state.source?.id ? categoricalFilters() : {};
  if (drilldown) filters[drilldown.column] = [drilldown.value];
  return {
    source_id: sourceId,
    item_id_column: byId("analysis-item").value,
    truth_column: byId("analysis-truth").value,
    ori_prediction_column: byId("analysis-ori-prediction").value,
    crop_prediction_column: byId("analysis-crop-prediction").value,
    original_image_column: byId("analysis-original-image").value,
    crop_image_column: byId("analysis-crop-image").value,
    subclass_column: byId("analysis-subclass").value,
    identity_column: byId("analysis-identity").value,
    quality_column: byId("analysis-quality").value,
    quality_mode: byId("analysis-quality-mode").value,
    exclude_unknown_subclass: byId("analysis-exclude-unknown").checked,
    excluded_identities: selectedValues(byId("analysis-excluded-identities")),
    categorical_filters: filters,
    threshold: Number(byId("analysis-threshold").value),
    outcome: byId("analysis-outcome").value,
    search: byId("analysis-search").value,
    sort: byId("analysis-sort").value,
    page,
    page_size: Number(byId("analysis-page-size").value),
  };
}

function metricTable(summary) {
  const cells = name => {
    const values = summary[name];
    return `<tr><th>${name}</th><td>${values.rows.toLocaleString()}</td><td>${formatPercent(values.accuracy)}</td><td>${formatPercent(values.apcer)} (${values.apcer_errors}/${values.attack_rows})</td><td>${formatPercent(values.bpcer)} (${values.bpcer_errors}/${values.genuine_rows})</td></tr>`;
  };
  return `<div class="metric-table-wrap"><table class="metric-table"><thead><tr><th>Model</th><th>Rows</th><th>Accuracy</th><th>APCER</th><th>BPCER</th></tr></thead><tbody>${cells("ori")}${cells("crop")}${cells("merged")}</tbody></table></div>`;
}

function breakdownTable(title, rows, groupColumns) {
  if (!rows.length) return "";
  const visible = rows.slice(0, 100);
  const heading = groupColumns.map(column => `<th>${escapeText(column)}</th>`).join("");
  const body = visible.map(row => {
    const attributes = groupColumns.map(column => `data-${column === groupColumns[0] ? "group-column" : "secondary-column"}="${escapeText(column)}" data-${column === groupColumns[0] ? "group-value" : "secondary-value"}="${escapeText(row[column])}"`).join(" ");
    return `<tr tabindex="0" ${attributes}>${groupColumns.map(column => `<td>${escapeText(row[column])}</td>`).join("")}<td>${row.count.toLocaleString()}</td><td>${formatPercent(row.ori_error_rate)}</td><td>${formatPercent(row.crop_error_rate)}</td><td>${formatPercent(row.merged_error_rate)}</td></tr>`;
  }).join("");
  const limited = rows.length > visible.length ? `<p class="field-help">Showing first ${visible.length} of ${rows.length} groups.</p>` : "";
  return `<section class="breakdown-panel"><h2>${escapeText(title)}</h2><div class="breakdown-table-wrap"><table class="breakdown-table"><thead><tr>${heading}<th>Rows</th><th>Ori err</th><th>Crop err</th><th>Merged err</th></tr></thead><tbody>${body}</tbody></table></div>${limited}</section>`;
}

function renderSummary(result, request) {
  const summary = result.summary;
  byId("analysis-metrics").innerHTML = [
    metric("Filtered rows", result.filtered_rows.toLocaleString()),
    metric("Original accuracy", formatPercent(summary.ori.accuracy)),
    metric("Crop accuracy", formatPercent(summary.crop.accuracy)),
    metric("Merged accuracy", formatPercent(summary.merged.accuracy)),
    metric("Rows visualized", result.total.toLocaleString()),
  ].join("");
  const selection = drilldown ? ` · ${drilldown.column} = ${drilldown.value}` : "";
  byId("analysis-summary").textContent = `${request.ori_prediction_column} + ${request.crop_prediction_column} at threshold ${request.threshold}${selection}`;
  byId("analysis-clear-drilldown").classList.toggle("hidden", !drilldown);
  byId("analysis-breakdowns").innerHTML = metricTable(summary)
    + breakdownTable("By recapture subclass", result.breakdowns.subclass, [request.subclass_column].filter(Boolean))
    + breakdownTable("By data identity", result.breakdowns.identity, [request.identity_column].filter(Boolean))
    + breakdownTable("Identity × subclass", result.breakdowns.identity_subclass, [request.identity_column, request.subclass_column].filter(Boolean));
}

function renderRows(rows) {
  const gallery = byId("analysis-gallery");
  if (!rows.length) {
    gallery.innerHTML = '<div class="empty-state">No images match this analysis selection.</div>';
    return;
  }
  gallery.innerHTML = rows.map(row => `<article class="analysis-card">
    <div class="analysis-card-heading"><div><strong>${escapeText(row.item_id)}</strong><span>${escapeText(row.identity)} · ${escapeText(row.subclass)}</span></div><span class="status ${escapeText(row.outcome.replaceAll("_", "-"))}">${escapeText(row.outcome.replaceAll("_", " "))}</span></div>
    <div class="analysis-images">${imagePane(row.original_image_url, "Original")}${imagePane(row.crop_image_url, "Crop")}</div>
    <div class="analysis-scores">
      <span>Ori <strong>${formatScore(row.ori_score)}</strong> · ${escapeText(row.ori_failure_type)}</span>
      <span>Crop <strong>${formatScore(row.crop_score)}</strong> · ${escapeText(row.crop_failure_type)}</span>
      <span>Merged <strong>${formatScore(row.merged_score)}</strong> · ${escapeText(row.merged_failure_type)}</span>
    </div>
  </article>`).join("");
}

export async function loadAnalysis() {
  if (!schema) return;
  setBusyCallback(true);
  try {
    const request = payload();
    const result = await getAnalysis(request);
    page = result.page;
    pages = result.pages;
    renderSummary(result, request);
    renderRows(result.rows);
    byId("analysis-page-label").textContent = `Page ${page} of ${pages}`;
    byId("analysis-previous").disabled = page <= 1;
    byId("analysis-next").disabled = page >= pages;
    showErrorCallback("");
  } catch (error) {
    showErrorCallback(error.message);
  } finally {
    setBusyCallback(false);
  }
}

function applyBreakdownRow(row) {
  const column = row.dataset.groupColumn;
  const value = row.dataset.groupValue;
  if (!column || !value) return;
  drilldown = { column, value };
  page = 1;
  loadAnalysis();
}

export function bindAnalysis(setBusy, showError) {
  setBusyCallback = setBusy;
  showErrorCallback = showError;
  byId("analysis-controls").addEventListener("submit", event => {
    event.preventDefault();
    drilldown = null;
    page = 1;
    loadAnalysis();
  });
  byId("analysis-source").addEventListener("change", async event => {
    setBusy(true);
    try {
      await loadSchema(event.target.value);
      page = 1;
    } catch (error) {
      showError(error.message);
    } finally {
      setBusy(false);
    }
  });
  byId("analysis-identity").addEventListener("change", populateIdentityExclusions);
  byId("analysis-previous").addEventListener("click", () => {
    page = Math.max(1, page - 1);
    loadAnalysis();
  });
  byId("analysis-next").addEventListener("click", () => {
    page = Math.min(pages, page + 1);
    loadAnalysis();
  });
  byId("analysis-clear-drilldown").addEventListener("click", () => {
    drilldown = null;
    page = 1;
    loadAnalysis();
  });
  byId("analysis-breakdowns").addEventListener("click", event => {
    const row = event.target.closest("tbody tr");
    if (row) applyBreakdownRow(row);
  });
  byId("analysis-breakdowns").addEventListener("keydown", event => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest("tbody tr");
    if (row) applyBreakdownRow(row);
  });
  bindGalleryImages("analysis-gallery", openImageViewer);
}
