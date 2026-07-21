// Independent two-experiment comparison view.

import { getComparison, getSchema } from "./api.js";
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

let schemaA = null;
let schemaB = null;
let page = 1;
let pages = 1;
let setBusyCallback = () => {};
let showErrorCallback = () => {};

const sideIds = side => ({
  item: `comparison-item-${side}`,
  truth: `comparison-truth-${side}`,
  prediction: `comparison-prediction-${side}`,
  image: `comparison-image-${side}`,
  gradcam: `comparison-gradcam-${side}`,
});

function sourceSelectOptions(select, sources, preferred) {
  setOptions(select, sources.map(source => source.id), preferred, false);
  [...select.options].forEach((option, index) => { option.textContent = sources[index].label; });
}

function populateSide(schema, side) {
  const ids = sideIds(side);
  setOptions(byId(ids.item), schema.columns, schema.defaults.item_id_column, false);
  setOptions(byId(ids.truth), schema.columns, schema.defaults.truth_column, false);
  setOptions(byId(ids.prediction), schema.numeric_columns, schema.defaults.prediction_column, false);
  setOptions(byId(ids.image), schema.image_columns, schema.defaults.image_column, true);
  setOptions(byId(ids.gradcam), schema.gradcam_columns, schema.gradcam_columns[0] || "", true);
  if (!schema.gradcam_columns.length && schema.prepared_gradcam_methods.length) {
    byId(ids.gradcam).options[0].textContent = "Auto-detect prepared CAM";
  }
  if (schema.review_preset?.threshold != null) {
    byId(`comparison-threshold-${side}`).value = String(schema.review_preset.threshold);
  }
}

function preferredMetadataColumn(schema, exactName, fallbackPattern) {
  return schema.columns.find(column => column === exactName)
    || schema.categorical_columns.find(column => fallbackPattern.test(column))
    || "";
}

function populateMetadata(schema) {
  setOptions(
    byId("comparison-subclass"),
    schema.columns,
    preferredMetadataColumn(schema, "Recapture_Subclass", /subclass|group/i),
    true,
  );
  setOptions(
    byId("comparison-identity"),
    schema.columns,
    preferredMetadataColumn(schema, "Data_Identity", /identity|month|source/i),
    true,
  );
  setOptions(
    byId("comparison-quality"),
    schema.columns,
    preferredMetadataColumn(schema, "Quality_Issue", /quality/i),
    true,
  );
}

async function loadSide(side) {
  const sourceId = byId(`comparison-source-${side}`).value;
  const schema = await getSchema(sourceId);
  if (side === "a") {
    schemaA = schema;
    populateMetadata(schema);
  } else {
    schemaB = schema;
  }
  populateSide(schema, side);
}

export async function configureComparison(sources, primarySourceId) {
  const fallbackB = sources.find(source => source.id !== primarySourceId && /crop/i.test(source.label))?.id
    || sources.find(source => source.id !== primarySourceId)?.id
    || primarySourceId;
  sourceSelectOptions(byId("comparison-source-a"), sources, primarySourceId);
  sourceSelectOptions(byId("comparison-source-b"), sources, fallbackB);
  await Promise.all([loadSide("a"), loadSide("b")]);
}

function payload() {
  const sourceA = byId("comparison-source-a").value;
  return {
    source_a_id: sourceA,
    source_b_id: byId("comparison-source-b").value,
    item_id_column_a: byId("comparison-item-a").value,
    item_id_column_b: byId("comparison-item-b").value,
    truth_column_a: byId("comparison-truth-a").value,
    truth_column_b: byId("comparison-truth-b").value,
    prediction_column_a: byId("comparison-prediction-a").value,
    prediction_column_b: byId("comparison-prediction-b").value,
    image_column_a: byId("comparison-image-a").value,
    image_column_b: byId("comparison-image-b").value,
    gradcam_column_a: byId("comparison-gradcam-a").value,
    gradcam_column_b: byId("comparison-gradcam-b").value,
    gradcam_method: byId("comparison-gradcam-method").value,
    gradcam_target: byId("comparison-gradcam-target").value,
    threshold_a: Number(byId("comparison-threshold-a").value),
    threshold_b: Number(byId("comparison-threshold-b").value),
    subclass_column: byId("comparison-subclass").value,
    identity_column: byId("comparison-identity").value,
    quality_column: byId("comparison-quality").value,
    categorical_filters: sourceA === state.source?.id ? categoricalFilters() : {},
    outcomes: [byId("comparison-outcome").value].filter(Boolean),
    search: byId("comparison-search").value,
    sort: byId("comparison-sort").value,
    page,
    page_size: Number(byId("comparison-page-size").value),
  };
}

function outcomeButton(outcome, label, count) {
  const active = byId("comparison-outcome").value === outcome ? " active" : "";
  return `<button class="outcome-cell${active}" type="button" data-outcome="${outcome}"><strong>${count.toLocaleString()}</strong><span>${label}</span></button>`;
}

function renderSummary(result) {
  const summary = result.summary;
  byId("comparison-metrics").innerHTML = [
    metric("Aligned rows", summary.comparable_rows.toLocaleString()),
    metric("Experiment A accuracy", formatPercent(summary.a.accuracy)),
    metric("Experiment B accuracy", formatPercent(summary.b.accuracy)),
    metric("Prediction disagreements", summary.prediction_disagreements.toLocaleString()),
    metric("Truth mismatches", summary.alignment.truth_mismatches.toLocaleString()),
  ].join("");
  byId("comparison-summary").textContent = `${result.total.toLocaleString()} rows after filters · ${summary.alignment.matched.toLocaleString()} globally aligned · ${summary.alignment.only_in_a.toLocaleString()} only in A · ${summary.alignment.only_in_b.toLocaleString()} only in B`;
  byId("comparison-matrix").innerHTML = `
    <div class="matrix-heading"><strong>Correctness matrix</strong><span>Click a cell to inspect its samples</span></div>
    <div class="matrix-grid">
      ${outcomeButton("both_correct", "Both correct", summary.outcomes.both_correct)}
      ${outcomeButton("a_only_correct", "A only correct", summary.outcomes.a_only_correct)}
      ${outcomeButton("b_only_correct", "B only correct", summary.outcomes.b_only_correct)}
      ${outcomeButton("both_wrong", "Both wrong", summary.outcomes.both_wrong)}
    </div>
    <div class="aux-outcomes">
      ${outcomeButton("unscored", "Unscored", summary.outcomes.unscored)}
      ${outcomeButton("truth_mismatch", "Truth mismatch", summary.outcomes.truth_mismatch)}
      ${outcomeButton("only_in_a", "Only in A", summary.outcomes.only_in_a)}
      ${outcomeButton("only_in_b", "Only in B", summary.outcomes.only_in_b)}
    </div>`;
}

function renderRows(rows) {
  const gallery = byId("comparison-gallery");
  if (!rows.length) {
    gallery.innerHTML = '<div class="empty-state">No aligned rows match this comparison.</div>';
    return;
  }
  gallery.innerHTML = rows.map(row => {
    const metadata = Object.entries(row.metadata || {})
      .map(([column, value]) => `${escapeText(column)}: ${escapeText(value)}`)
      .join(" · ");
    const camPanes = row.a_gradcam_url || row.b_gradcam_url
      ? imagePane(row.a_gradcam_url, "A CAM") + imagePane(row.b_gradcam_url, "B CAM")
      : "";
    return `<article class="comparison-card">
      <div class="comparison-card-heading">
        <div><strong>${escapeText(row.item_id)}${row.occurrence ? ` · occurrence ${row.occurrence + 1}` : ""}</strong><span>${metadata}</span></div>
        <span class="status ${escapeText(row.outcome.replaceAll("_", "-"))}">${escapeText(row.outcome.replaceAll("_", " "))}</span>
      </div>
      <div class="comparison-images">
        ${imagePane(row.a_image_url, "A original")}${imagePane(row.b_image_url, "B original")}${camPanes}
      </div>
      <div class="comparison-scores">
        <span>A <strong>${formatScore(row.a_score)}</strong> · ${escapeText(row.a_failure_type)}</span>
        <span>B <strong>${formatScore(row.b_score)}</strong> · ${escapeText(row.b_failure_type)}</span>
        <span>True-confidence Δ <strong>${formatScore(row.true_confidence_delta)}</strong></span>
      </div>
    </article>`;
  }).join("");
}

export async function loadComparison() {
  if (!schemaA || !schemaB) return;
  setBusyCallback(true);
  try {
    const result = await getComparison(payload());
    page = result.page;
    pages = result.pages;
    renderSummary(result);
    renderRows(result.rows);
    byId("comparison-page-label").textContent = `Page ${page} of ${pages}`;
    byId("comparison-previous").disabled = page <= 1;
    byId("comparison-next").disabled = page >= pages;
    showErrorCallback("");
  } catch (error) {
    showErrorCallback(error.message);
  } finally {
    setBusyCallback(false);
  }
}

function detachedViewer(side) {
  const parameters = new URLSearchParams({
    view: "review",
    source: byId(`comparison-source-${side}`).value,
    item: byId(`comparison-item-${side}`).value,
    truth: byId(`comparison-truth-${side}`).value,
    prediction: byId(`comparison-prediction-${side}`).value,
    image: byId(`comparison-image-${side}`).value,
    gradcam: byId(`comparison-gradcam-${side}`).value,
    threshold: byId(`comparison-threshold-${side}`).value,
    target: byId("comparison-gradcam-target").value,
  });
  window.open(`/?${parameters}`, "_blank", "noopener");
}

export function bindComparison(setBusy, showError) {
  setBusyCallback = setBusy;
  showErrorCallback = showError;
  byId("comparison-controls").addEventListener("submit", event => {
    event.preventDefault();
    page = 1;
    loadComparison();
  });
  for (const side of ["a", "b"]) {
    byId(`comparison-source-${side}`).addEventListener("change", async () => {
      setBusy(true);
      try {
        await loadSide(side);
        page = 1;
      } catch (error) {
        showError(error.message);
      } finally {
        setBusy(false);
      }
    });
    byId(`open-viewer-${side}`).addEventListener("click", () => detachedViewer(side));
  }
  byId("comparison-previous").addEventListener("click", () => {
    page = Math.max(1, page - 1);
    loadComparison();
  });
  byId("comparison-next").addEventListener("click", () => {
    page = Math.min(pages, page + 1);
    loadComparison();
  });
  byId("comparison-matrix").addEventListener("click", event => {
    const cell = event.target.closest("[data-outcome]");
    if (!cell) return;
    byId("comparison-outcome").value = cell.dataset.outcome;
    page = 1;
    loadComparison();
  });
  bindGalleryImages("comparison-gallery", openImageViewer);
}
