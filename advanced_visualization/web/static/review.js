import { getReview } from "./api.js";
import { selectedValues, state } from "./state.js";

const byId = id => document.getElementById(id);

function requestPayload() {
  const filterColumn = byId("filter-column").value;
  const filters = filterColumn ? { [filterColumn]: selectedValues(byId("filter-values")) } : {};
  return {
    source_id: state.source.id,
    item_id_column: byId("item-column").value,
    image_column: byId("image-column").value,
    gradcam_column: byId("gradcam-column").value,
    subclass_column: byId("subclass-column").value,
    truth_column: byId("truth-column").value,
    prediction_column: byId("prediction-column").value,
    threshold: Number(byId("threshold").value),
    truth_rows: byId("truth-rows").value,
    failure_mode: byId("failure-mode").value,
    search: byId("search").value,
    search_columns: [byId("item-column").value, byId("subclass-column").value].filter(Boolean),
    categorical_filters: filters,
    sort: byId("sort").value,
    page: state.page,
    page_size: Number(byId("page-size").value),
  };
}

function metric(label, value) {
  return `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`;
}

function escapeText(value) {
  const element = document.createElement("span");
  element.textContent = value ?? "Missing";
  return element.innerHTML;
}

function imagePane(url, label) {
  if (!url) return `<div class="card-image"><div class="missing-image">No ${label}</div></div>`;
  return `<button class="card-image" type="button" data-image="${escapeText(url)}"><img loading="lazy" src="${escapeText(url)}" alt="${label}"><span class="image-label">${label}</span></button>`;
}

function renderRows(rows, payload) {
  const gallery = byId("gallery");
  gallery.replaceChildren();
  if (!rows.length) {
    gallery.innerHTML = '<div class="empty-state">No rows match the current filters.</div>';
    return;
  }
  const idColumn = payload.item_id_column;
  const groupColumn = payload.subclass_column;
  for (const row of rows) {
    const failure = String(row.values.__failure_type || "unscored");
    const score = row.values[payload.prediction_column];
    const confidence = row.values.__confidence;
    const panes = state.imageMode === "both"
      ? imagePane(row.image_url, "Original") + imagePane(row.gradcam_url, "Grad-CAM")
      : state.imageMode === "gradcam" ? imagePane(row.gradcam_url, "Grad-CAM") : imagePane(row.image_url, "Original");
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<div class="card-images">${panes}</div><div class="card-body">
      <div class="card-title"><strong>${escapeText(row.values[idColumn] ?? `Row ${row.row_id}`)}</strong><span class="status ${failure.replaceAll(" ", "-")}">${escapeText(failure)}</span></div>
      <div class="card-meta">${escapeText(row.values[groupColumn] ?? "No group")} | pred ${score == null ? "-" : Number(score).toFixed(4)} | conf ${confidence == null ? "-" : Number(confidence).toFixed(3)}</div>
    </div>`;
    gallery.append(card);
  }
}

export async function loadReview(setBusy, showError) {
  if (!state.source) return;
  setBusy(true);
  try {
    const payload = requestPayload();
    const result = await getReview(payload);
    state.page = result.page;
    state.pages = result.pages;
    byId("metrics").innerHTML = [
      metric("Rows", result.metrics.rows.toLocaleString()),
      metric("Scored", result.metrics.scored.toLocaleString()),
      metric("Failures", result.metrics.failures.toLocaleString()),
      metric("Failure rate", `${(result.metrics.failure_rate * 100).toFixed(1)}%`),
      metric("High-conf failures", result.metrics.high_confidence_failures.toLocaleString()),
    ].join("");
    byId("review-summary").textContent = `${result.total.toLocaleString()} matching rows from ${state.source.label}`;
    byId("page-label").textContent = `Page ${result.page} of ${result.pages}`;
    byId("previous-page").disabled = result.page <= 1;
    byId("next-page").disabled = result.page >= result.pages;
    renderRows(result.rows, payload);
    showError("");
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

export function bindReview(setBusy, showError) {
  byId("review-controls").addEventListener("submit", event => { event.preventDefault(); state.page = 1; loadReview(setBusy, showError); });
  byId("previous-page").addEventListener("click", () => { state.page = Math.max(1, state.page - 1); loadReview(setBusy, showError); });
  byId("next-page").addEventListener("click", () => { state.page = Math.min(state.pages, state.page + 1); loadReview(setBusy, showError); });
  byId("threshold").addEventListener("input", event => { byId("threshold-value").value = Number(event.target.value).toFixed(2); });
  byId("grid-size").addEventListener("input", event => { byId("grid-value").value = event.target.value; byId("gallery").style.setProperty("--grid-columns", event.target.value); });
  byId("image-mode").addEventListener("click", event => {
    const button = event.target.closest("button");
    if (!button) return;
    state.imageMode = button.dataset.value;
    for (const item of byId("image-mode").querySelectorAll("button")) item.classList.toggle("active", item === button);
    loadReview(setBusy, showError);
  });
  byId("gallery").addEventListener("click", event => {
    const pane = event.target.closest("[data-image]");
    if (!pane) return;
    byId("dialog-image").src = `${pane.dataset.image}&max_side=0`;
    byId("image-dialog").showModal();
  });
  byId("gallery").addEventListener("error", event => {
    if (event.target.tagName !== "IMG") return;
    const pane = event.target.closest(".card-image");
    if (pane) pane.innerHTML = '<div class="missing-image">Image file unavailable</div>';
  }, true);
}
