import { getReview } from "./api.js";
import { openImageViewer, withMaxSide } from "./image-viewer.js";
import { categoricalFilters, state } from "./state.js";

const byId = id => document.getElementById(id);
const THUMBNAIL_SIZES = [256, 480, 768, 1024, 1440];
let renderedRows = [];
let renderedPayload = null;

function requestPayload() {
  return {
    source_id: state.source.id,
    item_id_column: byId("item-column").value,
    image_column: byId("image-column").value,
    gradcam_column: byId("gradcam-column").value,
    gradcam_method: byId("gradcam-method").value,
    gradcam_target: byId("gradcam-target").value,
    subclass_column: byId("subclass-column").value,
    truth_column: byId("truth-column").value,
    prediction_column: byId("prediction-column").value,
    threshold: Number(byId("threshold").value),
    truth_rows: byId("truth-rows").value,
    failure_mode: byId("failure-mode").value,
    search: byId("search").value,
    search_columns: [byId("item-column").value, byId("subclass-column").value].filter(Boolean),
    categorical_filters: categoricalFilters(),
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
  return `<button class="card-image" type="button" data-image="${escapeText(url)}" data-label="${label}"><img loading="lazy" data-thumbnail="${escapeText(url)}" alt="${label}"><span class="image-label">${label}</span></button>`;
}

function thumbnailSize(target) {
  return THUMBNAIL_SIZES.find(size => size >= target) || THUMBNAIL_SIZES[THUMBNAIL_SIZES.length - 1];
}

function updateGalleryLayout() {
  const gallery = byId("gallery");
  const preferred = Number(byId("grid-size").value);
  const count = Number(gallery.dataset.cardCount || 0);
  const columns = Math.max(1, Math.min(preferred, count || 1));
  gallery.dataset.columns = String(columns);
  gallery.style.setProperty("--grid-columns", String(columns));

  const card = gallery.querySelector(".card");
  if (!card) return;
  const cardWidth = card.getBoundingClientRect().width;
  const mediaHeight = state.imageMode === "both"
    ? Math.min(360, Math.max(190, 175 + cardWidth * 0.26))
    : Math.min(540, Math.max(215, 180 + cardWidth * 0.6));
  gallery.style.setProperty("--card-media-height", `${Math.round(mediaHeight)}px`);

  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  for (const image of gallery.querySelectorAll("img[data-thumbnail]")) {
    const pane = image.closest(".card-image");
    const target = Math.max(pane.clientWidth, pane.clientHeight) * pixelRatio;
    const size = thumbnailSize(target);
    if (Number(image.dataset.maxSide || 0) >= size) continue;
    image.dataset.maxSide = String(size);
    image.src = withMaxSide(image.dataset.thumbnail, size);
  }
}

function renderRows(rows, payload) {
  const gallery = byId("gallery");
  gallery.replaceChildren();
  gallery.dataset.cardCount = String(rows.length);
  if (!rows.length) {
    gallery.innerHTML = '<div class="empty-state">No rows match the current filters.</div>';
    updateGalleryLayout();
    return;
  }
  const idColumn = payload.item_id_column;
  const groupColumn = payload.subclass_column;
  for (const row of rows) {
    const failure = String(row.values.__failure_type || "unscored");
    const score = row.values[payload.prediction_column];
    const confidence = row.values.__confidence;
    const panes = state.imageMode === "both"
      ? imagePane(row.image_url, "Original") + imagePane(row.gradcam_url, `${payload.gradcam_target === "genuine" ? "Genuine" : "Fraud"} CAM`)
      : state.imageMode === "gradcam" ? imagePane(row.gradcam_url, `${payload.gradcam_target === "genuine" ? "Genuine" : "Fraud"} CAM`) : imagePane(row.image_url, "Original");
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<div class="card-images">${panes}</div><div class="card-body">
      <div class="card-title"><strong>${escapeText(row.values[idColumn] ?? `Row ${row.row_id}`)}</strong><span class="status ${failure.replaceAll(" ", "-")}">${escapeText(failure)}</span></div>
      <div class="card-meta">${escapeText(row.values[groupColumn] ?? "No group")} | pred ${score == null ? "-" : Number(score).toFixed(4)} | conf ${confidence == null ? "-" : Number(confidence).toFixed(3)}</div>
    </div>`;
    gallery.append(card);
  }
  updateGalleryLayout();
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
    renderedRows = result.rows;
    renderedPayload = payload;
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
  byId("grid-size").addEventListener("input", event => {
    byId("grid-value").value = event.target.value;
    updateGalleryLayout();
  });
  byId("image-mode").addEventListener("click", event => {
    const button = event.target.closest("button");
    if (!button) return;
    state.imageMode = button.dataset.value;
    for (const item of byId("image-mode").querySelectorAll("button")) item.classList.toggle("active", item === button);
    if (renderedPayload) renderRows(renderedRows, renderedPayload);
  });
  byId("gallery").addEventListener("click", event => {
    const pane = event.target.closest("[data-image]");
    if (!pane) return;
    openImageViewer(pane.dataset.image, pane.dataset.label || "Image preview");
  });
  byId("gallery").addEventListener("error", event => {
    if (event.target.tagName !== "IMG") return;
    const pane = event.target.closest(".card-image");
    if (pane) pane.innerHTML = '<div class="missing-image">Image file unavailable</div>';
  }, true);
  new ResizeObserver(updateGalleryLayout).observe(byId("review-view"));
}
