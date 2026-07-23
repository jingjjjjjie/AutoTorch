import { getReview } from "./api.js";
import { bindGalleryImages, byId, escapeText, imagePane, metric } from "./dom.js";
import { openImageViewer, withMaxSide } from "./image-viewer.js";
import { categoricalFilters, state } from "./state.js";

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
    gradcam_layer: byId("gradcam-layer").value,
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

function reviewImagePane(url, label) {
  return imagePane(url, label).replace("src=", "data-thumbnail=");
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
  const mediaHeight = state.imageMode === "montage"
    ? Math.min(760, Math.max(420, 360 + cardWidth * 0.55))
    : ["both", "side_by_side"].includes(state.imageMode)
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
    const targetLabel = payload.gradcam_target === "genuine" ? "Genuine" : "Fraud";
    const selectedCam = payload.gradcam_target === "genuine"
      ? row.genuine_gradcam_url
      : row.fraud_gradcam_url;
    const targetPanes = payload.gradcam_target === "both"
      ? reviewImagePane(row.genuine_gradcam_url, `${payload.gradcam_layer} · Genuine`)
        + reviewImagePane(row.fraud_gradcam_url, `${payload.gradcam_layer} · Fraud`)
      : reviewImagePane(selectedCam, `${payload.gradcam_layer} · ${targetLabel}`);
    const montagePanes = (row.gradcam_layers || []).map(layer => {
      if (payload.gradcam_target === "genuine") return reviewImagePane(layer.genuine_url, `${layer.layer} · Genuine`);
      if (payload.gradcam_target === "fraud") return reviewImagePane(layer.fraud_url, `${layer.layer} · Fraud`);
      return reviewImagePane(layer.genuine_url, `${layer.layer} · Genuine`)
        + reviewImagePane(layer.fraud_url, `${layer.layer} · Fraud`);
    }).join("");
    const panes = state.imageMode === "both"
      ? reviewImagePane(row.image_url, "Original") + targetPanes
      : state.imageMode === "gradcam"
        ? targetPanes
        : state.imageMode === "side_by_side"
          ? reviewImagePane(row.genuine_gradcam_url, `${payload.gradcam_layer} · Genuine`)
            + reviewImagePane(row.fraud_gradcam_url, `${payload.gradcam_layer} · Fraud`)
          : state.imageMode === "montage"
            ? montagePanes
            : reviewImagePane(row.image_url, "Original");
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `<div class="card-images ${state.imageMode === "montage" ? "montage" : ""}">${panes}</div><div class="card-body">
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
  bindGalleryImages("gallery", openImageViewer);
  new ResizeObserver(updateGalleryLayout).observe(byId("review-view"));
}
