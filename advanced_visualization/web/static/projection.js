import { getPoint, getProjection } from "./api.js";
import { openImageViewer } from "./image-viewer.js";
import { createScatterPlot } from "./scatter-plot.js";
import { categoricalFilters, state } from "./state.js";
import {
  bindSubclassLimits,
  projectionClassLimits,
  updateSubclassAvailability,
} from "./subclass-limits.js";

const byId = id => document.getElementById(id);
let plot = null;
let projectionRequest = 0;
let pointRequest = 0;

function updateDisplayedCount(visible, total) {
  byId("displayed-image-count").textContent = visible.toLocaleString();
  byId("displayed-image-total").textContent = visible === total
    ? `${total.toLocaleString()} projected`
    : `of ${total.toLocaleString()} projected`;
}

export function clearProjection() {
  projectionRequest += 1;
  resetProjectionView("Select a projection and run it.", "Not projected");
}

function resetProjectionView(summary, countStatus) {
  pointRequest += 1;
  state.projection = null;
  plot?.setData([]);
  byId("projection-summary").textContent = summary;
  byId("point-detail").innerHTML = '<div class="empty-state">Select a point to inspect its source image.</div>';
  byId("displayed-image-count").textContent = "0";
  byId("displayed-image-total").textContent = countStatus;
}

async function showPoint(point, showError) {
  const requestId = ++pointRequest;
  const detail = byId("point-detail");
  detail.innerHTML = '<div class="empty-state">Loading point...</div>';
  const sourceId = state.source.id;
  try {
    const result = await getPoint(sourceId, point.row_id, {
      image_column: byId("feature-image-column").value,
      gradcam_column: byId("feature-gradcam-column").value,
      item_id_column: byId("item-column").value,
      prediction_column: byId("prediction-column").value,
      group_column: byId("color-column").value,
    });
    if (requestId !== pointRequest || state.source.id !== sourceId) return;
    detail.replaceChildren();
    const title = document.createElement("h3");
    title.textContent = point.label;
    const coordinates = document.createElement("p");
    coordinates.className = "point-coordinates";
    coordinates.textContent = `Row ${point.row_id} | x ${point.x.toFixed(3)} | y ${point.y.toFixed(3)}`;
    detail.append(title, coordinates);
    const metadata = document.createElement("dl");
    metadata.className = "point-metadata";
    for (const [label, value] of Object.entries(result.values)) {
      const term = document.createElement("dt"); term.textContent = label;
      const description = document.createElement("dd"); description.textContent = value ?? "Missing";
      metadata.append(term, description);
    }
    if (metadata.children.length) detail.append(metadata);

    const media = [
      ["Original", result.image_url],
      ["Grad-CAM", result.gradcam_url],
      ["Grad-CAM++", result.gradcam_plus_url],
    ].filter(([, url]) => url);
    if (!media.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state"; empty.textContent = "No prepared image artifacts for this point.";
      detail.append(empty); return;
    }
    const actions = document.createElement("div");
    actions.className = "inspector-actions";
    const image = document.createElement("img");
    image.className = "inspector-image";
    image.alt = "Selected point artifact";
    const activate = (label, url, button) => {
      image.src = url; image.alt = label; image.dataset.image = url;
      for (const item of actions.querySelectorAll("button")) item.classList.toggle("active", item === button);
    };
    for (const [label, url] of media) {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = label;
      button.addEventListener("click", () => activate(label, url, button));
      actions.append(button);
    }
    detail.append(actions, image);
    activate(media[0][0], media[0][1], actions.firstElementChild);
  } catch (error) {
    if (requestId !== pointRequest || state.source.id !== sourceId) return;
    detail.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "empty-state"; empty.textContent = error.message;
    detail.append(empty);
    showError(error.message);
  }
}

function projectionPayload(sourceId) {
  return {
    source_id: sourceId,
    method: byId("projection-method").value,
    feature_columns: state.schema.feature_columns,
    color_column: byId("color-column").value,
    categorical_filters: categoricalFilters(),
    scale: byId("scale-features").checked,
    max_rows: Number(byId("max-rows").value),
    ...projectionClassLimits(),
    perplexity: Number(byId("perplexity").value),
    umap_neighbors: Number(byId("umap-neighbors").value),
    umap_min_dist: Number(byId("umap-min-dist").value),
  };
}

export async function loadProjection(setBusy, showError) {
  if (!state.source) return;
  const requestId = ++projectionRequest;
  const sourceId = state.source.id;
  const request = projectionPayload(sourceId);
  const requestFingerprint = JSON.stringify(request);
  resetProjectionView("Calculating projection...", "Calculating...");
  setBusy(true);
  try {
    const result = await getProjection(request);
    if (requestId !== projectionRequest || state.source.id !== sourceId) return;
    if (JSON.stringify(projectionPayload(sourceId)) !== requestFingerprint) {
      resetProjectionView("Controls changed. Run the projection again.", "Not projected");
      showError("");
      return;
    }
    state.projection = result;
    updateSubclassAvailability(result.class_counts);
    const countSummary = result.available_rows > result.rows
      ? `${result.rows.toLocaleString()} of ${result.available_rows.toLocaleString()} filtered points.`
      : `${result.rows.toLocaleString()} points.`;
    byId("projection-summary").textContent = `${countSummary} ${result.subtitle}`.trim();
    plot.setData(result.points, request.color_column || "All points");
    showError("");
  } catch (error) {
    if (requestId === projectionRequest && state.source?.id === sourceId) {
      const controlsChanged = JSON.stringify(projectionPayload(sourceId)) !== requestFingerprint;
      resetProjectionView(
        controlsChanged ? "Controls changed. Run the projection again." : "Projection could not be generated.",
        "Not projected",
      );
      showError(controlsChanged ? "" : error.message);
    }
  } finally {
    if (requestId === projectionRequest && state.source?.id === sourceId) setBusy(false);
  }
}

export function bindProjection(setBusy, showError) {
  plot = createScatterPlot({
    onPointSelect: point => showPoint(point, showError),
    onVisibleCount: updateDisplayedCount,
  });
  bindSubclassLimits(() => {
    if (state.projection) {
      byId("projection-summary").textContent = "Subclass limits changed. Run the projection to update.";
    }
  });
  byId("feature-controls").addEventListener("submit", event => { event.preventDefault(); loadProjection(setBusy, showError); });
  byId("point-detail").addEventListener("click", event => {
    const image = event.target.closest("[data-image]");
    if (!image) return;
    openImageViewer(image.dataset.image, image.alt || "Selected point artifact");
  });
  const updateMethodFields = () => {
    const method = byId("projection-method").value;
    byId("perplexity-field").classList.toggle("hidden", method !== "tsne");
    byId("umap-fields").classList.toggle("hidden", method !== "umap");
  };
  byId("projection-method").addEventListener("change", updateMethodFields);
  updateMethodFields();
  byId("plot-fullscreen").addEventListener("click", () => {
    const layout = document.querySelector(".projection-layout");
    const fullscreen = layout.classList.toggle("fullscreen");
    byId("plot-fullscreen").textContent = fullscreen ? "Exit fullscreen" : "Fullscreen";
    setTimeout(() => plot.redraw(), 0);
  });
  document.addEventListener("keydown", event => {
    const layout = document.querySelector(".projection-layout");
    if (event.key === "Escape" && layout.classList.contains("fullscreen")) {
      layout.classList.remove("fullscreen");
      byId("plot-fullscreen").textContent = "Fullscreen";
      setTimeout(() => plot.redraw(), 0);
    }
  });
}
