import { getPoint, getProjection } from "./api.js";
import { openImageViewer } from "./image-viewer.js";
import { categoricalFilters, state } from "./state.js";

const byId = id => document.getElementById(id);
const palette = ["#63d59a", "#f1b85b", "#ff786f", "#80b9ff", "#d89cff", "#e7df73", "#72d9d2", "#f39ac1"];
let screenPoints = [];
let colors = new Map();

function drawProjection() {
  if (!state.projection) return;
  const canvas = byId("projection-canvas");
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(bounds.width * ratio));
  canvas.height = Math.max(1, Math.floor(bounds.height * ratio));
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, bounds.width, bounds.height);
  const points = state.projection.points;
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const point of points) {
    xMin = Math.min(xMin, point.x); xMax = Math.max(xMax, point.x);
    yMin = Math.min(yMin, point.y); yMax = Math.max(yMax, point.y);
  }
  const pad = 28;
  const sx = value => pad + ((value - xMin) / (xMax - xMin || 1)) * (bounds.width - pad * 2);
  const sy = value => bounds.height - pad - ((value - yMin) / (yMax - yMin || 1)) * (bounds.height - pad * 2);
  const labels = [...new Set(points.map(point => point.label))].sort();
  colors = new Map(labels.map((label, index) => [label, palette[index % palette.length]]));
  screenPoints = points.map(point => ({ ...point, sx: sx(point.x), sy: sy(point.y) }));
  for (const point of screenPoints) {
    context.beginPath(); context.arc(point.sx, point.sy, 3.2, 0, Math.PI * 2);
    context.fillStyle = colors.get(point.label); context.globalAlpha = 0.72; context.fill();
  }
  context.globalAlpha = 1;
  byId("legend").innerHTML = labels.slice(0, 40).map(label => `<span class="legend-item"><i class="legend-swatch" style="background:${colors.get(label)}"></i>${label}</span>`).join("");
}

function nearestPoint(event) {
  const bounds = event.currentTarget.getBoundingClientRect();
  const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
  let nearest = null, distance = 12;
  for (const point of screenPoints) {
    const current = Math.hypot(point.sx - x, point.sy - y);
    if (current < distance) { distance = current; nearest = point; }
  }
  return { point: nearest, x, y };
}

async function showPoint(point, showError) {
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
    if (state.source.id !== sourceId) return;
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
    detail.replaceChildren();
    const empty = document.createElement("div");
    empty.className = "empty-state"; empty.textContent = error.message;
    detail.append(empty);
    showError(error.message);
  }
}

export async function loadProjection(setBusy, showError) {
  if (!state.source) return;
  setBusy(true);
  try {
    const result = await getProjection({
      source_id: state.source.id,
      method: byId("projection-method").value,
      feature_columns: state.schema.feature_columns,
      color_column: byId("color-column").value,
      categorical_filters: categoricalFilters(),
      scale: byId("scale-features").checked,
      max_rows: Number(byId("max-rows").value),
      perplexity: Number(byId("perplexity").value),
      umap_neighbors: Number(byId("umap-neighbors").value),
      umap_min_dist: Number(byId("umap-min-dist").value),
    });
    state.projection = result;
    byId("projection-summary").textContent = `${result.rows.toLocaleString()} points. ${result.subtitle}`;
    drawProjection();
    showError("");
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

export function bindProjection(setBusy, showError) {
  byId("feature-controls").addEventListener("submit", event => { event.preventDefault(); loadProjection(setBusy, showError); });
  byId("projection-canvas").addEventListener("mousemove", event => {
    const { point, x, y } = nearestPoint(event), tooltip = byId("plot-tooltip");
    tooltip.classList.toggle("hidden", !point);
    if (point) { tooltip.textContent = `${point.label} | row ${point.row_id}`; tooltip.style.left = `${x + 10}px`; tooltip.style.top = `${y + 10}px`; }
  });
  byId("projection-canvas").addEventListener("mouseleave", () => byId("plot-tooltip").classList.add("hidden"));
  byId("projection-canvas").addEventListener("click", event => { const { point } = nearestPoint(event); if (point) showPoint(point, showError); });
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
    setTimeout(drawProjection, 0);
  });
  document.addEventListener("keydown", event => {
    const layout = document.querySelector(".projection-layout");
    if (event.key === "Escape" && layout.classList.contains("fullscreen")) {
      layout.classList.remove("fullscreen");
      byId("plot-fullscreen").textContent = "Fullscreen";
      setTimeout(drawProjection, 0);
    }
  });
  new ResizeObserver(drawProjection).observe(document.querySelector(".canvas-wrap"));
}
