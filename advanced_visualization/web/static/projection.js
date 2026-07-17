import { getProjection } from "./api.js";
import { state } from "./state.js";

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
  const xs = points.map(point => point.x), ys = points.map(point => point.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
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

function showPoint(point) {
  byId("point-detail").innerHTML = `<h3>${point.label}</h3><p>Row ${point.row_id}</p>${point.image_url ? `<img src="${point.image_url}" alt="Selected source image">` : '<div class="empty-state">No image column selected.</div>'}`;
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
      image_column: byId("feature-image-column").value,
      scale: byId("scale-features").checked,
      max_rows: Number(byId("max-rows").value),
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
  byId("projection-canvas").addEventListener("click", event => { const { point } = nearestPoint(event); if (point) showPoint(point); });
  window.addEventListener("resize", drawProjection);
}

