const byId = id => document.getElementById(id);

const MIN_SCALE = 1;
const MAX_SCALE = 8;
const OPEN_SIZE = 2048;
const DETAIL_SIZE = 4096;

let scale = MIN_SCALE;
let offsetX = 0;
let offsetY = 0;
let drag = null;
let sourceUrl = "";
let detailLoaded = false;
let bound = false;

export function withMaxSide(url, maxSide) {
  if (!url) return "";
  if (url.startsWith("data:") || url.startsWith("blob:")) return url;
  const parsed = new URL(url, window.location.href);
  parsed.searchParams.set("max_side", String(maxSide));
  return parsed.origin === window.location.origin
    ? `${parsed.pathname}${parsed.search}${parsed.hash}`
    : parsed.href;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function constrainPan() {
  const image = byId("dialog-image");
  const viewport = byId("dialog-viewport");
  const viewportRatio = viewport.clientWidth / Math.max(1, viewport.clientHeight);
  const imageRatio = image.naturalWidth / Math.max(1, image.naturalHeight);
  const fittedWidth = imageRatio >= viewportRatio
    ? viewport.clientWidth
    : viewport.clientHeight * imageRatio;
  const fittedHeight = imageRatio >= viewportRatio
    ? viewport.clientWidth / Math.max(imageRatio, 0.001)
    : viewport.clientHeight;
  const maxX = Math.max(0, (fittedWidth * scale - viewport.clientWidth) / 2);
  const maxY = Math.max(0, (fittedHeight * scale - viewport.clientHeight) / 2);
  offsetX = clamp(offsetX, -maxX, maxX);
  offsetY = clamp(offsetY, -maxY, maxY);
}

function renderTransform() {
  constrainPan();
  const image = byId("dialog-image");
  image.style.transform = `translate3d(${offsetX}px, ${offsetY}px, 0) scale(${scale})`;
  image.classList.toggle("zoomed", scale > MIN_SCALE);
  byId("zoom-reset").textContent = `${Math.round(scale * 100)}%`;
  byId("zoom-out").disabled = scale <= MIN_SCALE;
  byId("zoom-in").disabled = scale >= MAX_SCALE;
}

function loadDetailImage() {
  if (detailLoaded || !sourceUrl || scale < 2) return;
  detailLoaded = true;
  byId("dialog-image").src = withMaxSide(sourceUrl, DETAIL_SIZE);
}

function setScale(nextScale, clientX, clientY) {
  const previous = scale;
  scale = clamp(nextScale, MIN_SCALE, MAX_SCALE);
  if (scale === previous) return;

  if (clientX != null && clientY != null) {
    const bounds = byId("dialog-viewport").getBoundingClientRect();
    const anchorX = clientX - bounds.left - bounds.width / 2;
    const anchorY = clientY - bounds.top - bounds.height / 2;
    const ratio = scale / previous;
    offsetX = anchorX - (anchorX - offsetX) * ratio;
    offsetY = anchorY - (anchorY - offsetY) * ratio;
  }
  if (scale === MIN_SCALE) {
    offsetX = 0;
    offsetY = 0;
  }
  loadDetailImage();
  renderTransform();
}

function resetZoom() {
  scale = MIN_SCALE;
  offsetX = 0;
  offsetY = 0;
  drag = null;
  renderTransform();
}

function closeViewer() {
  const dialog = byId("image-dialog");
  if (dialog.open) dialog.close();
}

export function openImageViewer(url, label = "Image preview") {
  if (!url) return;
  const dialog = byId("image-dialog");
  const image = byId("dialog-image");
  sourceUrl = url;
  detailLoaded = false;
  byId("dialog-title").textContent = label;
  image.alt = label;
  image.src = withMaxSide(url, OPEN_SIZE);
  resetZoom();
  if (!dialog.open) dialog.showModal();
  byId("zoom-in").focus();
}

export function bindImageViewer() {
  if (bound) return;
  bound = true;
  const dialog = byId("image-dialog");
  const viewport = byId("dialog-viewport");
  const image = byId("dialog-image");

  byId("zoom-in").addEventListener("click", () => setScale(scale * 1.25));
  byId("zoom-out").addEventListener("click", () => setScale(scale / 1.25));
  byId("zoom-reset").addEventListener("click", resetZoom);
  byId("close-dialog").addEventListener("click", closeViewer);

  viewport.addEventListener("wheel", event => {
    event.preventDefault();
    setScale(scale * (event.deltaY < 0 ? 1.2 : 1 / 1.2), event.clientX, event.clientY);
  }, { passive: false });

  viewport.addEventListener("dblclick", event => {
    setScale(scale > MIN_SCALE ? MIN_SCALE : 2, event.clientX, event.clientY);
  });

  image.addEventListener("pointerdown", event => {
    if (scale <= MIN_SCALE || event.button !== 0) return;
    event.preventDefault();
    image.setPointerCapture(event.pointerId);
    drag = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, offsetX, offsetY };
    image.classList.add("dragging");
  });
  image.addEventListener("pointermove", event => {
    if (!drag || drag.pointerId !== event.pointerId) return;
    offsetX = drag.offsetX + event.clientX - drag.x;
    offsetY = drag.offsetY + event.clientY - drag.y;
    renderTransform();
  });
  const stopDragging = event => {
    if (!drag || (event.pointerId != null && drag.pointerId !== event.pointerId)) return;
    drag = null;
    image.classList.remove("dragging");
  };
  image.addEventListener("pointerup", stopDragging);
  image.addEventListener("pointercancel", stopDragging);

  image.addEventListener("load", renderTransform);
  dialog.addEventListener("click", event => {
    if (event.target === dialog) closeViewer();
  });
  dialog.addEventListener("close", () => {
    resetZoom();
    image.removeAttribute("src");
    sourceUrl = "";
    detailLoaded = false;
  });
  document.addEventListener("keydown", event => {
    if (!dialog.open) return;
    if (event.key === "+" || event.key === "=") setScale(scale * 1.25);
    if (event.key === "-") setScale(scale / 1.25);
    if (event.key === "0") resetZoom();
  });
  new ResizeObserver(renderTransform).observe(viewport);
}
