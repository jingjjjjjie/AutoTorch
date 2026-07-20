const byId = id => document.getElementById(id);

const PADDING = 32;
const HIT_CELL = 18;
const HIT_RADIUS = 12;
const MAX_ZOOM = 40;
const palette = [
  "#63d59a", "#f1b85b", "#ff786f", "#80b9ff", "#d89cff", "#e7df73",
  "#72d9d2", "#f39ac1", "#a8d56a", "#ff9f72", "#94a7ff", "#d6a96f",
];

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function dataBounds(points) {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const point of points) {
    xMin = Math.min(xMin, point.x); xMax = Math.max(xMax, point.x);
    yMin = Math.min(yMin, point.y); yMax = Math.max(yMax, point.y);
  }
  const xPadding = xMax > xMin ? (xMax - xMin) * 0.06 : 1;
  const yPadding = yMax > yMin ? (yMax - yMin) * 0.06 : 1;
  return {
    xMin: xMin - xPadding, xMax: xMax + xPadding,
    yMin: yMin - yPadding, yMax: yMax + yPadding,
  };
}

function copyBounds(bounds) {
  return bounds ? { ...bounds } : null;
}

export function createScatterPlot({ onPointSelect, onVisibleCount }) {
  const canvas = byId("projection-canvas");
  const tooltip = byId("plot-tooltip");
  const legend = byId("legend");
  const legendControls = byId("legend-controls");
  const zoomIn = byId("plot-zoom-in");
  const zoomOut = byId("plot-zoom-out");
  const reset = byId("plot-reset");

  let points = [];
  let labels = [];
  let labelCounts = new Map();
  let colors = new Map();
  let hiddenLabels = new Set();
  let home = null;
  let view = null;
  let zoomLevel = 1;
  let hitGrid = new Map();
  let selectedRowId = null;
  let drag = null;
  let suppressClick = false;
  let drawFrame = 0;
  let hoverFrame = 0;
  let pendingHover = null;
  let legendClickTimer = 0;

  function cancelLegendClick() {
    clearTimeout(legendClickTimer);
    legendClickTimer = 0;
  }

  function visibleCount() {
    let count = 0;
    for (const [label, labelCount] of labelCounts) {
      if (!hiddenLabels.has(label)) count += labelCount;
    }
    return count;
  }

  function updateCount() {
    onVisibleCount(visibleCount(), points.length);
  }

  function updateNavigation() {
    zoomIn.disabled = !view || zoomLevel >= MAX_ZOOM;
    zoomOut.disabled = !view || zoomLevel <= 1;
    reset.disabled = !view;
  }

  function hideTooltip() {
    pendingHover = null;
    tooltip.classList.add("hidden");
  }

  function renderLegend() {
    cancelLegendClick();
    legend.replaceChildren();
    legendControls.classList.toggle("hidden", !labels.length);
    byId("legend-summary").textContent = `${labels.length - hiddenLabels.size}/${labels.length} shown`;
    byId("legend-show-all").disabled = hiddenLabels.size === 0;
    byId("legend-hide-all").disabled = hiddenLabels.size === labels.length;

    for (const label of labels) {
      const visible = !hiddenLabels.has(label);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `legend-item${visible ? "" : " is-hidden"}`;
      button.setAttribute("aria-pressed", String(visible));
      button.title = `${visible ? "Hide" : "Show"} ${label}; double-click to isolate`;

      const swatch = document.createElement("i");
      swatch.className = "legend-swatch";
      swatch.style.background = colors.get(label);
      const text = document.createElement("span");
      text.className = "legend-label";
      text.textContent = label;
      const count = document.createElement("span");
      count.className = "legend-count";
      count.textContent = labelCounts.get(label).toLocaleString();
      button.append(swatch, text, count);
      button.addEventListener("click", event => {
        if (event.detail > 1) {
          cancelLegendClick();
          hiddenLabels = new Set(labels.filter(item => item !== label));
          hideTooltip(); renderLegend(); updateCount(); scheduleDraw();
          return;
        }
        cancelLegendClick();
        legendClickTimer = setTimeout(() => {
          legendClickTimer = 0;
          hiddenLabels.has(label) ? hiddenLabels.delete(label) : hiddenLabels.add(label);
          hideTooltip(); renderLegend(); updateCount(); scheduleDraw();
        }, 180);
      });
      legend.append(button);
    }
  }

  function scheduleDraw() {
    if (drawFrame) return;
    drawFrame = requestAnimationFrame(() => {
      drawFrame = 0;
      draw();
    });
  }

  function drawGrid(context, width, height) {
    context.strokeStyle = "#252925";
    context.lineWidth = 1;
    context.beginPath();
    for (let index = 0; index <= 4; index += 1) {
      const x = PADDING + (width - PADDING * 2) * index / 4;
      const y = PADDING + (height - PADDING * 2) * index / 4;
      context.moveTo(x, PADDING); context.lineTo(x, height - PADDING);
      context.moveTo(PADDING, y); context.lineTo(width - PADDING, y);
    }
    context.stroke();
  }

  function draw() {
    const bounds = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.max(1, Math.floor(bounds.width * ratio));
    const pixelHeight = Math.max(1, Math.floor(bounds.height * ratio));
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, bounds.width, bounds.height);
    hitGrid = new Map();
    if (!view || bounds.width <= PADDING * 2 || bounds.height <= PADDING * 2) return;

    drawGrid(context, bounds.width, bounds.height);
    const plotWidth = bounds.width - PADDING * 2;
    const plotHeight = bounds.height - PADDING * 2;
    const xSpan = view.xMax - view.xMin || 1;
    const ySpan = view.yMax - view.yMin || 1;
    const toX = value => PADDING + (value - view.xMin) / xSpan * plotWidth;
    const toY = value => bounds.height - PADDING - (value - view.yMin) / ySpan * plotHeight;
    const buckets = new Map();
    let selectedPoint = null;

    for (const point of points) {
      if (hiddenLabels.has(point.label)) continue;
      const screenPoint = { ...point, sx: toX(point.x), sy: toY(point.y) };
      if (
        screenPoint.sx < PADDING - 4 || screenPoint.sx > bounds.width - PADDING + 4 ||
        screenPoint.sy < PADDING - 4 || screenPoint.sy > bounds.height - PADDING + 4
      ) continue;
      if (!buckets.has(point.label)) buckets.set(point.label, []);
      buckets.get(point.label).push(screenPoint);
      const cellX = Math.floor(screenPoint.sx / HIT_CELL);
      const cellY = Math.floor(screenPoint.sy / HIT_CELL);
      const key = `${cellX}:${cellY}`;
      if (!hitGrid.has(key)) hitGrid.set(key, []);
      hitGrid.get(key).push(screenPoint);
      if (point.row_id === selectedRowId) selectedPoint = screenPoint;
    }

    context.save();
    context.beginPath();
    context.rect(PADDING, PADDING, plotWidth, plotHeight);
    context.clip();
    context.globalAlpha = 0.76;
    for (const [label, bucket] of buckets) {
      context.beginPath();
      for (const point of bucket) {
        context.moveTo(point.sx + 3.2, point.sy);
        context.arc(point.sx, point.sy, 3.2, 0, Math.PI * 2);
      }
      context.fillStyle = colors.get(label);
      context.fill();
    }
    context.globalAlpha = 1;
    if (selectedPoint) {
      context.beginPath();
      context.arc(selectedPoint.sx, selectedPoint.sy, 6.5, 0, Math.PI * 2);
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.5;
      context.stroke();
    }
    context.restore();
  }

  function nearestPoint(clientX, clientY) {
    const bounds = canvas.getBoundingClientRect();
    const x = clientX - bounds.left;
    const y = clientY - bounds.top;
    const cellX = Math.floor(x / HIT_CELL);
    const cellY = Math.floor(y / HIT_CELL);
    let nearest = null;
    let nearestDistance = HIT_RADIUS * HIT_RADIUS;
    for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        for (const point of hitGrid.get(`${cellX + offsetX}:${cellY + offsetY}`) || []) {
          const distance = (point.sx - x) ** 2 + (point.sy - y) ** 2;
          if (distance < nearestDistance) {
            nearestDistance = distance;
            nearest = point;
          }
        }
      }
    }
    return { point: nearest, x, y };
  }

  function renderHover(clientX, clientY) {
    const { point, x, y } = nearestPoint(clientX, clientY);
    tooltip.classList.toggle("hidden", !point);
    if (!point) return;
    tooltip.textContent = `${point.item_id || `Row ${point.row_id}`} | ${point.label}`;
    const bounds = canvas.getBoundingClientRect();
    tooltip.style.left = `${Math.max(4, Math.min(x + 10, bounds.width - tooltip.offsetWidth - 4))}px`;
    tooltip.style.top = `${Math.max(4, Math.min(y + 10, bounds.height - tooltip.offsetHeight - 4))}px`;
  }

  function queueHover(event) {
    pendingHover = { x: event.clientX, y: event.clientY };
    if (hoverFrame) return;
    hoverFrame = requestAnimationFrame(() => {
      hoverFrame = 0;
      if (pendingHover) renderHover(pendingHover.x, pendingHover.y);
    });
  }

  function resetView() {
    view = copyBounds(home);
    zoomLevel = 1;
    hideTooltip();
    updateNavigation();
    scheduleDraw();
  }

  function zoomAt(targetZoom, clientX, clientY) {
    if (!view || !home) return;
    targetZoom = clamp(targetZoom, 1, MAX_ZOOM);
    if (targetZoom <= 1.001) {
      resetView();
      return;
    }
    const bounds = canvas.getBoundingClientRect();
    const plotWidth = Math.max(1, bounds.width - PADDING * 2);
    const plotHeight = Math.max(1, bounds.height - PADDING * 2);
    const ratioX = clamp((clientX - bounds.left - PADDING) / plotWidth, 0, 1);
    const ratioY = clamp((clientY - bounds.top - PADDING) / plotHeight, 0, 1);
    const anchorX = view.xMin + ratioX * (view.xMax - view.xMin);
    const anchorY = view.yMax - ratioY * (view.yMax - view.yMin);
    const xSpan = (home.xMax - home.xMin) / targetZoom;
    const ySpan = (home.yMax - home.yMin) / targetZoom;
    view = {
      xMin: anchorX - ratioX * xSpan,
      xMax: anchorX + (1 - ratioX) * xSpan,
      yMin: anchorY - (1 - ratioY) * ySpan,
      yMax: anchorY + ratioY * ySpan,
    };
    zoomLevel = targetZoom;
    hideTooltip();
    updateNavigation();
    scheduleDraw();
  }

  canvas.addEventListener("wheel", event => {
    if (!view) return;
    event.preventDefault();
    const delta = clamp(event.deltaY, -120, 120);
    zoomAt(zoomLevel * Math.exp(-delta * 0.002), event.clientX, event.clientY);
  }, { passive: false });

  canvas.addEventListener("pointerdown", event => {
    if (!view || event.button !== 0) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    drag = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      view: copyBounds(view),
      moved: false,
    };
    canvas.classList.add("panning");
    hideTooltip();
  });

  canvas.addEventListener("pointermove", event => {
    if (!drag || drag.pointerId !== event.pointerId) {
      queueHover(event);
      return;
    }
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    if (!drag.moved && Math.hypot(deltaX, deltaY) < 4) return;
    drag.moved = true;
    const bounds = canvas.getBoundingClientRect();
    const plotWidth = Math.max(1, bounds.width - PADDING * 2);
    const plotHeight = Math.max(1, bounds.height - PADDING * 2);
    const xShift = -deltaX / plotWidth * (drag.view.xMax - drag.view.xMin);
    const yShift = deltaY / plotHeight * (drag.view.yMax - drag.view.yMin);
    view = {
      xMin: drag.view.xMin + xShift,
      xMax: drag.view.xMax + xShift,
      yMin: drag.view.yMin + yShift,
      yMax: drag.view.yMax + yShift,
    };
    scheduleDraw();
  });

  function stopDragging(event) {
    if (!drag || drag.pointerId !== event.pointerId) return;
    const moved = drag.moved;
    drag = null;
    canvas.classList.remove("panning");
    if (moved) {
      suppressClick = true;
      setTimeout(() => { suppressClick = false; }, 0);
    }
  }

  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);
  canvas.addEventListener("lostpointercapture", stopDragging);
  canvas.addEventListener("pointerleave", () => { if (!drag) hideTooltip(); });
  canvas.addEventListener("click", event => {
    if (suppressClick) return;
    const { point } = nearestPoint(event.clientX, event.clientY);
    if (!point) return;
    selectedRowId = point.row_id;
    scheduleDraw();
    onPointSelect(point);
  });

  zoomIn.addEventListener("click", () => {
    const bounds = canvas.getBoundingClientRect();
    zoomAt(zoomLevel * 1.35, bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
  });
  zoomOut.addEventListener("click", () => {
    const bounds = canvas.getBoundingClientRect();
    zoomAt(zoomLevel / 1.35, bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
  });
  reset.addEventListener("click", resetView);
  byId("legend-show-all").addEventListener("click", () => {
    cancelLegendClick();
    hiddenLabels.clear();
    renderLegend(); updateCount(); scheduleDraw();
  });
  byId("legend-hide-all").addEventListener("click", () => {
    cancelLegendClick();
    hiddenLabels = new Set(labels);
    hideTooltip(); renderLegend(); updateCount(); scheduleDraw();
  });
  new ResizeObserver(scheduleDraw).observe(canvas.parentElement);

  return {
    setData(nextPoints, legendTitle = "Subclasses") {
      points = nextPoints;
      labelCounts = new Map();
      for (const point of points) {
        labelCounts.set(point.label, (labelCounts.get(point.label) || 0) + 1);
      }
      labels = [...labelCounts.keys()].sort();
      colors = new Map(labels.map((label, index) => [label, palette[index % palette.length]]));
      hiddenLabels = new Set();
      home = points.length ? dataBounds(points) : null;
      view = copyBounds(home);
      zoomLevel = 1;
      selectedRowId = null;
      byId("legend-title").textContent = legendTitle || "Subclasses";
      hideTooltip();
      renderLegend();
      updateCount();
      updateNavigation();
      scheduleDraw();
    },
    redraw: scheduleDraw,
    resetView,
  };
}
