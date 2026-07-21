// Small, reusable DOM render helpers.

export const byId = id => document.getElementById(id);

export function escapeText(value) {
  const element = document.createElement("span");
  element.textContent = value ?? "Missing";
  return element.innerHTML;
}

export function metric(label, value) {
  return `<div class="metric"><strong>${escapeText(value)}</strong><span>${escapeText(label)}</span></div>`;
}

export function imagePane(url, label) {
  if (!url) return `<div class="card-image"><div class="missing-image">No ${escapeText(label)}</div></div>`;
  return `<button class="card-image" type="button" data-image="${escapeText(url)}" data-label="${escapeText(label)}"><img loading="lazy" src="${escapeText(url)}" alt="${escapeText(label)}"><span class="image-label">${escapeText(label)}</span></button>`;
}

export const formatScore = value => value == null ? "–" : Number(value).toFixed(4);
export const formatPercent = value => `${(Number(value || 0) * 100).toFixed(2)}%`;

export function bindGalleryImages(containerId, openImageViewer) {
  const container = byId(containerId);
  container.addEventListener("click", event => {
    const pane = event.target.closest("[data-image]");
    if (pane) openImageViewer(pane.dataset.image, pane.dataset.label || "Image preview");
  });
  container.addEventListener("error", event => {
    if (event.target.tagName !== "IMG") return;
    const pane = event.target.closest(".card-image");
    if (pane) pane.innerHTML = '<div class="missing-image">Image file unavailable</div>';
  }, true);
}
