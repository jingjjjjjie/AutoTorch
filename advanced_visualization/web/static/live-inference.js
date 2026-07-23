import { getLiveModels, runLiveInference } from "./api.js";
import { byId } from "./dom.js";
import { openImageViewer } from "./image-viewer.js";

let selectedFile = null;
let previewUrl = "";
let setBusy = () => {};
let showError = () => {};

const percent = value => `${(Number(value) * 100).toFixed(2)}%`;

function setPreview(file) {
  selectedFile = file || null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : "";
  byId("live-empty").classList.toggle("hidden", Boolean(file));
  byId("live-preview-wrap").classList.toggle("hidden", !file);
  byId("live-preview").src = previewUrl;
  byId("live-file-name").textContent = file
    ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`
    : "PNG, JPEG, WebP, or BMP up to 20 MB";
  byId("run-live-inference").disabled = !file;
  byId("live-results").classList.add("hidden");
}

function renderResult(result) {
  const isFraud = result.predicted_class === "fraud";
  const badge = byId("live-prediction-badge");
  badge.textContent = result.predicted_class;
  badge.className = `live-prediction ${isFraud ? "fraud" : "genuine"}`;
  byId("live-fraud-score").textContent = percent(result.fraud_probability);
  byId("live-genuine-score").textContent = percent(result.genuine_probability);
  byId("live-fraud-bar").style.width = percent(result.fraud_probability);
  byId("live-genuine-bar").style.width = percent(result.genuine_probability);
  byId("live-result-meta").textContent =
    `${result.width} × ${result.height}${result.branch ? ` · ${result.branch}` : ""} · logit ${Number(result.logit).toFixed(4)} · ${result.elapsed_ms.toLocaleString()} ms`;
  const cams = byId("live-cams");
  cams.replaceChildren();
  cams.append(imageCard(result.input_image || previewUrl, "Input image", "Image sent to the selected model"));
  const layers = Array.isArray(result.gradcams) && result.gradcams.length
    ? result.gradcams
    : [{ label: "Target layer", genuine_gradcam: result.genuine_gradcam, fraud_gradcam: result.fraud_gradcam }];
  for (const layer of layers) {
    cams.append(
      imageCard(layer.genuine_gradcam, `${layer.label} · Genuine`, "Regions supporting the genuine class"),
      imageCard(layer.fraud_gradcam, `${layer.label} · Fraud`, "Regions supporting the fraud class"),
    );
  }
  byId("live-results").classList.remove("hidden");
  byId("live-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function imageCard(src, label, description) {
  const figure = document.createElement("figure");
  const caption = document.createElement("figcaption");
  const title = document.createElement("strong");
  const note = document.createElement("span");
  const button = document.createElement("button");
  const image = document.createElement("img");
  title.textContent = label;
  note.textContent = description;
  caption.append(title, note);
  button.className = "card-image";
  button.type = "button";
  image.src = src;
  image.alt = label;
  button.append(image);
  button.addEventListener("click", () => openImageViewer(src, label));
  figure.append(caption, button);
  return figure;
}

async function submit() {
  if (!selectedFile) {
    showError("Choose an image before running inference.");
    return;
  }
  showError("");
  setBusy(true);
  byId("run-live-inference").disabled = true;
  try {
    const result = await runLiveInference(
      selectedFile,
      byId("live-model").value,
      Number(byId("live-threshold").value),
      byId("live-method").value,
    );
    renderResult(result);
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
    byId("run-live-inference").disabled = !selectedFile;
  }
}

export async function configureLiveInference() {
  const models = await getLiveModels();
  const select = byId("live-model");
  select.replaceChildren();
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.key;
    const framework = model.framework ? ` · ${model.framework}` : "";
    const layers = Array.isArray(model.layers) ? ` · ${model.layers.length} layers` : "";
    option.textContent = `${model.label} · ${model.image_size}px${framework}${layers}${model.available ? "" : " (unavailable)"}`;
    option.disabled = !model.available;
    option.dataset.threshold = String(model.threshold);
    select.append(option);
  }
  const firstAvailable = [...select.options].find(option => !option.disabled);
  if (firstAvailable) {
    select.value = firstAvailable.value;
    byId("live-threshold").value = firstAvailable.dataset.threshold || "0.5";
  }
  byId("run-live-inference").disabled = !selectedFile || !firstAvailable;
  byId("live-model-note").textContent = firstAvailable
    ? `${models.filter(model => model.available).length} model(s) ready. TensorFlow VAN Small returns Genuine and Fraud maps for norm3, block3.3, block4.1, and norm4.`
    : (models.find(model => model.status)?.status || "No configured model checkpoint is currently available.");
}

export function bindLiveInference(busyHandler, errorHandler) {
  setBusy = busyHandler;
  showError = errorHandler;
  const input = byId("live-file");
  const dropzone = byId("live-dropzone");
  input.addEventListener("change", () => setPreview(input.files?.[0]));
  dropzone.addEventListener("click", event => {
    if (event.target.closest("button")) return;
    input.click();
  });
  dropzone.addEventListener("dragover", event => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
  dropzone.addEventListener("drop", event => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
    setPreview(event.dataTransfer.files?.[0]);
  });
  byId("choose-live-file").addEventListener("click", event => {
    event.stopPropagation();
    input.click();
  });
  byId("run-live-inference").addEventListener("click", submit);
  byId("live-model").addEventListener("change", event => {
    const option = event.target.selectedOptions[0];
    if (option?.dataset.threshold) byId("live-threshold").value = option.dataset.threshold;
  });
}
