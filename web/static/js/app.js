import { Viewer3D } from "./viewer3d.js";

const viewer = new Viewer3D(document.querySelector("#viewer3d"));
const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const chooseFile = document.querySelector("#chooseFile");
const uploadButton = document.querySelector("#uploadButton");
const clearButton = document.querySelector("#clearButton");
const analyzeButton = document.querySelector("#analyzeButton");
const fileDetails = document.querySelector("#fileDetails");
const statusText = document.querySelector("#statusText");
const statusDetail = document.querySelector("#statusDetail");
let selectedFile = null;

// Slice viewer state
let currentVolumeId = null;
let currentAxis = "axial";
let currentSliceIndex = 0;
let sliceTotals = { axial: 0, coronal: 0, sagittal: 0 };

chooseFile.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => setSelectedFile(event.dataTransfer.files[0]));
clearButton.addEventListener("click", clearSelection);
uploadButton.addEventListener("click", uploadFile);
analyzeButton.addEventListener("click", analyze);
document.querySelector("#fullscreenButton").addEventListener("click", () => viewer.toggleFullscreen());
document.querySelector("#resetCamera").addEventListener("click", () => viewer.resetCamera());
document.querySelector("#demoToggle").addEventListener("change", (event) => {
  viewer.setDemoObject(event.target.checked);
  document.querySelector("#demoLabel").hidden = !event.target.checked;
  document.querySelector("#viewerEmpty").hidden = event.target.checked;
});
document.querySelector("#wireframeToggle").addEventListener("change", (event) => viewer.setWireframe(event.target.checked));
document.querySelector("#opacityRange").addEventListener("input", (event) => {
  const value = Number(event.target.value) / 100;
  viewer.setOpacity(value);
  document.querySelector("#opacityValue").value = `${event.target.value}%`;
  document.querySelector("#opacityValue").textContent = `${event.target.value}%`;
});

// Slice viewer controls
document.querySelectorAll(".axis-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".axis-btn").forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    currentAxis = btn.dataset.axis;
    currentSliceIndex = Math.floor(sliceTotals[currentAxis] / 2);
    updateSliderRange();
    loadSlice();
  });
});
document.querySelector("#slicePrev").addEventListener("click", () => {
  if (currentSliceIndex > 0) { currentSliceIndex--; updateSliderRange(); loadSlice(); }
});
document.querySelector("#sliceNext").addEventListener("click", () => {
  if (currentSliceIndex < sliceTotals[currentAxis] - 1) { currentSliceIndex++; updateSliderRange(); loadSlice(); }
});
document.querySelector("#sliceSlider").addEventListener("input", (event) => {
  currentSliceIndex = Number(event.target.value);
  loadSlice();
});

function updateSliderRange() {
  const total = sliceTotals[currentAxis] || 1;
  const slider = document.querySelector("#sliceSlider");
  slider.max = total - 1;
  slider.value = currentSliceIndex;
  document.querySelector("#sliceIndexOut").textContent = currentSliceIndex;
  document.querySelector("#sliceCounter").textContent = `${currentSliceIndex + 1} / ${total}`;
}

async function loadSlice() {
  if (!currentVolumeId || sliceTotals[currentAxis] === 0) return;
  const url = `/api/volume/${currentVolumeId}/slice/${currentAxis}/${currentSliceIndex}`;
  try {
    const response = await fetch(url);
    if (!response.ok) return;
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const canvas = document.querySelector("#sliceCanvas");
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(objectUrl);
    };
    img.src = objectUrl;
    document.querySelector("#sliceEmpty").hidden = true;
    document.querySelector("#sliceIndexOut").textContent = currentSliceIndex;
    document.querySelector("#sliceCounter").textContent = `${currentSliceIndex + 1} / ${sliceTotals[currentAxis]}`;
  } catch (_) { /* network error — leave current slice */ }
}

function showSliceViewer(metadata) {
  const section = document.querySelector("#sliceviewer");
  section.hidden = false;
  sliceTotals = {
    axial: metadata.shape[2],
    coronal: metadata.shape[1],
    sagittal: metadata.shape[0],
  };
  currentSliceIndex = Math.floor(sliceTotals[currentAxis] / 2);
  updateSliderRange();
  loadSlice();
  renderMetaTable(metadata);
}

function renderMetaTable(meta) {
  const table = document.querySelector("#volumeMetaTable");
  const rows = [
    ["Filename", escapeHtml(meta.filename)],
    ["Shape", meta.shape.join(" × ")],
    ["Voxel spacing (mm)", meta.voxel_spacing.map((v) => v.toFixed(3)).join(", ")],
    ["Orientation", meta.orientation.join("")],
    ["Data type", meta.dtype],
    ["Min intensity", meta.min_intensity?.toFixed(4) ?? "--"],
    ["Max intensity", meta.max_intensity?.toFixed(4) ?? "--"],
    ["Mean intensity", meta.mean_intensity?.toFixed(4) ?? "--"],
    ["Median intensity", meta.median_intensity?.toFixed(4) ?? "--"],
    ["Slices (axial)", meta.number_of_slices],
  ];
  table.innerHTML = rows.map(([k, v]) => `<div class="meta-row"><span>${k}</span><strong>${v}</strong></div>`).join("");
}

function setSelectedFile(file) {
  if (!file) return;
  selectedFile = file;
  fileDetails.innerHTML = `<span class="empty-symbol">FILE</span><strong>${escapeHtml(file.name)}</strong><span>${fileType(file.name)} / ${formatBytes(file.size)}</span><span class="file-validation">Ready for local validation</span>`;
  uploadButton.disabled = false;
  statusText.textContent = "Input selected";
  statusDetail.textContent = "Validate the file to make it available to the local session.";
}

async function uploadFile() {
  if (!selectedFile) return;
  uploadButton.disabled = true;
  uploadButton.textContent = "Validating...";
  const formData = new FormData();
  formData.append("file", selectedFile);
  try {
    const response = await fetch("/api/upload", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || "Upload failed");
    fileDetails.querySelector(".file-validation").textContent = payload.upload.validation.message;
    statusText.textContent = "Input validated";
    if (payload.upload.file_type === "3d_volume") {
      statusDetail.textContent = "3D NIfTI volume loaded for multi-planar slicing. 3D segmentation is unavailable until BraTS dataset and model resources are configured.";
    } else {
      statusDetail.textContent = "Classification model not loaded. Connect a trained EfficientNet-B4 checkpoint to enable analysis.";
    }
    analyzeButton.disabled = false;
    if (payload.upload.file_type === "3d_volume" && payload.upload.metadata) {
      currentVolumeId = payload.upload.volume_id;
      showSliceViewer(payload.upload.metadata);
    }
  } catch (error) {
    fileDetails.querySelector(".file-validation").textContent = error.message;
    statusText.textContent = "Validation failed";
    statusDetail.textContent = "The file was not retained by the application.";
  } finally {
    uploadButton.disabled = false;
    uploadButton.textContent = "Validate and upload";
  }
}

async function analyze() {
  analyzeButton.disabled = true;
  try {
    const response = await fetch("/api/analyze", { method: "POST" });
    const payload = await response.json();
    if (response.ok && payload.status === "completed") {
      statusText.textContent = "Analysis completed";
      statusDetail.textContent = payload.message || "2D MRI classification completed successfully.";
      if (payload.classification && payload.classification.status === "available") {
        document.querySelector("#classificationState").textContent = `${payload.classification.predicted_class} (${(payload.classification.confidence * 100).toFixed(1)}%)`;
        const desc = document.querySelector("#classificationDesc");
        if (desc) desc.textContent = `Model: ${payload.classification.model_architecture || "EfficientNet-B4"} (Epoch ${payload.classification.epoch ?? "--"})`;
      }
    } else {
      statusText.textContent = "Model not loaded";
      statusDetail.textContent = payload.message || "Classification model not loaded. Connect a trained EfficientNet-B4 checkpoint to enable analysis.";
    }
  } catch (error) {
    statusText.textContent = "Model not loaded";
    statusDetail.textContent = "Classification model not loaded. Connect a trained EfficientNet-B4 checkpoint to enable analysis.";
  } finally {
    analyzeButton.disabled = false;
  }
}

function clearSelection() {
  selectedFile = null;
  fileInput.value = "";
  fileDetails.innerHTML = '<span class="empty-symbol">--</span><strong>No MRI selected</strong><span>Validation details will appear here.</span>';
  uploadButton.disabled = true;
  analyzeButton.disabled = true;
  statusText.textContent = "Waiting for MRI input";
  statusDetail.textContent = "No analysis has been performed.";
  currentVolumeId = null;
  document.querySelector("#sliceviewer").hidden = true;
}

function fileType(name) {
  return name.toLowerCase().endsWith(".nii.gz") || name.toLowerCase().endsWith(".nii") ? "3D MRI Volume" : "2D MRI";
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[c]);
}

// ── Dataset status panel ────────────────────────────────────
async function loadDatasetStatus() {
  try {
    const response = await fetch("/api/datasets");
    if (!response.ok) return;
    const data = await response.json();
    renderClassificationStatus(data.classification);
    renderSegmentationStatus(data.segmentation);
    document.querySelector("#datasetRefreshNote").textContent = "Live from disk";
  } catch (_) {
    document.querySelector("#datasetRefreshNote").textContent = "Status unavailable";
  }
}

function renderClassificationStatus(cls) {
  const pill = document.querySelector("#dsClassPill");
  const stats = document.querySelector("#dsClassStats");
  const ready = cls.status === "ready";
  pill.textContent = ready ? "READY" : "NOT DOWNLOADED";
  pill.className = "state-pill " + (ready ? "ready" : "not-ready");
  if (ready) {
    const classRows = Object.entries(cls.class_counts || {}).map(([k, v]) => `<div class="ds-row"><span>${k}</span><strong>${v}</strong></div>`).join("");
    const splitRows = Object.entries(cls.split_counts || {}).map(([k, v]) => `<div class="ds-row"><span>${k}</span><strong>${v} images</strong></div>`).join("");
    stats.innerHTML = `<div class="ds-row"><span>Total images</span><strong>${cls.total_images}</strong></div>${splitRows}${classRows}`;
  } else {
    stats.innerHTML = `<span>Run: <code>python3 scripts/download_classification_dataset.py</code></span>`;
  }
}

function renderSegmentationStatus(seg) {
  const pill = document.querySelector("#dsSegPill");
  const stats = document.querySelector("#dsSegStats");
  const ready = seg.status === "ready";
  pill.textContent = ready ? "READY" : "NOT IMPORTED";
  pill.className = "state-pill " + (ready ? "ready" : "not-ready");
  if (ready) {
    const splitRows = Object.entries(seg.split_counts || {}).map(([k, v]) => `<div class="ds-row"><span>${k.replace("_subjects", "")}</span><strong>${v} subjects</strong></div>`).join("");
    stats.innerHTML = `<div class="ds-row"><span>Valid subjects</span><strong>${seg.subjects}</strong></div>${splitRows}`;
  } else {
    stats.innerHTML = `<span>Run: <code>python3 scripts/import_brats_dataset.py --source /path/to/brats</code></span>`;
  }
}

// ── System status cards ─────────────────────────────────────
async function loadSystemStatus() {
  const grid = document.querySelector("#statusCardsGrid");
  const note = document.querySelector("#sysStatusNote");
  try {
    const response = await fetch("/api/system/status");
    if (!response.ok) throw new Error("status endpoint error");
    const s = await response.json();
    note.textContent = "Live from runtime";

    const cards = [
      {
        label: "Python Environment",
        value: `Python ${s.python_version}`,
        valueClass: "ok",
        sub: s.platform,
      },
      {
        label: "PyTorch",
        value: s.torch_version === "unavailable" ? "Not installed" : `v${s.torch_version}`,
        valueClass: s.torch_version === "unavailable" ? "err" : "ok",
        sub: s.torch_version === "unavailable" ? "" : "Installed",
      },
      {
        label: "Compute Device",
        value: s.compute_device,
        valueClass: s.mps_available ? "ok" : "warn",
        sub: s.mps_available ? "MPS available" : "CPU fallback",
      },
      {
        label: "2D Classifier",
        value: s.classifier_model === "available" ? "Checkpoint loaded" : "Requires Checkpoint",
        valueClass: s.classifier_model === "available" ? "ok" : "warn",
        sub: s.classifier_model === "available"
          ? s.classifier_checkpoints.slice(0, 1).join(", ")
          : "Model not loaded",
      },
      {
        label: "3D U-Net",
        value: s.segmenter_model === "available" ? "Checkpoint loaded" : "Not Imported",
        valueClass: s.segmenter_model === "available" ? "ok" : "warn",
        sub: s.segmenter_model === "available"
          ? s.segmenter_checkpoints.slice(0, 1).join(", ")
          : "Architecture ready",
      },
    ];

    grid.innerHTML = cards.map(({ label, value, valueClass, sub }) => `
      <div class="status-card-item">
        <span class="sc-label">${label}</span>
        <span class="sc-value ${valueClass}">${escapeHtml(value)}</span>
        ${sub ? `<span class="sc-sub">${escapeHtml(sub)}</span>` : ""}
      </div>`).join("");
  } catch (_) {
    note.textContent = "Unavailable";
    grid.innerHTML = `<div class="status-card-item loading-card"><span class="mono-sm">System status unavailable</span></div>`;
  }
}

// ── View tabs handler ───────────────────────────────────────
document.querySelectorAll(".view-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".view-tab").forEach((t) => t.classList.remove("selected"));
    tab.classList.add("selected");
    const placeholder = document.querySelector("#viewPlaceholder");
    if (!placeholder) return;
    if (tab.dataset.view === "original") {
      if (selectedFile) {
        placeholder.innerHTML = `<span class="placeholder-cross">MRI</span><strong>Original MRI Loaded</strong><span>${escapeHtml(selectedFile.name)} (local session input)</span>`;
      } else {
        placeholder.innerHTML = `<span class="placeholder-cross">[]</span><strong>No MRI Selected</strong><span>Upload a 2D image or 3D NIfTI volume to view input.</span>`;
      }
    } else {
      placeholder.innerHTML = `<span class="placeholder-cross">+</span><strong>Visualization output unavailable</strong><span>These views will appear after a verified model analysis.</span>`;
    }
  });
});

// ── Medical report action ───────────────────────────────────
const reportBtn = document.querySelector("#reportButton");
if (reportBtn) {
  reportBtn.addEventListener("click", async () => {
    const notice = document.querySelector("#reportNotice");
    try {
      const resp = await fetch("/api/report", { method: "POST" });
      const data = await resp.json();
      if (notice) notice.textContent = data.message || "Analysis is required before a report can be generated.";
    } catch (_) {
      if (notice) notice.textContent = "Analysis is required before a report can be generated.";
    }
  });
}

// ── Real dataset sample browser ─────────────────────────────
async function loadSamples() {
  const grid = document.querySelector("#samplesGrid");
  grid.innerHTML = `<span class="muted-text">Loading real dataset samples…</span>`;
  try {
    const response = await fetch("/api/samples/classification?per_class=3");
    if (!response.ok) throw new Error("samples endpoint error");
    const data = await response.json();
    if (data.status !== "ready" || data.samples.length === 0) {
      grid.innerHTML = `<span class="muted-text">No dataset samples available. Verify data/classification/ exists.</span>`;
      return;
    }
    grid.innerHTML = data.samples.map((s) => `
      <div class="sample-tile" data-url="${escapeHtml(s.image_url)}" data-class="${escapeHtml(s.class)}" data-file="${escapeHtml(s.filename)}" role="button" tabindex="0" aria-label="Load ${escapeHtml(s.class)} sample">
        <img src="${escapeHtml(s.image_url)}" alt="${escapeHtml(s.class)} MRI sample" loading="lazy">
        <span class="sample-tile-label">${escapeHtml(s.class)}</span>
      </div>`).join("");

    // Click → load into file details for visual preview only (no fake upload)
    grid.querySelectorAll(".sample-tile").forEach((tile) => {
      const activate = () => {
        grid.querySelectorAll(".sample-tile").forEach((t) => t.classList.remove("selected-sample"));
        tile.classList.add("selected-sample");
        const cls = tile.dataset.class;
        const file = tile.dataset.file;
        fileDetails.innerHTML = `
          <span class="empty-symbol">IMG</span>
          <strong>${escapeHtml(file)}</strong>
          <span>Dataset sample · class: ${escapeHtml(cls)}</span>
          <span class="file-validation">Real dataset image — displayed for demonstration</span>
          <img src="${escapeHtml(tile.dataset.url)}" alt="${escapeHtml(cls)} sample" style="max-width:100%;max-height:180px;margin-top:10px;border:1px solid var(--line);">`;
        statusText.textContent = "Dataset sample loaded";
        statusDetail.textContent = "This is a real dataset image shown for demonstration. Upload a file to run validation.";
      };
      tile.addEventListener("click", activate);
      tile.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); }});
    });
  } catch (_) {
    grid.innerHTML = `<span class="muted-text">Could not load samples.</span>`;
  }
}

document.querySelector("#refreshSamples").addEventListener("click", loadSamples);

// ── Initialise ──────────────────────────────────────────────
loadSystemStatus();
loadDatasetStatus();
loadSamples();
