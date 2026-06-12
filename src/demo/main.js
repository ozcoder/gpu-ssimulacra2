import { initWebGPU, SSIMULACRA2Pipeline, computeScore } from "../lib/index.js";

const MAX_DIM = 4096;

const $ = (id) => document.getElementById(id);
const el = {
  nogpu: $("nogpu"),
  error: $("error"),
  origInput: $("file-orig"),
  distInput: $("file-dist"),
  previewOrig: $("preview-orig"),
  previewDist: $("preview-dist"),
  placeholderOrig: $("placeholder-orig"),
  placeholderDist: $("placeholder-dist"),
  boxOrig: $("box-orig"),
  boxDist: $("box-dist"),
  btn: $("btn-compare"),
  result: $("result"),
  resultLabel: $("result-label"),
  canvas: $("canvas"),
};

let origData = null;
let distData = null;
let pipeline = null;

function showError(msg) {
  el.error.textContent = msg;
  el.error.style.display = "block";
}

function hideError() {
  el.error.textContent = "";
  el.error.style.display = "none";
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      img._blobUrl = url;
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to decode"));
    };
    img.src = url;
  });
}

function imageToData(img) {
  const c = el.canvas;
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const id = ctx.getImageData(0, 0, img.width, img.height);
  return { width: img.width, height: img.height, data: id.data.buffer };
}

async function handleFile(file, isOrig) {
  hideError();
  const img = await loadImage(file);
  const preview = isOrig ? el.previewOrig : el.previewDist;
  const placeholder = isOrig ? el.placeholderOrig : el.placeholderDist;
  const box = isOrig ? el.boxOrig : el.boxDist;
  preview.src = img._blobUrl;
  preview.onload = () => URL.revokeObjectURL(img._blobUrl);
  preview.style.display = "block";
  placeholder.style.display = "none";
  box.style.borderColor = "#4caf50";

  const data = imageToData(img);
  if (isOrig) {
    origData = data;
  } else {
    distData = data;
  }

  if (origData && distData) {
    const dim = Math.max(origData.width, origData.height);
    if (dim > MAX_DIM) {
      showError(`Images must be ≤${MAX_DIM}px on the longest side`);
      el.btn.disabled = true;
      return;
    }
    el.btn.disabled = false;
  }
}

el.origInput.addEventListener("change", () => {
  if (el.origInput.files[0]) handleFile(el.origInput.files[0], true);
});
el.distInput.addEventListener("change", () => {
  if (el.distInput.files[0]) handleFile(el.distInput.files[0], false);
});

// Drag-and-drop
[el.boxOrig, el.boxDist].forEach((box, i) => {
  box.addEventListener("dragover", (e) => { e.preventDefault(); box.classList.add("dragover"); });
  box.addEventListener("dragleave", () => box.classList.remove("dragover"));
  box.addEventListener("drop", (e) => {
    e.preventDefault();
    box.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, i === 0);
  });
});

el.btn.addEventListener("click", async () => {
  if (!origData || !distData) return;
  hideError();
  el.btn.disabled = true;
  el.btn.textContent = "Computing…";
  el.result.textContent = "";

  try {
    if (!pipeline) {
      const { device } = await initWebGPU();
      pipeline = new SSIMULACRA2Pipeline(device);
      await pipeline.allocate(MAX_DIM, MAX_DIM);
    }

    const scores = await pipeline.compute(origData, distData);
    const numScales = scores.length / 18;
    console.log("=== Raw sub-scores per scale ===");
    for (let s = 0; s < numScales; s++) {
      const off = s * 18;
      const ssim = Array.from(scores.subarray(off, off + 6)).map(v => v.toFixed(6));
      const ed = Array.from(scores.subarray(off + 6, off + 18)).map(v => v.toFixed(6));
      console.log(`  scale ${s} (1:${1<<s}) ssim=[${ssim.join(",")}] ed=[${ed.join(",")}]`);
    }
    const score = computeScore(scores, numScales);
    el.result.textContent = score.toFixed(1);
    el.resultLabel.textContent = "/ 100 — SSIMULACRA2 score";
  } catch (err) {
    showError(err.message);
  } finally {
    el.btn.disabled = false;
    el.btn.textContent = "Compare";
  }
});

// Check WebGPU availability
(async () => {
  if (!navigator.gpu) {
    el.nogpu.style.display = "block";
    el.btn.disabled = true;
  }
})();
