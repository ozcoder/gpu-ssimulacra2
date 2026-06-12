# gpu-ssimulacra2

**WebGPU‑accelerated SSIMULACRA2** — a perceptual image quality metric that runs entirely on the GPU, in the browser.

Analyze two images and get a score from **0 (completely different) to 100 (identical)**, matching the libjxl reference implementation.

## Demo

Open `index.html` via Vite, upload two images, click **Compare**.

| Feature | Detail |
|---|---|
| Input | Any image format the browser can decode (PNG, JPEG, WebP, AVIF, …) |
| Upload | File picker or drag‑and‑drop onto either box |
| Preview | Thumbnails shown after upload; green border confirms loaded |
| Result | Score (0–100) displayed below the Compare button |
| Limit | Images up to **2048 px** on the longest side |
| Fallback | Shows error if WebGPU is unavailable |

## Quick start

```sh
npm install
npm run dev      # Vite dev server → http://localhost:5173
npm run build    # production build → dist/
```

## Library API

```js
import { initWebGPU, SSIMULACRA2Pipeline, computeScore } from "gpu-ssimulacra2";
```

### `initWebGPU()`

Requests a WebGPU adapter and device. Returns `{ adapter, device }`.
Throws if WebGPU is not supported.

### `SSIMULACRA2Pipeline`

Orchestrates the full GPU pipeline. Create once, reuse for multiple pairs.

```js
const pipeline = new SSIMULACRA2Pipeline(device);
await pipeline.allocate(maxWidth, maxHeight);

const result = await pipeline.compute(
  { width, height, data: origBuffer },
  { width, height, data: distBuffer },
);
```

- `allocate(maxW, maxH)` — pre‑allocates all GPU buffers for the given maximum dimensions.
- `compute(orig, dist)` — runs the 6‑scale pipeline on the GPU. Returns a `Float32Array` of **18 × N** values (N = number of scales that fit, up to 6). `orig.data` and `dist.data` must be `ArrayBuffer` views of raw RGBA `Uint8` pixels.

### `computeScore(scores, numScales)`

Applies the 108‑weight formula and nonlinear post‑processing. Returns the final 0–100 score.

```js
const numScales = scores.length / 18;
const score = computeScore(scores, numScales);
```

## How it works

The pipeline runs **6 scales** (1:1, 1:2, 1:4, … 1:32), processing each in sequence.
For every scale, 17 compute passes execute on the GPU:

1. **sRGB → linear** (scale 0) / **2× box downsample** (scales 1–5) — in linear RGB
2. **Linear → XYB** — OpsinAbsorbance matrix + cube root + MakePositiveXYB
3. **IIR Gaussian blur** — separable recursive filter, 3‑pole, one thread per row/column
4. **Element‑wise multiply** — i₁·i₁, i₂·i₂, i₁·i₂ for σ₁², σ₂², σ₁₂
5. **Per‑workgroup reduction** — SSIM + EdgeDiff maps, summed into shared memory
6. **Global reduction** — sums workgroup results, normalizes by pixel count, writes 18 values

Results are copied to a staging buffer for CPU readback. See `AGENTS.md` for the full buffer naming scheme and pass‑splitting rationale.

## Limitations

- **WebGPU only** — no CPU fallback
- **SDR only** — Opsin constants are hardcoded; no PQ/HLG support
- **Browser only** — requires `navigator.gpu`; not available in Node.js
- **Precision** — intermediate accumulation uses `f32`, not `double`
  (estimated score difference < 0.1 vs the C++ reference)

## References

- [libjxl / SSIMULACRA2](https://github.com/libjxl/libjxl/tree/main/tools) — C++ reference implementation
- Jon Sneyers, *SSIMULACRA2: Improved image quality assessment*, Cloudinary Blog, 2022
- Charalampidis, *Recursive Gaussian filter using truncated cosine functions*, 2016
