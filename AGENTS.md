# gpu-ssimulacra2 — agent guide

## Status

Numerically verified against C++ reference (70.75 vs 70.68, 0.1% difference from GPU f32 precision). WebGPU port of the SSIMULACRA2 perceptual metric: sRGB input
→ linear → XYB → 6-scale Gaussian pyramid → SSIM + EdgeDiff → weighted score. Score
70.7 for Lena vs Lena JPEG (quality ~75), 100.0 for identical images.

## Project structure

```
src/
  lib/
    index.js          — public API (re-exports)
    pipeline.js       — WebGPU init, ceilDiv, etc.
    shaders.js        — all 10 WGSL shader modules as JS template strings
    ssimulacra2.js    — SSIMULACRA2Pipeline class (orchestrator)
    scoring.js        — computeScore() with 108-weight C++ reference formula
    gaussian-blur.js  — CreateRecursiveGaussian coeffs (JS translation)
  demo/
    index.html        — demo page
    main.js           — file upload, canvas -> ImageData, pipeline call
    style.css         — styles
original/             — untracked C++ reference from libjxl
```

## Running

```sh
npm run dev     # Vite dev server at localhost:5173
npm run build   # production build → dist/
```

Open browser at localhost:5173, upload two images, click Compare.

## Pipeline architecture

Each scale creates a `CommandEncoder` with ~17 sequential compute passes:

1. `to_linear` — sRGB RGBA u8 → 3× f32 linear planes (orig + dist)
2. `to_xyb` — linear RGB → XYB via OpsinAbsorbance matrix + MakePositiveXYB
3-6. `blur1-h/v`, `blur2-h/v` — separable IIR Gaussian for mu1, mu2
7-15. `s1/s2/s3-mul/h/v` — element-wise multiply then blur for σ1², σ2², σ12
16. `wg` — per-workgroup SSIM + EdgeDiff reduction (separate intermediate buffers)
17. `rd` — global reduction to final 18-values-per-scale output

Buffer sync rule: a buffer used as read-write in one dispatch cannot be read-only
in another dispatch within the same compute pass. Each producer→consumer chain
gets its own pass.

## Known differences from C++ reference

- **WebGPU only** — no CPU fallback
- **SDR only** — Opsin constants hardcoded; no PQ/HLG support
- **Coefficient layout** — `n2` and `d1` scalar (not duplicated ×4 for SIMD lanes)
- **Warmup** — each gauss_h/gauss_v shader iterates from `-N+1` to `-1` before writing output (same as C++ reference's `ptrdiff_t n = -N + 1` initialization). CRITICAL: without this warmup, the IIR filter (poles on unit circle) produces incorrect outputs that cause negative variances in the SSIM computation.
- **Output normalization** — SSIM sum divided by pixel count to match reference
- **Scoring weights** — copied from reference; may produce slightly different scores
  for the same images due to float ordering differences
- **Downsampling params fix** — the `downsample` shader must receive the PREVIOUS
  scale's dimensions in `Params`, not the current scale's. The fix uses
  `paramBufs[(si - 1) * 14]` as the params buffer for the downsample pass
  (`src/lib/ssimulacra2.js:249`).

## Buffer naming

| Prefix | Purpose |
|--------|---------|
| `i1*`/`i2*` | XYB planes for orig/dist at current scale |
| `m1*`/`m2*` | Gaussian-blurred means (mu1, mu2) |
| `s*` | σ1² = blurred(i1·i1) |
| `s2*` | σ2² = blurred(i2·i2) |
| `c12*` | σ12 = blurred(i1·i2) |
| `m*` | temp product buffer (multiply destination) |
| `t*` | temp blur intermediate (gauss_h → gauss_v) |

## Debugging

All‑zeros output usually means a buffer sync-scope violation. Add
`device.pushErrorScope("validation")` / `popErrorScope()` around submits to
catch them.
