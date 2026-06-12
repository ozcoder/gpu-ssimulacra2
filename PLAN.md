# gpu-ssimulacra2 — implementation plan

Port the SSIMULACRA2 perceptual metric to WebGPU. JavaScript library with a Vite-based demo page.

## Project structure

```
gpu-ssimulacra2/
  package.json           # vite dev dependency
  vite.config.js
  index.html             # Demo page (Vite entry)
  src/
    lib/
      index.js           # Library entry: exports computeSSIMULACRA2()
      shaders.js         # All WGSL as JS strings
      pipeline.js        # WebGPU init, pipeline & bind group factories
      ssimulacra2.js     # Main 6-scale loop orchestration
      scoring.js         # CPU weighted sum → final 0–100 score
    demo/
      main.js            # Image picker, canvas, score display
  original/              # Untouched C++ reference
```

## Data flow

```
Image A/B ──<canvas>──> ImageData ──> GPU u8 buffer
                                            │
                                 ┌──────────▼──────────┐
                                 │  sRGB → linear RGB  │
                                 │  (gamma expansion)  │
                                 └──────────┬──────────┘
                                            │
                                 ┌──────────▼──────────┐
                                 │  linear RGB → XYB   │
                                 │  MakePositiveXYB    │
                                 └──────────┬──────────┘
                                            │
              ┌─────────────────────────────┼──────────────┐
              │         scale 0             │  scales 1–5  │
              │       (full res)            │  downsample  │
              │                             │  linear RGB  │
              └─────────────┬───────────────┤  re-XYB      │
                            │               └──────────────┘
              ┌─────────────┴──────────────────────────────┐
              │  blur→mu1  blur→mu2  blur(mul=img1²)→σ1² │
              │  blur(mul=img2²)→σ2²  blur(mul=p)→σ12    │
              └─────────────────────┬─────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
       SSIMMap(mu1,mu2,σ²) → 6 vals    EdgeDiffMap(img1,mu1,img2,mu2) → 12 vals
              │                                           │
              └──────────────────┬────────────────────────┘
                                 ▼
                     108 sub-scores read back to CPU
                                 │
                        ┌────────▼────────┐
                        │ Weighted sum    │
                        │ → final 0–100   │
                        └─────────────────┘
```

## WebGPU shaders

| Shader              | Purpose                                   | Dispatch              |
|---------------------|-------------------------------------------|-----------------------|
| `gamma_expand`      | sRGB → linear per pixel                   | 1 thread / pixel      |
| `linear_to_xyb`     | Linear RGB → XYB + rescale                | 1 thread / pixel      |
| `downsample_2x`     | Box-filter 2× downscale                   | 1 thread / output px  |
| `gauss_blur_h`      | IIR horizontal pass                       | 1 thread / row        |
| `gauss_blur_v`      | IIR vertical pass                         | 1 thread / column     |
| `multiply`          | Element-wise plane multiply               | 1 thread / pixel      |
| `ssim_reduce`       | SSIM map + reduce to 6 norm values        | workgroup reduction   |
| `edgediff_reduce`   | Edge diff map + reduce to 12 norm values  | workgroup reduction   |

## Key design decisions

- **Planar f32 buffers** (3 planes per image) — matches C++ algorithm structure
- **Buffer reuse between scales** — avoids allocating for all 6 scales simultaneously
- **CPU scoring** — the weighted sum is trivial scalar math, not worth a GPU roundtrip
- **IIR blur = 1 thread per row/column** — each thread serially scans its row/col
- **Adaptive sizing** — buffers sized per-scale, not for max resolution

## Implementation order

1. Scaffold: `package.json`, `vite.config.js`, `.gitignore`, `index.html`
2. `pipeline.js` — adapter/device init, factory helpers
3. `shaders.js` — all 8 WGSL shaders
4. `ssimulacra2.js` — main pipeline orchestration
5. `scoring.js` — weighted sum
6. `demo/main.js` — image picker UI, rendering, score display

## Risks

- **GPU memory at 4K**: ~30 f32 planes × 33 MB ≈ 1 GB. Mitigate via buffer reuse; consider f16 if needed.
- **IIR blur occupancy**: 1 thread/workgroup is low occupancy, but work is memory-bound sequential scan — acceptable.
- **Browser WebGPU**: Chrome/Edge 113+, Firefox Nightly, Safari TP. Demo must check support with a clear error message.
