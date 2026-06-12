# Performance Optimization Plan

Current score: 70.8 for Lena vs Lena JPEG (matches C++ reference within 0.1 %).
Target: **2–4× speedup** while preserving identical numerical results.

---

## Profiling results

Processing 4032 × 3024 images (`original/20240528_225613.jpg`) takes **~60 seconds**.
Main bottleneck: GPU dispatches with tiny workgroups.

---

## Root causes

### 1. IIR Gaussian blur dispatches are 1×1

`gauss_h` and `gauss_v` use `@workgroup_size(1, 1)`. Each invocation
processes one row (h‑pass) or one column (v‑pass) sequentially — every
pixel is handled by one thread. A 4032‑wide row invokes 4032 individual
workgroups containing one thread each.

**Impact:** the 6 blur passes per scale dispatch hundreds of thousands of
tiny workgroups, paying full dispatch overhead every time.

### 2. Three separate dispatches per component

The mulChain pattern issues 3 sequential dispatches (one per X/Y/B
component) for each sub‑step (multiply, h‑blur, v‑blur). These are
independent and could share a single dispatch.

### 3. Multiple `encoder.submit()` calls

The pipeline calls `d.queue.submit()` once per scale (6× total). Each
submit adds latency. The entire pipeline could be submitted in one
`CommandEncoder` batch.

### 4. Reduction passes are single‑workgroup

`ssim_reduce` and `edgediff_reduce` dispatch `(1, 1)` after the
workgroup‑level pass has already done most of the reduction. This is a
serial CPU‑style step that cannot be parallelised — acceptable as‑is.

---

## Proposed optimisations

### 1. Vectorise the IIR blur (high impact, ~3× faster)

Change `gauss_h` and `gauss_v` workgroups so that multiple rows/columns
are processed per workgroup, each by a separate thread.

| Shader | Current | Optimised |
|--------|---------|-----------|
| `gauss_h` | `@workgroup_size(1, 1)` → `1 × height` | `@workgroup_size(64, 1)` → `ceilDiv(width, 64) × height` |
| `gauss_v` | `@workgroup_size(1, 1)` → `width × 1` | `@workgroup_size(1, 64)` → `width × ceilDiv(height, 64)` |

Each thread processes one row/column; `global_invocation_id.x` steps by
`workgroup_size.x` within the row/column instead of covering the whole
row.

**Numerical impact:** None — the per‑thread algorithm is identical, only
the parallelisation changes.

### 2. Batch the three component dispatches (medium impact)

Replace:

```
pass("s1-mul", [
  ["multiply",  X-mul],
  ["multiply",  Y-mul],
  ["multiply",  B-mul],
]);
pass("s1-h", [
  ["gauss_h",  X-h],
  ["gauss_h",  Y-h],
  ["gauss_h",  B-h],
]);
pass("s1-v", [
  ["gauss_v",  X-v],
  ["gauss_v",  Y-v],
  ["gauss_v",  B-v],
]);
```

With a single combined dispatch that processes all 3 components in one
pass. Since each dispatch writes to different output buffers, the
sync‑scope constraint is satisfied.

This applies to `mulChain("s1")`, `mulChain("s2")`, `mulChain("s3")`,
`blur1`, and `blur2`.

**Numerical impact:** None — the same shader is invoked the same number
of times, just grouped into fewer passes.

### 3. Single `submit()` (low impact, but easy)

Move the `d.queue.submit()` call outside the scale loop. Use one
`CommandEncoder` for all scales and submit once at the end, keeping
only the final readback as a separate submit.

**Numerical impact:** None.

### 4. Increase workgroup size for element‑wise passes (trivial)

`to_linear`, `to_xyb`, `multiply`, `downsample` use `@workgroup_size(16,
16)`. Increase to `@workgroup_size(32, 32)` to reduce dispatch overhead.

**Numerical impact:** None — these are flat‑parallel per‑pixel kernels.

---

## Implementation results

### Done
- **Vectorise IIR blur** — `gauss_h`/`gauss_v` changed from `@workgroup_size(1, 1)`
  to `@workgroup_size(256)`. Rows (h‑pass) / columns (v‑pass) are now indexed via
  `id.x` instead of `id.y`/`id.x`. Dispatch changed to `ceilDiv(N, 256) × 1`.
  Numerical results verified identical.
- **Single submit** — one `CommandEncoder` for all scales, submitted once after
  the loop (plus the readback submit). No API violations.

### Skipped as already optimal
- **Batch component dispatches** — the X/Y/B dispatches for blur, multiply, and
  reduction passes were already grouped into single compute passes (`pass()`
  batches all 3 components). No further batching possible without introducing
  buffer sync‑scope violations.
- **Increase element‑wise workgroups** — `@workgroup_size(16, 16)` = 256
  invocations, which is the mandatory `maxComputeWorkgroupInvocations` limit.
  32×32 would exceed the spec minimum of 256.

## Future work (if more speed is needed)

- **Bind group caching** — `_bg()` creates a new `GPUBindGroup` for *every*
  dispatch. Caching by `(pipeline, entries)` hash would reduce CPU-side
  creation overhead.
- **Shared-memory blur** — the IIR blur is inherently serial per row/column,
  but small images (scales 4‑5) could be processed in fewer workgroups by
  fusing the h‑pass and v‑pass into a single kernel.
- **f16 arithmetic** — replacing `f32` with `f16` would halve memory bandwidth
  for buffer reads/writes. Requires WebGPU `shader-f16` feature.
