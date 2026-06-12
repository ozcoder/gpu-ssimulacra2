const PARAMS = `
struct Params {
  width: u32,
  height: u32,
  radius: u32,
  component: u32,
  n2_1: f32,
  n2_3: f32,
  n2_5: f32,
  d1_1: f32,
  d1_3: f32,
  d1_5: f32,
  num_wg_x: u32,
  num_wg_y: u32,
  output_offset: u32,
};
`;

// Opsin absorbance matrix + bias — matches standalone ssimulacra2 / libjxl
// (kOpsinAbsorbanceMatrix, kOpsinAbsorbanceBias)
const OPSIN_MATRIX = `
const OPSIN_M0 = vec3<f32>(0.30, 0.622, 0.078);
const OPSIN_M1 = vec3<f32>(0.23, 0.692, 0.078);
const OPSIN_M2 = vec3<f32>(0.24342268924547819, 0.20476744424496821, 0.5518098665095536);
const OPSIN_BIAS = vec3<f32>(0.0037930732552754493, 0.0037930732552754493, 0.0037930732552754493);
`;

// Common WGSL utilities
const UTIL = `
fn srgb_to_linear(c: f32) -> f32 {
  let a = c / 12.92;
  let b = pow((c + 0.055) / 1.055, 2.4);
  return select(b, a, c <= 0.04045);
}

fn quartic(x: f32) -> f32 {
  let x2 = x * x;
  return x2 * x2;
}
`;

export const WGSL = {};

// ── sRGB RGBA u8 → 3 × f32 linear RGB ──────────────────────────────
WGSL.to_linear = `${PARAMS}
@group(0) @binding(0) var<storage, read> rgba: array<u32>;
@group(0) @binding(1) var<storage, read_write> plane_r: array<f32>;
@group(0) @binding(2) var<storage, read_write> plane_g: array<f32>;
@group(0) @binding(3) var<storage, read_write> plane_b: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;

${UTIL}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.width || id.y >= p.height) { return; }
  let idx = id.y * p.width + id.x;
  let px = rgba[idx];
  let r = srgb_to_linear(f32(px & 0xffu) / 255.0);
  let g = srgb_to_linear(f32((px >> 8u) & 0xffu) / 255.0);
  let b = srgb_to_linear(f32((px >> 16u) & 0xffu) / 255.0);
  plane_r[idx] = r;
  plane_g[idx] = g;
  plane_b[idx] = b;
}
`;

// ── 3 × f32 linear RGB → 3 × f32 XYB (with MakePositiveXYB) ────────
WGSL.to_xyb = `${PARAMS}
@group(0) @binding(0) var<storage, read> plane_r: array<f32>;
@group(0) @binding(1) var<storage, read> plane_g: array<f32>;
@group(0) @binding(2) var<storage, read> plane_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_x: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> out_z: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

${OPSIN_MATRIX}

fn linear_to_xyb(r: f32, g: f32, b: f32) -> vec3<f32> {
  let m0 = max(dot(OPSIN_M0, vec3<f32>(r, g, b)) + OPSIN_BIAS.x, 0.0);
  let m1 = max(dot(OPSIN_M1, vec3<f32>(r, g, b)) + OPSIN_BIAS.y, 0.0);
  let m2 = max(dot(OPSIN_M2, vec3<f32>(r, g, b)) + OPSIN_BIAS.z, 0.0);
  let l = pow(m0, 1.0 / 3.0) - pow(OPSIN_BIAS.x, 1.0 / 3.0);
  let m = pow(m1, 1.0 / 3.0) - pow(OPSIN_BIAS.y, 1.0 / 3.0);
  let s = pow(m2, 1.0 / 3.0) - pow(OPSIN_BIAS.z, 1.0 / 3.0);
  return vec3<f32>(0.5 * (l - m), 0.5 * (l + m), s);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.width || id.y >= p.height) { return; }
  let idx = id.y * p.width + id.x;
  let rgb = vec3<f32>(plane_r[idx], plane_g[idx], plane_b[idx]);
  var xyb = linear_to_xyb(rgb.r, rgb.g, rgb.b);
  // MakePositiveXYB — B uses old Y, then Y is incremented
  // (matching C++ reference order)
  xyb.z = xyb.z - xyb.y + 0.55;
  xyb.x = xyb.x * 14.0 + 0.42;
  xyb.y = xyb.y + 0.01;
  out_x[idx] = xyb.x;
  out_y[idx] = xyb.y;
  out_z[idx] = xyb.z;
}
`;

// ── Box-filter 2× downsample ────────────────────────────────────────
WGSL.downsample = `${PARAMS}
@group(0) @binding(0) var<storage, read> in0: array<f32>;
@group(0) @binding(1) var<storage, read> in1: array<f32>;
@group(0) @binding(2) var<storage, read> in2: array<f32>;
@group(0) @binding(3) var<storage, read_write> out0: array<f32>;
@group(0) @binding(4) var<storage, read_write> out1: array<f32>;
@group(0) @binding(5) var<storage, read_write> out2: array<f32>;
@group(0) @binding(6) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let ow = (p.width + 1u) / 2u;
  let oh = (p.height + 1u) / 2u;
  if (id.x >= ow || id.y >= oh) { return; }

  let in_w = p.width;
  let in_h = p.height;
  var s0 = 0.0; var s1 = 0.0; var s2 = 0.0;
  var count = 0u;
  for (var dy = 0u; dy < 2u; dy++) {
    for (var dx = 0u; dx < 2u; dx++) {
      let sx = min(id.x * 2u + dx, in_w - 1u);
      let sy = min(id.y * 2u + dy, in_h - 1u);
      let si = sy * in_w + sx;
      s0 += in0[si];
      s1 += in1[si];
      s2 += in2[si];
      count++;
    }
  }
  let oi = id.y * ow + id.x;
  let norm = 1.0 / f32(count);
  out0[oi] = s0 * norm;
  out1[oi] = s1 * norm;
  out2[oi] = s2 * norm;
}
`;

// ── 1D IIR Gaussian blur — horizontal pass ──────────────────────────
// Charalampidis [2016] two-tap symmetric input: reads input[x-N-1] +
// input[x+N-1] at each step. Single forward pass (no backward pass).
// Includes warmup from -N+1 to -1 to match C++ reference initialization.
WGSL.gauss_h = `${PARAMS}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = i32(id.x);
  if (row >= i32(p.height)) { return; }

  var p1 = 0.0; var p2_1 = 0.0;
  var p3 = 0.0; var p2_3 = 0.0;
  var p5 = 0.0; var p2_5 = 0.0;

  let W = i32(p.width);
  let N = i32(p.radius);
  let base = row * W;

  for (var n = -(N - 1); n < 0; n++) {
    let left = n - N - 1;
    let right = n + N - 1;

    var lv = 0.0;
    var rv = 0.0;
    if (left >= 0 && left < W) { lv = input[base + left]; }
    if (right >= 0 && right < W) { rv = input[base + right]; }

    let sum = lv + rv;
    let o1 = p.n2_1 * sum - p.d1_1 * p1 - p2_1;
    let o3 = p.n2_3 * sum - p.d1_3 * p3 - p2_3;
    let o5 = p.n2_5 * sum - p.d1_5 * p5 - p2_5;

    p2_1 = p1;   p1 = o1;
    p2_3 = p3;   p3 = o3;
    p2_5 = p5;   p5 = o5;
  }

  for (var x = 0; x < W; x++) {
    let left = x - N - 1;
    let right = x + N - 1;

    var lv = 0.0;
    var rv = 0.0;
    if (left >= 0 && left < W) { lv = input[base + left]; }
    if (right >= 0 && right < W) { rv = input[base + right]; }

    let sum = lv + rv;
    let o1 = p.n2_1 * sum - p.d1_1 * p1 - p2_1;
    let o3 = p.n2_3 * sum - p.d1_3 * p3 - p2_3;
    let o5 = p.n2_5 * sum - p.d1_5 * p5 - p2_5;

    p2_1 = p1;   p1 = o1;
    p2_3 = p3;   p3 = o3;
    p2_5 = p5;   p5 = o5;

    output[base + x] = o1 + o3 + o5;
  }
}
`;

// ── 1D IIR Gaussian blur — vertical pass ────────────────────────────
// Charalampidis [2016] two-tap symmetric input: reads input[y-N-1] +
// input[y+N-1] at each step. Single forward pass.
// Includes warmup from -N+1 to -1 to match C++ reference initialization.
WGSL.gauss_v = `${PARAMS}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let col = i32(id.x);
  if (col >= i32(p.width)) { return; }

  var p1 = 0.0; var p2_1 = 0.0;
  var p3 = 0.0; var p2_3 = 0.0;
  var p5 = 0.0; var p2_5 = 0.0;

  let W = i32(p.width);
  let H = i32(p.height);
  let N = i32(p.radius);

  for (var n = -(N - 1); n < 0; n++) {
    let top = n - N - 1;
    let bot = n + N - 1;

    var tv = 0.0;
    var bv = 0.0;
    if (top >= 0 && top < H) { tv = input[top * W + col]; }
    if (bot >= 0 && bot < H) { bv = input[bot * W + col]; }

    let sum = tv + bv;
    let o1 = p.n2_1 * sum - p.d1_1 * p1 - p2_1;
    let o3 = p.n2_3 * sum - p.d1_3 * p3 - p2_3;
    let o5 = p.n2_5 * sum - p.d1_5 * p5 - p2_5;

    p2_1 = p1;   p1 = o1;
    p2_3 = p3;   p3 = o3;
    p2_5 = p5;   p5 = o5;
  }

  for (var y = 0; y < H; y++) {
    let top = y - N - 1;
    let bot = y + N - 1;

    var tv = 0.0;
    var bv = 0.0;
    if (top >= 0 && top < H) { tv = input[top * W + col]; }
    if (bot >= 0 && bot < H) { bv = input[bot * W + col]; }

    let sum = tv + bv;
    let o1 = p.n2_1 * sum - p.d1_1 * p1 - p2_1;
    let o3 = p.n2_3 * sum - p.d1_3 * p3 - p2_3;
    let o5 = p.n2_5 * sum - p.d1_5 * p5 - p2_5;

    p2_1 = p1;   p1 = o1;
    p2_3 = p3;   p3 = o3;
    p2_5 = p5;   p5 = o5;

    output[y * W + col] = o1 + o3 + o5;
  }
}
`;

// ── Element‑wise plane multiply ──────────────────────────────────────
WGSL.multiply = `${PARAMS}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.width || id.y >= p.height) { return; }
  let idx = id.y * p.width + id.x;
  out[idx] = a[idx] * b[idx];
}
`;

// ── Per‑workgroup SSIM reduction (one component) ────────────────────
WGSL.ssim_wg = `${PARAMS}
@group(0) @binding(0) var<storage, read> mu1: array<f32>;
@group(0) @binding(1) var<storage, read> mu2: array<f32>;
@group(0) @binding(2) var<storage, read> s11: array<f32>;
@group(0) @binding(3) var<storage, read> s22: array<f32>;
@group(0) @binding(4) var<storage, read> s12: array<f32>;
@group(0) @binding(5) var<storage, read_write> wg_out: array<vec2<f32>>;
@group(0) @binding(6) var<uniform> p: Params;

${UTIL}
const kC2 = 0.0009;

var<workgroup> shd: array<f32, 256>;
var<workgroup> shq: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let idx = lid.y * 16u + lid.x;
  if (gid.x < p.width && gid.y < p.height) {
    let i = gid.y * p.width + gid.x;
    let m1 = mu1[i];
    let m2 = mu2[i];
    let m11 = m1 * m1;
    let m22 = m2 * m2;
    let m12 = m1 * m2;

    let nm = 1.0 - (m1 - m2) * (m1 - m2);
    let ns = 2.0 * (s12[i] - m12) + kC2;
    let ds = (s11[i] - m11) + (s22[i] - m22) + kC2;

    let d = max(1.0 - nm * ns / ds, 0.0);
    shd[idx] = d;
    shq[idx] = quartic(d);
  } else {
    shd[idx] = 0.0;
    shq[idx] = 0.0;
  }
  workgroupBarrier();

  for (var s = 128u; s > 0u; s >>= 1u) {
    if (idx < s) {
      shd[idx] += shd[idx + s];
      shq[idx] += shq[idx + s];
    }
    workgroupBarrier();
  }

  if (idx == 0u) {
    let total_wg = p.num_wg_x * p.num_wg_y;
    let wg_idx = wgid.y * p.num_wg_x + wgid.x;
    wg_out[p.component * total_wg + wg_idx] = vec2<f32>(shd[0], shq[0]);
  }
}
`;

// ── Global SSIM reduction (one component) ────────────────────────────
WGSL.ssim_reduce = `${PARAMS}
@group(0) @binding(0) var<storage, read> wg_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> shd: array<f32, 256>;
var<workgroup> shq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let num_wg = p.num_wg_x * p.num_wg_y;
  let base = p.component * num_wg;
  var sd = 0.0;
  var sq = 0.0;
  for (var i = lid.x; i < num_wg; i += 256u) {
    sd += wg_in[base + i].x;
    sq += wg_in[base + i].y;
  }
  shd[lid.x] = sd;
  shq[lid.x] = sq;
  workgroupBarrier();

  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      shd[lid.x] += shd[lid.x + s];
      shq[lid.x] += shq[lid.x + s];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    let num_pixels = f32(p.width * p.height);
    let off = p.output_offset + p.component * 2u;
    output[off]     = shd[0] / num_pixels;
    output[off + 1] = sqrt(sqrt(shq[0] / num_pixels));
  }
};
`;

// ── Per‑workgroup EdgeDiff reduction (one component) ─────────────────
WGSL.edgediff_wg = `${PARAMS}
@group(0) @binding(0) var<storage, read> img: array<f32>;
@group(0) @binding(1) var<storage, read> mu: array<f32>;
@group(0) @binding(2) var<storage, read> img2: array<f32>;
@group(0) @binding(3) var<storage, read> mu2: array<f32>;
@group(0) @binding(4) var<storage, read_write> wg_out: array<vec4<f32>>;
@group(0) @binding(5) var<uniform> p: Params;

${UTIL}

var<workgroup> sh_a: array<f32, 256>;
var<workgroup> sh_aq: array<f32, 256>;
var<workgroup> sh_d: array<f32, 256>;
var<workgroup> sh_dq: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let idx = lid.y * 16u + lid.x;
  if (gid.x < p.width && gid.y < p.height) {
    let i = gid.y * p.width + gid.x;
    let d1 = (1.0 + abs(img2[i] - mu2[i])) /
             (1.0 + abs(img[i] - mu[i])) - 1.0;
    let artifact = max(d1, 0.0);
    let detail_lost = max(-d1, 0.0);
    sh_a[idx]  = artifact;
    sh_aq[idx] = quartic(artifact);
    sh_d[idx]  = detail_lost;
    sh_dq[idx] = quartic(detail_lost);
  } else {
    sh_a[idx]  = 0.0;
    sh_aq[idx] = 0.0;
    sh_d[idx]  = 0.0;
    sh_dq[idx] = 0.0;
  }
  workgroupBarrier();

  for (var s = 128u; s > 0u; s >>= 1u) {
    if (idx < s) {
      sh_a[idx]  += sh_a[idx + s];
      sh_aq[idx] += sh_aq[idx + s];
      sh_d[idx]  += sh_d[idx + s];
      sh_dq[idx] += sh_dq[idx + s];
    }
    workgroupBarrier();
  }

  if (idx == 0u) {
    let total_wg = p.num_wg_x * p.num_wg_y;
    let wg_idx = wgid.y * p.num_wg_x + wgid.x;
    wg_out[p.component * total_wg + wg_idx] = vec4<f32>(sh_a[0], sh_aq[0], sh_d[0], sh_dq[0]);
  }
}
`;

// ── Global EdgeDiff reduction (one component) ─────────────────────────
WGSL.edgediff_reduce = `${PARAMS}
@group(0) @binding(0) var<storage, read> wg_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

var<workgroup> sha: array<f32, 256>;
var<workgroup> shaq: array<f32, 256>;
var<workgroup> shd: array<f32, 256>;
var<workgroup> shdq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let num_wg = p.num_wg_x * p.num_wg_y;
  let base = p.component * num_wg;
  var sa = 0.0; var saq = 0.0; var sd = 0.0; var sdq = 0.0;
  for (var i = lid.x; i < num_wg; i += 256u) {
    let v = wg_in[base + i];
    sa  += v[0];
    saq += v[1];
    sd  += v[2];
    sdq += v[3];
  }
  sha[lid.x]  = sa;
  shaq[lid.x] = saq;
  shd[lid.x]  = sd;
  shdq[lid.x] = sdq;
  workgroupBarrier();

  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      sha[lid.x]  += sha[lid.x + s];
      shaq[lid.x] += shaq[lid.x + s];
      shd[lid.x]  += shd[lid.x + s];
      shdq[lid.x] += shdq[lid.x + s];
    }
    workgroupBarrier();
  }

  if (lid.x == 0u) {
    let num_pixels = f32(p.width * p.height);
    let off = p.output_offset + 6u + p.component * 4u;
    output[off]     = sha[0]  / num_pixels;
    output[off + 1] = sqrt(sqrt(shaq[0] / num_pixels));
    output[off + 2] = shd[0]  / num_pixels;
    output[off + 3] = sqrt(sqrt(shdq[0] / num_pixels));
  }
}
`;
