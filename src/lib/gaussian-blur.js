function inv3x3(m) {
  const [a, b, c] = m[0];
  const [d, e, f] = m[1];
  const [g, h, i] = m[2];
  const det =
    a * (e * i - f * h) -
    b * (d * i - f * g) +
    c * (d * h - e * g);
  const id = 1.0 / det;
  return [
    [(e * i - f * h) * id, (c * h - b * i) * id, (b * f - c * e) * id],
    [(f * g - d * i) * id, (a * i - c * g) * id, (c * d - a * f) * id],
    [(d * h - e * g) * id, (b * g - a * h) * id, (a * e - b * d) * id],
  ];
}

function matMul3x3Vec(m, v) {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

// Implements "Recursive Implementation of the Gaussian Filter Using
// Truncated Cosine Functions" by Charalampidis [2016].
// Exact translation of JPEG XL's CreateRecursiveGaussian.
export function createGaussianCoeffs(sigma) {
  const kPi = Math.PI;
  const radius = Math.round(3.2795 * sigma + 0.2546);
  const pi_div_2r = kPi / (2.0 * radius);

  const omega = [pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r];
  const p = omega.map((w, i) => ((i & 1) ? -1 : 1) / Math.tan(0.5 * w));
  const r = omega.map((w, i) => ((i & 1) ? -1 : 1) * p[i] * p[i] / Math.sin(w));

  const nhs2 = -0.5 * sigma * sigma;
  const rho = omega.map((w) => Math.exp(nhs2 * w * w) / radius);

  const D_13 = p[0] * r[1] - r[0] * p[1];
  const D_35 = p[1] * r[2] - r[1] * p[2];
  const D_51 = p[2] * r[0] - r[2] * p[0];
  const rd13 = 1.0 / D_13;
  const z15 = D_35 * rd13;
  const z35 = D_51 * rd13;

  const A = [
    [p[0], p[1], p[2]],
    [r[0], r[1], r[2]],
    [z15, z35, 1.0],
  ];
  const gamma = [
    1.0,
    radius * radius - sigma * sigma,
    z15 * rho[0] + z35 * rho[1] + rho[2],
  ];
  const beta = matMul3x3Vec(inv3x3(A), gamma);

  const n2 = [];
  const d1 = [];
  for (let i = 0; i < 3; i++) {
    n2[i] = -beta[i] * Math.cos(omega[i] * (radius + 1.0));
    d1[i] = -2.0 * Math.cos(omega[i]);
  }

  return { radius, n2, d1 };
}
