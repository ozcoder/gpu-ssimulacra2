import { createGaussianCoeffs } from "./gaussian-blur.js";
import { ceilDiv } from "./pipeline.js";
import { WGSL } from "./shaders.js";

const MAX_SCALES = 6;
const OUT_PER_SCALE = 18;

export class SSIMULACRA2Pipeline {
  constructor(device) {
    this.device = device;
    this.ready = false;
  }

  async allocate(maxWidth, maxHeight) {
    this.maxW = maxWidth;
    this.maxH = maxHeight;
    this.coeffs = createGaussianCoeffs(1.5);
    this._createPipelines(this.device);
    this._createBuffers(this.device, maxWidth, maxHeight);
    this.ready = true;
  }

  _createPipelines(d) {
    this.pl = {};
    for (const n of [
      "to_linear", "to_xyb", "downsample",
      "gauss_h", "gauss_v", "multiply",
      "ssim_wg", "ssim_reduce",
      "edgediff_wg", "edgediff_reduce",
    ]) {
      this.pl[n] = d.createComputePipeline({
        layout: "auto",
        compute: { module: d.createShaderModule({ code: WGSL[n] }), entryPoint: "main" },
      });
    }
  }

  _createBuffers(d, maxW, maxH) {
    const ps = maxW * maxH * 4;
    const st = (l) =>
      d.createBuffer({
        label: l,
        size: ps,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });

    const inUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.buf = {
      origIn: d.createBuffer({ label: "origIn", size: ps, usage: inUsage }),
      distIn: d.createBuffer({ label: "distIn", size: ps, usage: inUsage }),

      loR: st("loR"), loG: st("loG"), loB: st("loB"),
      ldR: st("ldR"), ldG: st("ldG"), ldB: st("ldB"),
      or: st("or"), og: st("og"), ob: st("ob"),
      dr: st("dr"), dg: st("dg"), db: st("db"),

      i1x: st("i1x"), i1y: st("i1y"), i1b: st("i1b"),
      i2x: st("i2x"), i2y: st("i2y"), i2b: st("i2b"),

      m1x: st("m1x"), m1y: st("m1y"), m1b: st("m1b"),
      m2x: st("m2x"), m2y: st("m2y"), m2b: st("m2b"),

      sx: st("sx"), sy: st("sy"), sb: st("sb"),
      mx: st("mx"), my: st("my"), mb: st("mb"),
      tx: st("tx"), ty: st("ty"), tb: st("tb"),
      c12x: st("c12x"), c12y: st("c12y"), c12b: st("c12b"),
      s2x: st("s2x"), s2y: st("s2y"), s2b: st("s2b"),
    };

    const maxWg = ceilDiv(maxW, 16) * ceilDiv(maxH, 16);
    this.buf.ssimWg = d.createBuffer({
      label: "ssimWg",
      size: maxWg * 8 * 3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.buf.edWg = d.createBuffer({
      label: "edWg",
      size: maxWg * 16 * 3,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.buf.final = d.createBuffer({
      label: "final",
      size: Math.max(MAX_SCALES * OUT_PER_SCALE * 4, 256),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.buf.staging = d.createBuffer({
      label: "staging",
      size: Math.max(MAX_SCALES * OUT_PER_SCALE * 4, 256),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    this.paramBufs = [];
    for (let i = 0; i < MAX_SCALES * 14; i++) {
      this.paramBufs.push(d.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
    }
  }

  _ser(p) {
    const dv = new DataView(new ArrayBuffer(64));
    const W = (o, v) => dv.setUint32(o, v >>> 0, true);
    const F = (o, v) => dv.setFloat32(o, v, true);
    W(0, p.width); W(4, p.height); W(8, p.radius); W(12, p.component);
    F(16, p.n2_1); F(20, p.n2_3); F(24, p.n2_5);
    F(28, p.d1_1); F(32, p.d1_3); F(36, p.d1_5);
    W(40, p.num_wg_x); W(44, p.num_wg_y); W(48, p.output_offset);
    return dv.buffer;
  }

  _bg(pipeline, entries) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((e) => ({ binding: e[0], resource: e[1] })),
    });
  }

  _b(name) {
    return { buffer: this.buf[name] };
  }

  async compute(origData, distData) {
    if (origData.width !== distData.width || origData.height !== distData.height) {
      throw new Error("Images must be the same size");
    }
    const W = origData.width;
    const H = origData.height;
    if (W > this.maxW || H > this.maxH) {
      throw new Error(`Image ${W}x${H} exceeds max ${this.maxW}x${this.maxH}`);
    }

    const d = this.device;
    const b = this.buf;

    d.queue.writeBuffer(b.origIn, 0, origData.data);
    d.queue.writeBuffer(b.distIn, 0, distData.data);

    const scales = [];
    for (let sw = W, sh = H, i = 0; i < MAX_SCALES && sw >= 8 && sh >= 8; i++) {
      scales.push({ w: sw, h: sh, off: i * OUT_PER_SCALE });
      sw = Math.ceil(sw / 2);
      sh = Math.ceil(sh / 2);
    }

    const coeff = this.coeffs;
    const base = (w, h) => ({
      width: w, height: h, radius: coeff.radius,
      component: 0, n2_1: coeff.n2[0], n2_3: coeff.n2[1], n2_5: coeff.n2[2],
      d1_1: coeff.d1[0], d1_3: coeff.d1[1], d1_5: coeff.d1[2],
      num_wg_x: 0, num_wg_y: 0, output_offset: 0,
    });

    for (let si = 0; si < scales.length; si++) {
      const { w, h } = scales[si];
      const nwgX = ceilDiv(w, 16);
      const nwgY = ceilDiv(h, 16);
      const b0 = base(w, h);
      const writeP = (pi, extra) => {
        d.queue.writeBuffer(this.paramBufs[si * 14 + pi], 0,
          this._ser({ ...b0, ...extra }));
      };
      writeP(0, { num_wg_x: nwgX, num_wg_y: nwgY });
      for (let c = 0; c < 3; c++) {
        writeP(1 + c, { component: c, num_wg_x: nwgX, num_wg_y: nwgY });
        writeP(4 + c, { component: c, num_wg_x: nwgX, num_wg_y: nwgY, output_offset: scales[si].off });
        writeP(7 + c, { component: c, num_wg_x: nwgX, num_wg_y: nwgY });
        writeP(10 + c, { component: c, num_wg_x: nwgX, num_wg_y: nwgY, output_offset: scales[si].off });
      }
    }

    for (let si = 0; si < scales.length; si++) {
      const { w, h } = scales[si];
      const nwgX = ceilDiv(w, 16);
      const nwgY = ceilDiv(h, 16);
      const pb = (pi) => ({ buffer: this.paramBufs[si * 14 + pi] });

      const linO = [
        [this._b("loR"), this._b("loG"), this._b("loB")],
        [this._b("or"), this._b("og"), this._b("ob")],
      ];
      const linD = [
        [this._b("ldR"), this._b("ldG"), this._b("ldB")],
        [this._b("dr"), this._b("dg"), this._b("db")],
      ];
      const curO = linO[si & 1];
      const curD = linD[si & 1];
      const prevO = linO[(si + 1) & 1];
      const prevD = linD[(si + 1) & 1];

      const pl = (p) => [this._b(p + "x"), this._b(p + "y"), this._b(p + "b")];
      const i1 = pl("i1");
      const i2 = pl("i2");
      const mu1 = pl("m1");
      const mu2 = pl("m2");
      const sg1 = pl("s");
      const sg2 = [this._b("s2x"), this._b("s2y"), this._b("s2b")];
      const sg12 = [this._b("c12x"), this._b("c12y"), this._b("c12b")];

      const encoder = d.createCommandEncoder();

      // Helper: run one or more dispatches in a single compute pass.
      // Each pass creates a separate synchronization scope, avoiding
      // buffer read-write conflicts between dispatches.
      const pass = (label, jobs) => {
        const p = encoder.beginComputePass({ label });
        for (const [name, entries, dx, dy] of jobs) {
          p.setPipeline(this.pl[name]);
          p.setBindGroup(0, this._bg(this.pl[name], entries));
          p.dispatchWorkgroups(dx, dy);
        }
        p.end();
      };

      // Helper: build an entry list for a gauss_h dispatch reading `src` and
      // writing one of the t* temp planes.
      const hBlur = (src, ci) => [
        [0, src],
        [1, this._b("t" + "xyb"[ci])],
        [2, pb(0)],
      ];

      // Helper: build an entry list for a gauss_v dispatch reading the t*
      // temp plane and writing to `dst`.
      const vBlur = (ci, dst) => [
        [0, this._b("t" + "xyb"[ci])],
        [1, dst],
        [2, pb(0)],
      ];

      // to_linear/downsample and to_xyb must be in separate passes because
      // the linear RGB buffers (curO/curD) are read_write in the first and
      // read-only in the second.
      if (si === 0) {
        pass("to_linear", [
          ["to_linear", [
            [0, this._b("origIn")], [1, curO[0]], [2, curO[1]], [3, curO[2]],
            [4, pb(0)],
          ], nwgX, nwgY],
          ["to_linear", [
            [0, this._b("distIn")], [1, curD[0]], [2, curD[1]], [3, curD[2]],
            [4, pb(0)],
          ], nwgX, nwgY],
        ]);
      } else {
        const prevPb = { buffer: this.paramBufs[(si - 1) * 14] };
        pass("downsample", [
          ["downsample", [
            [0, prevO[0]], [1, prevO[1]], [2, prevO[2]],
            [3, curO[0]], [4, curO[1]], [5, curO[2]],
            [6, prevPb],
          ], nwgX, nwgY],
          ["downsample", [
            [0, prevD[0]], [1, prevD[1]], [2, prevD[2]],
            [3, curD[0]], [4, curD[1]], [5, curD[2]],
            [6, prevPb],
          ], nwgX, nwgY],
        ]);
      }
      pass("to_xyb", [
        ["to_xyb", [
          [0, curO[0]], [1, curO[1]], [2, curO[2]],
          [3, i1[0]], [4, i1[1]], [5, i1[2]],
          [6, pb(0)],
        ], nwgX, nwgY],
        ["to_xyb", [
          [0, curD[0]], [1, curD[1]], [2, curD[2]],
          [3, i2[0]], [4, i2[1]], [5, i2[2]],
          [6, pb(0)],
        ], nwgX, nwgY],
      ]);

      // Each separable blur (gauss_h → t* → gauss_v) must occupy its own
      // pair of passes because t* is read-write in gauss_h and read-only
      // in gauss_v.

      // blur3(i1) → mu1
      pass("blur1-h", [
        ["gauss_h", hBlur(i1[0], 0), 1, h],
        ["gauss_h", hBlur(i1[1], 1), 1, h],
        ["gauss_h", hBlur(i1[2], 2), 1, h],
      ]);
      pass("blur1-v", [
        ["gauss_v", vBlur(0, mu1[0]), w, 1],
        ["gauss_v", vBlur(1, mu1[1]), w, 1],
        ["gauss_v", vBlur(2, mu1[2]), w, 1],
      ]);

      // blur3(i2) → mu2
      pass("blur2-h", [
        ["gauss_h", hBlur(i2[0], 0), 1, h],
        ["gauss_h", hBlur(i2[1], 1), 1, h],
        ["gauss_h", hBlur(i2[2], 2), 1, h],
      ]);
      pass("blur2-v", [
        ["gauss_v", vBlur(0, mu2[0]), w, 1],
        ["gauss_v", vBlur(1, mu2[1]), w, 1],
        ["gauss_v", vBlur(2, mu2[2]), w, 1],
      ]);

      // Each mulBlur chain: multiply → m*, gauss_h(m*) → t*, gauss_v(t*) → dst
      const mulChain = (label, a, b, dst) => {
        pass(label + "-mul", [
          ["multiply", [[0, a[0]], [1, b[0]], [2, this._b("mx")], [3, pb(0)]], nwgX, nwgY],
          ["multiply", [[0, a[1]], [1, b[1]], [2, this._b("my")], [3, pb(0)]], nwgX, nwgY],
          ["multiply", [[0, a[2]], [1, b[2]], [2, this._b("mb")], [3, pb(0)]], nwgX, nwgY],
        ]);
        pass(label + "-h", [
          ["gauss_h", [[0, this._b("mx")], [1, this._b("tx")], [2, pb(0)]], 1, h],
          ["gauss_h", [[0, this._b("my")], [1, this._b("ty")], [2, pb(0)]], 1, h],
          ["gauss_h", [[0, this._b("mb")], [1, this._b("tb")], [2, pb(0)]], 1, h],
        ]);
        pass(label + "-v", [
          ["gauss_v", [[0, this._b("tx")], [1, dst[0]], [2, pb(0)]], w, 1],
          ["gauss_v", [[0, this._b("ty")], [1, dst[1]], [2, pb(0)]], w, 1],
          ["gauss_v", [[0, this._b("tb")], [1, dst[2]], [2, pb(0)]], w, 1],
        ]);
      };

      mulChain("s1", i1, i1, sg1);
      mulChain("s2", i2, i2, sg2);
      mulChain("s3", i1, i2, sg12);

      // SSIM + EdgeDiff workgroup passes write to ssimWg / edWg.
      // They share a pass because they write to *different* intermediate buffers.
      const ssimWgJobs = [];
      const edWgJobs = [];
      for (let c = 0; c < 3; c++) {
        ssimWgJobs.push(["ssim_wg", [
          [0, mu1[c]], [1, mu2[c]],
          [2, sg1[c]], [3, sg2[c]], [4, sg12[c]],
          [5, this._b("ssimWg")],
          [6, pb(1 + c)],
        ], nwgX, nwgY]);
        edWgJobs.push(["edgediff_wg", [
          [0, i1[c]], [1, mu1[c]], [2, i2[c]], [3, mu2[c]],
          [4, this._b("edWg")],
          [5, pb(7 + c)],
        ], nwgX, nwgY]);
      }
      pass("wg", [...ssimWgJobs, ...edWgJobs]);

      // Reduction passes read back from ssimWg / edWg and write to final.
      const ssimRdJobs = [];
      const edRdJobs = [];
      for (let c = 0; c < 3; c++) {
        ssimRdJobs.push(["ssim_reduce", [
          [0, this._b("ssimWg")],
          [1, this._b("final")],
          [2, pb(4 + c)],
        ], 1, 1]);
        edRdJobs.push(["edgediff_reduce", [
          [0, this._b("edWg")],
          [1, this._b("final")],
          [2, pb(10 + c)],
        ], 1, 1]);
      }
      pass("rd", [...ssimRdJobs, ...edRdJobs]);

      d.queue.submit([encoder.finish()]);
    }

    const numOut = scales.length * OUT_PER_SCALE;
    const readSize = numOut * 4;
    const readEncoder = d.createCommandEncoder();
    readEncoder.copyBufferToBuffer(b.final, 0, b.staging, 0, readSize);
    d.queue.submit([readEncoder.finish()]);

    await b.staging.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(b.staging.getMappedRange(0, readSize));
    const result = new Float32Array(raw);
    b.staging.unmap();
    return result;
  }
}
