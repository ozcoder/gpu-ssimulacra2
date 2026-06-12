export async function initWebGPU() {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("No WebGPU adapter found.");
  }
  const device = await adapter.requestDevice({
    requiredLimits: adapter.limits,
  });
  return { adapter, device };
}

export function makeShader(device, code) {
  return device.createShaderModule({ code });
}

export function makePipeline(device, shader, entry) {
  return device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: entry },
  });
}

export function makeBindGroup(device, pipeline, entries) {
  const layout = pipeline.getBindGroupLayout(0);
  return device.createBindGroup({
    layout,
    entries: entries.map((e) => ({
      binding: e.binding,
      resource: e.resource,
    })),
  });
}

export function ceilDiv(a, b) {
  return Math.ceil(a / b);
}
