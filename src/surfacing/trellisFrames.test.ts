import { describe, expect, it } from "vitest";
import {
  decodeFlowFrames,
  locateFrame,
  stageLengths,
  type FlowAlign,
} from "./trellisFrames";

/** Build a bundle the way the adapter does: magic, version, header, frames. */
function bundle(options: {
  grid: number;
  structure: number;
  latent: number;
  align?: FlowAlign | null;
  version?: number;
  truncate?: number;
  fill?: (stage: string, step: number) => number;
}): ArrayBuffer {
  const { grid, structure, latent } = options;
  const frameBytes = grid ** 3;
  const header = JSON.stringify({
    grid,
    frameBytes,
    align: options.align ?? null,
    stages: [
      { name: "structure", offset: 0, steps: structure },
      { name: "latent", offset: structure * frameBytes, steps: latent },
    ],
  });
  const headerBytes = new TextEncoder().encode(header);
  const payload = new Uint8Array((structure + latent) * frameBytes);
  if (options.fill) {
    for (let step = 0; step < structure; step++) {
      payload.fill(
        options.fill("structure", step),
        step * frameBytes,
        (step + 1) * frameBytes,
      );
    }
    for (let step = 0; step < latent; step++) {
      const base = (structure + step) * frameBytes;
      payload.fill(options.fill("latent", step), base, base + frameBytes);
    }
  }

  const total = 12 + headerBytes.length + payload.length;
  const bytes = new Uint8Array(total);
  bytes.set(new TextEncoder().encode("TRLZ"), 0);
  const view = new DataView(bytes.buffer);
  view.setUint32(4, options.version ?? 1, true);
  view.setUint32(8, headerBytes.length, true);
  bytes.set(headerBytes, 12);
  bytes.set(payload, 12 + headerBytes.length);
  return bytes.buffer.slice(0, options.truncate ?? total);
}

describe("decodeFlowFrames", () => {
  it("splits the payload into one volume per step, per stage", () => {
    const frames = decodeFlowFrames(
      bundle({ grid: 4, structure: 3, latent: 2 }),
    );
    expect(frames.grid).toBe(4);
    expect(stageLengths(frames)).toEqual({
      structure: 3,
      latent: 2,
      total: 5,
    });
    for (const volume of frames.stages.structure) {
      expect(volume.length).toBe(64);
    }
  });

  it("keeps frames in step order and does not mix the stages", () => {
    // each frame is filled with a value that identifies it, so a wrong
    // stride or offset shows up as the wrong byte rather than as a crash
    const frames = decodeFlowFrames(
      bundle({
        grid: 2,
        structure: 2,
        latent: 3,
        fill: (stage, step) => (stage === "structure" ? 10 + step : 20 + step),
      }),
    );
    expect(frames.stages.structure.map((v) => v[0])).toEqual([10, 11]);
    expect(frames.stages.latent.map((v) => v[0])).toEqual([20, 21, 22]);
  });

  it("carries the alignment through, and null when there was no fit", () => {
    const align: FlowAlign = {
      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      scale: 2.5,
      translation: [1, 2, 3],
    };
    expect(decodeFlowFrames(bundle({ grid: 2, structure: 1, latent: 1, align })).align)
      .toEqual(align);
    expect(decodeFlowFrames(bundle({ grid: 2, structure: 1, latent: 1 })).align)
      .toBeNull();
  });

  it("rejects a foreign buffer rather than reading noise as a volume", () => {
    const notABundle = new TextEncoder().encode("glTF____________").buffer;
    expect(() => decodeFlowFrames(notABundle)).toThrow(/not a flow bundle/);
  });

  it("rejects a version it does not know", () => {
    expect(() =>
      decodeFlowFrames(bundle({ grid: 2, structure: 1, latent: 0, version: 2 })),
    ).toThrow(/version 2/);
  });

  it("rejects a short payload instead of handing back a partial frame", () => {
    // one whole frame's worth of bytes missing from the end
    const full = bundle({ grid: 2, structure: 2, latent: 0 });
    expect(() =>
      decodeFlowFrames(
        bundle({ grid: 2, structure: 2, latent: 0, truncate: full.byteLength - 8 }),
      ),
    ).toThrow(/short/);
  });
});

describe("locateFrame", () => {
  const frames = decodeFlowFrames(
    bundle({
      grid: 2,
      structure: 2,
      latent: 3,
      fill: (stage, step) => (stage === "structure" ? 10 + step : 20 + step),
    }),
  );

  it("walks structure first, then latent", () => {
    expect(locateFrame(frames, 0)).toMatchObject({ stage: "structure", step: 0 });
    expect(locateFrame(frames, 1)).toMatchObject({ stage: "structure", step: 1 });
    expect(locateFrame(frames, 2)).toMatchObject({ stage: "latent", step: 0 });
    expect(locateFrame(frames, 4)).toMatchObject({ stage: "latent", step: 2 });
  });

  it("returns the volume that belongs to the position", () => {
    expect(locateFrame(frames, 3)?.volume[0]).toBe(21);
  });

  it("has nothing past the end", () => {
    expect(locateFrame(frames, 5)).toBeNull();
  });
});
