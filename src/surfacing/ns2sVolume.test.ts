import { describe, expect, it } from "vitest";
import { decodeOccupancyVolume, type VolumeAlign } from "./ns2sVolume";

/** Build a bundle the way the adapter does: magic, version, header, voxels. */
function bundle(options: {
  grid: number;
  align?: VolumeAlign | null;
  max?: number;
  mean?: number;
  version?: number;
  magic?: string;
  truncate?: number;
  /** Bytes to lop off the end of the payload, leaving the header intact. */
  dropVoxels?: number;
  fill?: number;
}): ArrayBuffer {
  const { grid } = options;
  const header = JSON.stringify({
    grid,
    align: options.align ?? null,
    max: options.max ?? 1,
    mean: options.mean ?? 0,
  });
  const headerBytes = new TextEncoder().encode(header);
  const payload = new Uint8Array(grid ** 3).fill(options.fill ?? 0);

  const total = 12 + headerBytes.length + payload.length;
  const bytes = new Uint8Array(total);
  bytes.set(new TextEncoder().encode(options.magic ?? "NSVX"), 0);
  const view = new DataView(bytes.buffer);
  view.setUint32(4, options.version ?? 1, true);
  view.setUint32(8, headerBytes.length, true);
  bytes.set(headerBytes, 12);
  bytes.set(payload, 12 + headerBytes.length);
  return bytes.buffer.slice(0, options.truncate ?? total - (options.dropVoxels ?? 0));
}

const ALIGN: VolumeAlign = {
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  scale: 2.4,
  translation: [1, 2, 3],
};

describe("decodeOccupancyVolume", () => {
  it("reads the grid, the alignment and the whole payload", () => {
    const volume = decodeOccupancyVolume(
      bundle({ grid: 4, align: ALIGN, max: 0.97, mean: 0.03, fill: 128 }),
    );
    expect(volume.grid).toBe(4);
    expect(volume.align).toEqual(ALIGN);
    expect(volume.voxels.length).toBe(64);
    expect(volume.voxels[63]).toBe(128);
    expect(volume.max).toBeCloseTo(0.97);
    expect(volume.mean).toBeCloseTo(0.03);
  });

  it("reports a missing alignment rather than inventing one", () => {
    expect(decodeOccupancyVolume(bundle({ grid: 2 })).align).toBeNull();
  });

  it("rejects a bundle it does not recognise", () => {
    expect(() => decodeOccupancyVolume(bundle({ grid: 2, magic: "TRLZ" })))
      .toThrow(/not a probability volume/);
    expect(() => decodeOccupancyVolume(bundle({ grid: 2, version: 2 })))
      .toThrow(/version 2/);
    expect(() => decodeOccupancyVolume(new Uint8Array(4).buffer))
      .toThrow(/truncated/);
  });

  it("rejects a short payload instead of rendering a misread field", () => {
    expect(() => decodeOccupancyVolume(bundle({ grid: 4, dropVoxels: 10 })))
      .toThrow(/needs 64 bytes/);
  });
});
