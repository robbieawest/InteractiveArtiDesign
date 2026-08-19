import { describe, expect, it } from "vitest";
import { blurVolume, gaussianKernel } from "./volumeBlur";

/** A field with one lit voxel at the centre of an odd-sized grid. */
function impulse(grid: number, value = 255): Uint8Array {
  const voxels = new Uint8Array(grid ** 3);
  const middle = (grid - 1) / 2;
  voxels[middle + middle * grid + middle * grid * grid] = value;
  return voxels;
}

const at = (voxels: Uint8Array, grid: number, x: number, y: number, z: number) =>
  voxels[x + y * grid + z * grid * grid];

describe("gaussianKernel", () => {
  it("is normalized counting both tails", () => {
    const kernel = gaussianKernel(1.5);
    const total = kernel.reduce((sum, w, k) => sum + (k === 0 ? w : 2 * w), 0);
    expect(total).toBeCloseTo(1, 6);
  });

  it("falls off monotonically and reaches out to 3 sigma", () => {
    const kernel = gaussianKernel(2);
    expect(kernel.length - 1).toBe(6);
    for (let k = 1; k < kernel.length; k++) {
      expect(kernel[k]).toBeLessThan(kernel[k - 1]);
    }
  });
});

describe("blurVolume", () => {
  it("passes the field through untouched at sigma 0", () => {
    const voxels = impulse(5);
    expect(blurVolume(voxels, 5, 0)).toBe(voxels);
    expect(blurVolume(voxels, 5, -1)).toBe(voxels);
  });

  it("spreads an impulse symmetrically and leaves the peak highest", () => {
    const grid = 9;
    const blurred = blurVolume(impulse(grid), grid, 1);
    const middle = (grid - 1) / 2;
    const peak = at(blurred, grid, middle, middle, middle);

    expect(peak).toBeGreaterThan(0);
    for (const [x, y, z] of [
      [middle + 1, middle, middle],
      [middle - 1, middle, middle],
      [middle, middle + 1, middle],
      [middle, middle - 1, middle],
      [middle, middle, middle + 1],
      [middle, middle, middle - 1],
    ]) {
      const neighbour = at(blurred, grid, x, y, z);
      expect(neighbour).toBeGreaterThan(0);
      expect(neighbour).toBeLessThan(peak);
    }
    // the six face neighbours are one kernel step away along different axes,
    // so an isotropic blur gives them all the same value
    expect(at(blurred, grid, middle + 1, middle, middle)).toBe(
      at(blurred, grid, middle, middle + 1, middle),
    );
    expect(at(blurred, grid, middle, middle, middle - 1)).toBe(
      at(blurred, grid, middle + 1, middle, middle),
    );
  });

  it("softens the peak further as sigma grows", () => {
    const grid = 9;
    const middle = (grid - 1) / 2;
    const gentle = blurVolume(impulse(grid), grid, 0.7);
    const heavy = blurVolume(impulse(grid), grid, 2);
    expect(at(heavy, grid, middle, middle, middle)).toBeLessThan(
      at(gentle, grid, middle, middle, middle),
    );
  });

  it("keeps a constant field constant, including at the edges", () => {
    const grid = 6;
    const flat = new Uint8Array(grid ** 3).fill(200);
    const blurred = blurVolume(flat, grid, 1.5);
    // clamped edges: a border voxel borrows itself, so nothing darkens
    expect([...blurred].every((value) => value === 200)).toBe(true);
  });

  it("rejects a field that is not grid ** 3 long", () => {
    expect(() => blurVolume(new Uint8Array(10), 4, 1)).toThrow(/expected 64/);
  });
});
