import { describe, expect, it } from "vitest";
import { buildRibbonGeometry, computeVertexWidths } from "./ribbon";

describe("computeVertexWidths", () => {
  it("tapers to zero at both ends and is widest in the bulk", () => {
    const pressure = new Float32Array(20).fill(0.5);
    const widths = computeVertexWidths(pressure);

    expect(widths[0]).toBe(0);
    expect(widths[19]).toBe(0);
    // bulk = base 3 + 0.5 * 16
    expect(widths[10]).toBeCloseTo(11);
    // ramps monotonically through the tails
    expect(widths[1]).toBeGreaterThan(0);
    expect(widths[1]).toBeLessThan(widths[2]);
    expect(widths[18]).toBeLessThan(widths[17]);
  });

  it("pressure increases width", () => {
    const soft = computeVertexWidths(new Float32Array(10).fill(0));
    const hard = computeVertexWidths(new Float32Array(10).fill(1));
    expect(hard[5]).toBeGreaterThan(soft[5]);
  });
});

describe("buildRibbonGeometry", () => {
  it("emits two vertices per point and two triangles per segment", () => {
    const points = new Float32Array([0, 0, 0, 1, 0, 0, 2, 1, 0]); // 3 points
    const widths = new Float32Array([0, 5, 0]);
    const geometry = buildRibbonGeometry(points, widths);

    expect(geometry.attributes.position.count).toBe(6);
    expect(geometry.index!.count).toBe(2 * 2 * 3); // 2 segments × 2 tris × 3

    // duplicated vertices sit exactly on the centerline
    const pos = geometry.attributes.position.array;
    expect([pos[0], pos[1], pos[2]]).toEqual([pos[3], pos[4], pos[5]]);

    // side alternates +1/-1, width is per-point duplicated
    expect(Array.from(geometry.attributes.side.array)).toEqual([
      1, -1, 1, -1, 1, -1,
    ]);
    expect(Array.from(geometry.attributes.width.array)).toEqual([
      0, 0, 5, 5, 0, 0,
    ]);

    // previous of the first point clamps to itself, next of the last likewise
    const prev = geometry.attributes.previous.array;
    expect([prev[0], prev[1], prev[2]]).toEqual([0, 0, 0]);
    const next = geometry.attributes.next.array;
    expect([next[12], next[13], next[14]]).toEqual([2, 1, 0]);
  });
});
