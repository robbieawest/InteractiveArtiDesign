import { describe, expect, it } from "vitest";
import { isTubeVertexCount, reconstructTubeCenterline } from "./tubeCurve";

// Builds a SketchLab-layout tube around a given centerline: rings of 3 verts
// (angles 0/120/240° in the xy plane) so the centroid and radius recover
// exactly, plus the two cap verts, in the order the exporter uses:
// [ring0a, ring0b, cap, ring0c, ring1a, ring1b, ring1c, ..., cap].
function buildTube(centers: number[][], radius: number): Float32Array {
  const ringVert = (c: number[], angle: number) => [
    c[0] + radius * Math.cos(angle),
    c[1] + radius * Math.sin(angle),
    c[2],
  ];
  const out: number[] = [];
  const angles = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3];
  centers.forEach((c, i) => {
    const [a, b, d] = angles.map((angle) => ringVert(c, angle));
    if (i === 0) {
      out.push(...a, ...b, ...[c[0], c[1], c[2] - 1] /* start cap */, ...d);
    } else {
      out.push(...a, ...b, ...d);
    }
  });
  const last = centers[centers.length - 1];
  out.push(last[0], last[1], last[2] + 1); // end cap
  return new Float32Array(out);
}

describe("isTubeVertexCount", () => {
  it("accepts 3k + 2 vertex counts", () => {
    expect(isTubeVertexCount(5)).toBe(true);
    expect(isTubeVertexCount(11)).toBe(true);
    expect(isTubeVertexCount(1409)).toBe(true);
  });

  it("rejects other counts", () => {
    expect(isTubeVertexCount(2)).toBe(false);
    expect(isTubeVertexCount(6)).toBe(false);
    expect(isTubeVertexCount(12)).toBe(false);
  });
});

describe("reconstructTubeCenterline", () => {
  it("recovers ring centroids and radii exactly", () => {
    const centers = [
      [0, 0, 0],
      [1, 0.5, 2],
      [2, 1, 4],
      [3, 1.5, 6],
    ];
    const { points, radii } = reconstructTubeCenterline(buildTube(centers, 0.4));

    expect(points.length).toBe(centers.length * 3);
    centers.forEach((c, i) => {
      expect(points[i * 3]).toBeCloseTo(c[0], 5);
      expect(points[i * 3 + 1]).toBeCloseTo(c[1], 5);
      expect(points[i * 3 + 2]).toBeCloseTo(c[2], 5);
      expect(radii[i]).toBeCloseTo(0.4, 5);
    });
  });

  it("rejects non-tube vertex counts", () => {
    expect(() => reconstructTubeCenterline(new Float32Array(12 * 3))).toThrow(
      /not a SketchLab tube/,
    );
  });
});
