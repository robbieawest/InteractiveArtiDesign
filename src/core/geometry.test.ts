import { describe, expect, it } from "vitest";
import { recenterPoints } from "./geometry";

describe("recenterPoints", () => {
  it("moves the bounding-box center to the origin", () => {
    const { points, center } = recenterPoints(
      new Float32Array([1, 2, 3, 3, 6, 5]),
    );
    expect(center).toEqual({ x: 2, y: 4, z: 4 });
    expect(Array.from(points)).toEqual([-1, -2, -1, 1, 2, 1]);
  });

  it("handles empty input", () => {
    const { points, center } = recenterPoints(new Float32Array([]));
    expect(points.length).toBe(0);
    expect(center).toEqual({ x: 0, y: 0, z: 0 });
  });
});
