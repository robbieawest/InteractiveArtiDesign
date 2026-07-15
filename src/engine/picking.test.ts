import { describe, expect, it } from "vitest";
import * as THREE from "three";
import { pickStroke } from "./picking";

function stroke(id: string, points: number[], matrix?: THREE.Matrix4) {
  return {
    id,
    points: new Float32Array(points),
    matrixWorld: matrix ?? new THREE.Matrix4(),
  };
}

describe("pickStroke", () => {
  // Ray shooting down -z through the origin.
  const ray = new THREE.Ray(
    new THREE.Vector3(0, 0, 10),
    new THREE.Vector3(0, 0, -1),
  );

  it("hits a stroke crossing the ray and misses a distant one", () => {
    const crossing = stroke("hit", [-1, 0, 0, 1, 0, 0]);
    const far = stroke("far", [-1, 5, 0, 1, 5, 0]);
    expect(pickStroke(ray, [crossing, far])).toBe("hit");
    expect(pickStroke(ray, [far])).toBeUndefined();
  });

  it("prefers the stroke nearest the ray origin", () => {
    const near = stroke("near", [-1, 0, 5, 1, 0, 5]);
    const behind = stroke("behind", [-1, 0, -5, 1, 0, -5]);
    expect(pickStroke(ray, [behind, near])).toBe("near");
  });

  it("respects the stroke's world matrix", () => {
    // Points far away locally, but translated onto the ray.
    const moved = stroke(
      "moved",
      [-1, 100, 0, 1, 100, 0],
      new THREE.Matrix4().makeTranslation(0, -100, 0),
    );
    expect(pickStroke(ray, [moved])).toBe("moved");
  });

  it("respects the threshold", () => {
    const offAxis = stroke("off", [-1, 0.2, 0, 1, 0.2, 0]);
    expect(pickStroke(ray, [offAxis], 0.1)).toBeUndefined();
    expect(pickStroke(ray, [offAxis], 0.3)).toBe("off");
  });
});
