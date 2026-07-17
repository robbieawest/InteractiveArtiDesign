import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { dofRangeVisual } from "./jointRangeVisual";

function worldPoints(mesh: THREE.Mesh): THREE.Vector3[] {
  const positions = mesh.geometry.getAttribute("position");
  const points: THREE.Vector3[] = [];
  for (let i = 0; i < positions.count; i++) {
    points.push(
      new THREE.Vector3().fromBufferAttribute(positions, i).add(mesh.position),
    );
  }
  return points;
}

describe("dofRangeVisual", () => {
  it("returns null for a locked range", () => {
    expect(dofRangeVisual("twist", [0, 0], 1)).toBeNull();
  });

  it("puts each rotational sector in the plane its DoF sweeps", () => {
    // joint-local frame: X = axis, Y = u, Z = v
    const flat: Record<string, (p: THREE.Vector3) => number> = {
      twist: (p) => p.x, // spins u about the axis → sector ⊥ axis
      swingU: (p) => p.y, // tilts the axis about u → sector ⊥ u
      swingV: (p) => p.z,
    };
    for (const dof of ["twist", "swingU", "swingV"] as const) {
      const mesh = dofRangeVisual(dof, [-0.5, 1], 1)!;
      for (const p of worldPoints(mesh)) {
        expect(flat[dof](p)).toBeCloseTo(0);
      }
    }
  });

  it("starts rotational sectors at the rest direction", () => {
    // a [0, π/2] twist sweeps u (+Y) toward v (+Z), right-handed about X
    const points = worldPoints(dofRangeVisual("twist", [0, Math.PI / 2], 1)!);
    for (const p of points) {
      expect(p.y).toBeGreaterThanOrEqual(-1e-6);
      expect(p.z).toBeGreaterThanOrEqual(-1e-6);
    }
    // and reaches both ends of the span
    expect(Math.max(...points.map((p) => p.y))).toBeGreaterThan(0.5);
    expect(Math.max(...points.map((p) => p.z))).toBeGreaterThan(0.5);
  });

  it("covers the slide travel along the axis", () => {
    const points = worldPoints(dofRangeVisual("translation", [-0.25, 1], 2)!);
    expect(Math.min(...points.map((p) => p.x))).toBeCloseTo(-0.25);
    expect(Math.max(...points.map((p) => p.x))).toBeCloseTo(1);
  });
});
