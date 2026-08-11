import { describe, expect, it } from "vitest";
import { orbitDirections } from "./strokeViews";

/** Elevation above the horizon of a (not necessarily normalized) direction. */
function pitchOf(direction: { x: number; y: number; z: number }): number {
  const length = Math.hypot(direction.x, direction.y, direction.z);
  return Math.asin(direction.y / length);
}

function yawOf(direction: { x: number; z: number }): number {
  const yaw = Math.atan2(direction.x, direction.z);
  return yaw < 0 ? yaw + 2 * Math.PI : yaw;
}

describe("orbitDirections", () => {
  it("spaces the yaws evenly whatever the layout", () => {
    for (const layout of ["ring", "helix"] as const) {
      const yaws = orbitDirections(4, 0.35, layout, 1.2).map(yawOf);
      expect(yaws.map((y) => Math.round((y * 180) / Math.PI))).toEqual([
        0, 90, 180, 270,
      ]);
    }
  });

  it("holds one elevation for a ring", () => {
    for (const direction of orbitDirections(5, 0.35, "ring", 1.2)) {
      expect(pitchOf(direction)).toBeCloseTo(0.35, 6);
    }
  });

  it("climbs from pitch to pitchMax across a helix", () => {
    const pitches = orbitDirections(5, 0.2, "helix", 1.0).map(pitchOf);
    expect(pitches[0]).toBeCloseTo(0.2, 6);
    expect(pitches[pitches.length - 1]).toBeCloseTo(1.0, 6);
    for (let i = 1; i < pitches.length; i++) {
      expect(pitches[i]).toBeGreaterThan(pitches[i - 1]);
    }
  });

  // straight down leaves lookAt with no way to choose a roll, so the renderer
  // stops short of vertical; a spec asking for more must be clamped rather
  // than produce an arbitrarily spun image
  it("stops short of vertical", () => {
    const pitches = orbitDirections(3, 0.2, "helix", Math.PI / 2).map(pitchOf);
    expect(Math.max(...pitches)).toBeLessThan(Math.PI / 2 - 0.1);
  });

  // count - 1 is the helix's denominator, and a single view has nowhere to
  // climb to
  it("survives a one-view helix", () => {
    const [only] = orbitDirections(1, 0.35, "helix", 1.2);
    expect(pitchOf(only)).toBeCloseTo(0.35, 6);
  });

  it("defaults to the old constant-pitch ring", () => {
    const bare = orbitDirections(4, 0.35);
    expect(bare).toEqual(orbitDirections(4, 0.35, "ring", 0.35));
  });
});
