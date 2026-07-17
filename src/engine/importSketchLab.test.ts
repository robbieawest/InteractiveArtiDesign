import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { importSketchLabGltf } from "./importSketchLab";
import { JOINT_DOF_NAMES, dofUnlocked, jointKindLabel } from "../core/types";

// Integration test against the real SketchLab sample (the excavator from the
// "Rapid Design of Articulated Objects" paper, exported via Sketchfab).
const sampleDir = fileURLToPath(
  new URL("../../SampleModels/p2-c_autonomous_excavator/", import.meta.url),
);

function loadSample() {
  const json = JSON.parse(readFileSync(`${sampleDir}scene.gltf`, "utf8"));
  const buf = readFileSync(`${sampleDir}scene.bin`);
  const bin = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return importSketchLabGltf(json, bin);
}

describe("importSketchLabGltf (excavator sample)", () => {
  const { strokes, parts, joints } = loadSample();

  it("imports every stroke, part, and joint", () => {
    expect(strokes.length).toBe(973);
    expect(parts.length).toBe(23);
    expect(joints.length).toBe(22);
  });

  it("derives joint DoFs, axes, and ranges from the animation", () => {
    const partIds = new Set(parts.map((p) => p.id));
    const kinds = new Set(joints.map(jointKindLabel));
    expect(kinds.has("revolute")).toBe(true);
    expect(kinds.has("prismatic")).toBe(true);
    for (const joint of joints) {
      expect(partIds.has(joint.parentPartId)).toBe(true);
      expect(partIds.has(joint.childPartId)).toBe(true);
      // unit axis
      expect(
        Math.hypot(joint.axis.x, joint.axis.y, joint.axis.z),
      ).toBeCloseTo(1, 5);
      for (const dof of JOINT_DOF_NAMES) {
        expect(joint.dofs[dof].value).toBe(0);
        // ranges always contain the rest value
        expect(joint.dofs[dof].range[0]).toBeLessThanOrEqual(0);
        expect(joint.dofs[dof].range[1]).toBeGreaterThanOrEqual(0);
      }
      // the demo animation never swings off-axis
      expect(dofUnlocked(joint.dofs.swingU)).toBe(false);
      expect(dofUnlocked(joint.dofs.swingV)).toBe(false);
      // pivots live inside the model's few-units world box
      expect(Math.abs(joint.pivot.x)).toBeLessThan(10);
      expect(Math.abs(joint.pivot.y)).toBeLessThan(10);
      expect(Math.abs(joint.pivot.z)).toBeLessThan(10);
    }
    // each part is driven by at most one joint (a tree, not a graph)
    const children = joints.map((j) => j.childPartId);
    expect(new Set(children).size).toBe(children.length);
  });

  it("tags strokes with their owning part", () => {
    const partIds = new Set(parts.map((p) => p.id));
    let tagged = 0;
    for (const stroke of strokes) {
      if (stroke.partId !== undefined) {
        expect(partIds.has(stroke.partId)).toBe(true);
        tagged++;
      }
    }
    // Two curves sit outside any part in the sample file.
    expect(strokes.length - tagged).toBe(2);
    // Every part should own at least one stroke in this model.
    const used = new Set(strokes.map((s) => s.partId));
    for (const id of partIds) expect(used.has(id)).toBe(true);
  });

  it("bakes transforms to sane world coordinates (cm → units, y-up)", () => {
    const min = { x: Infinity, y: Infinity, z: Infinity };
    const max = { x: -Infinity, y: -Infinity, z: -Infinity };
    for (const stroke of strokes) {
      expect(stroke.transform.quaternion).toEqual({ x: 0, y: 0, z: 0, w: 1 });
      for (let i = 0; i < stroke.points.length; i += 3) {
        const x = stroke.points[i] + stroke.transform.position.x;
        const y = stroke.points[i + 1] + stroke.transform.position.y;
        const z = stroke.points[i + 2] + stroke.transform.position.z;
        min.x = Math.min(min.x, x);
        min.y = Math.min(min.y, y);
        min.z = Math.min(min.z, z);
        max.x = Math.max(max.x, x);
        max.y = Math.max(max.y, y);
        max.z = Math.max(max.z, z);
      }
    }
    // The excavator is a few units across once the 0.01 root scale is baked.
    const extent = Math.max(max.x - min.x, max.y - min.y, max.z - min.z);
    expect(extent).toBeGreaterThan(1);
    expect(extent).toBeLessThan(20);
    // y-up after the Sketchfab root rotation: the model stands taller than 0.
    expect(max.y).toBeGreaterThan(0);
  });

  it("recovers stroke style from the tube geometry and materials", () => {
    for (const stroke of strokes) {
      expect(stroke.style.color).toMatch(/^#[0-9a-f]{6}$/);
      expect(stroke.style.width).toBeGreaterThan(0);
      expect(stroke.style.width).toBeLessThan(0.1);
      expect(stroke.fill.visible).toBe(false);
      expect(stroke.pressure.length).toBe(stroke.points.length / 3);
    }
  });
});
