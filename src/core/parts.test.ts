import { describe, expect, it } from "vitest";
import { SketchDocument } from "./SketchDocument";
import {
  UndoStack,
  collapseCommand,
  explodeStateCommand,
  type PartOffsetChange,
  type StrokeTransformChange,
} from "./undo";
import { computeExplodeLayout } from "./explode";
import { deserializeDocument, serializeDocument } from "./serialization";
import type { Stroke } from "./types";
import { cloneTransform, identityTransform, newStrokeId } from "./types";

function makeStroke(x: number, partId?: string): Stroke {
  const t = identityTransform();
  t.position.x = x;
  return {
    id: newStrokeId(),
    points: new Float32Array([0, 0, 0, 1, 0, 0]),
    pressure: new Float32Array(2),
    style: { visible: true, color: "#000", width: 0.01 },
    fill: { visible: false, color: "#000" },
    transform: t,
    surface: { shape: "plane", transform: identityTransform() },
    ...(partId && { partId }),
  };
}

function docWithTwoParts(): { doc: SketchDocument; a: Stroke; b: Stroke } {
  const doc = new SketchDocument();
  doc.addPart({ id: "p1", name: "Part 1" });
  doc.addPart({ id: "p2", name: "Part 2" });
  const a = makeStroke(-2, "p1");
  const b = makeStroke(2, "p2");
  doc.addStroke(a);
  doc.addStroke(b);
  return { doc, a, b };
}

describe("parts", () => {
  it("removePart unassigns its strokes", () => {
    const { doc, a } = docWithTwoParts();
    doc.removePart("p1");
    expect(doc.getStroke(a.id)!.partId).toBeUndefined();
    expect(doc.allParts().map((p) => p.id)).toEqual(["p2"]);
  });

  it("round-trips screw joints and migrates version-4 typed joints", () => {
    const { doc } = docWithTwoParts();
    doc.addJoint({
      id: "j1",
      name: "Joint 1",
      parentPartId: "p1",
      childPartId: "p2",
      pivot: { x: 1, y: 2, z: 3 },
      axis: { x: 0, y: 1, z: 0 },
      dofs: {
        translation: { range: [0, 0.5], value: 0.1 },
        twist: { range: [-1, 1], value: 0 },
        swingU: { range: [0, 0], value: 0 },
        swingV: { range: [-0.2, 0.2], value: 0 },
      },
    });
    const json = JSON.parse(JSON.stringify(serializeDocument(doc)));
    const restored = deserializeDocument(json);
    expect(restored.getJoint("j1")).toEqual(doc.getJoint("j1"));

    // a version-4 document stored typed joints with one scalar range/value
    json.version = 4;
    json.joints = [
      {
        id: "old",
        name: "Old",
        parentPartId: "p1",
        childPartId: "p2",
        type: "revolute",
        pivot: { x: 0, y: 0, z: 0 },
        axis: { x: 1, y: 0, z: 0 },
        range: [-0.5, 1.5],
        value: 0.25,
      },
    ];
    const migrated = deserializeDocument(json).getJoint("old")!;
    expect(migrated.dofs.twist).toEqual({ range: [-0.5, 1.5], value: 0.25 });
    expect(migrated.dofs.translation.range).toEqual([0, 0]);
    expect(migrated.dofs.swingU.range).toEqual([0, 0]);
  });

  it("round-trips parts, poses and exploded state through JSON", () => {
    const { doc } = docWithTwoParts();
    doc.addPose({
      id: "pose1",
      name: "Pose 1",
      thumbnail: "data:image/jpeg;base64,abc",
      transforms: { x: identityTransform() },
    });
    const restored = deserializeDocument(
      JSON.parse(JSON.stringify(serializeDocument(doc))),
    );
    expect(restored.allParts().map((p) => p.id).sort()).toEqual(["p1", "p2"]);
    expect(restored.allPoses()[0].name).toBe("Pose 1");
    expect(restored.allStrokes().every((s) => s.partId)).toBe(true);
  });
});

/** Build the command the explode tool pushes after dragging to `factor`.
 *  `offsets` are the factor-1 base offsets, computed once at rest — exactly
 *  how the tool caches them on activation. */
function explodeTo(
  doc: SketchDocument,
  offsets: Map<string, { x: number; y: number; z: number }>,
  factor: number,
) {
  const strokeChanges: StrokeTransformChange[] = [];
  const offsetChanges: PartOffsetChange[] = [];
  for (const [partId, base] of offsets) {
    const target = { x: base.x * factor, y: base.y * factor, z: base.z * factor };
    const current = doc.getPart(partId)?.explodeOffset;
    offsetChanges.push({
      partId,
      before: current ? { ...current } : undefined,
      after: factor > 0 ? target : undefined,
    });
    for (const stroke of doc.strokesInPart(partId)) {
      const after = cloneTransform(stroke.transform);
      after.position.x += target.x - (current?.x ?? 0);
      after.position.y += target.y - (current?.y ?? 0);
      after.position.z += target.z - (current?.z ?? 0);
      strokeChanges.push({
        id: stroke.id,
        before: cloneTransform(stroke.transform),
        after,
      });
    }
  }
  return explodeStateCommand(doc, "Adjust explode", strokeChanges, offsetChanges);
}

describe("explode / collapse", () => {
  it("pushes parts apart and stores reversible offsets", () => {
    const { doc, a, b } = docWithTwoParts();
    const undo = new UndoStack();

    undo.push(explodeTo(doc, computeExplodeLayout(doc).offsets, 1));
    expect(doc.exploded).toBe(true);
    const ax = doc.getStroke(a.id)!.transform.position.x;
    const bx = doc.getStroke(b.id)!.transform.position.x;
    expect(ax).toBeLessThan(-2); // moved further out
    expect(bx).toBeGreaterThan(2);
    expect(doc.getPart("p1")!.explodeOffset).toBeDefined();

    undo.push(collapseCommand(doc));
    expect(doc.exploded).toBe(false);
    expect(doc.getStroke(a.id)!.transform.position.x).toBeCloseTo(-2);
    expect(doc.getStroke(b.id)!.transform.position.x).toBeCloseTo(2);
    expect(doc.getPart("p1")!.explodeOffset).toBeUndefined();
  });

  it("collapse brings along strokes segmented while exploded", () => {
    const { doc } = docWithTwoParts();
    const undo = new UndoStack();
    undo.push(explodeTo(doc, computeExplodeLayout(doc).offsets, 1));

    // draw a new stroke at the exploded location of part 1 and segment it
    const added = makeStroke(-5, "p1");
    doc.addStroke(added);
    const offset = doc.getPart("p1")!.explodeOffset!;

    undo.push(collapseCommand(doc));
    expect(doc.getStroke(added.id)!.transform.position.x).toBeCloseTo(
      -5 - offset.x,
    );
  });

  it("explode then undo restores exactly", () => {
    const { doc, a } = docWithTwoParts();
    const undo = new UndoStack();
    undo.push(explodeTo(doc, computeExplodeLayout(doc).offsets, 1));
    undo.undo();
    expect(doc.exploded).toBe(false);
    expect(doc.getStroke(a.id)!.transform.position.x).toBeCloseTo(-2);
    expect(doc.getPart("p1")!.explodeOffset).toBeUndefined();
  });

  it("re-exploding to a larger factor scales from the original pose", () => {
    const { doc, a } = docWithTwoParts();
    const undo = new UndoStack();
    const base = computeExplodeLayout(doc).offsets;
    undo.push(explodeTo(doc, base, 1));
    const once = doc.getStroke(a.id)!.transform.position.x;
    // the layout is rest-based: asking again while exploded gives the same
    // base offsets (explode mode persists across tool switches, so the tool
    // recomputes them from an already-exploded document)
    for (const [partId, offset] of computeExplodeLayout(doc).offsets) {
      expect(offset.x).toBeCloseTo(base.get(partId)!.x);
      expect(offset.y).toBeCloseTo(base.get(partId)!.y);
      expect(offset.z).toBeCloseTo(base.get(partId)!.z);
    }
    undo.push(explodeTo(doc, base, 2));
    // twice the spread relative to rest, not relative to the first explode
    expect(doc.getStroke(a.id)!.transform.position.x).toBeCloseTo(
      -2 + (once - -2) * 2,
    );
    // dragging back down to zero restores the rest pose and clears offsets
    undo.push(explodeTo(doc, base, 0));
    expect(doc.exploded).toBe(false);
    expect(doc.getStroke(a.id)!.transform.position.x).toBeCloseTo(-2);
    expect(doc.getPart("p1")!.explodeOffset).toBeUndefined();
  });
});
