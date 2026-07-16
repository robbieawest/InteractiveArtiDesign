import { describe, expect, it } from "vitest";
import { SketchDocument } from "./SketchDocument";
import { UndoStack, collapseCommand, explodeCommand } from "./undo";
import { deserializeDocument, serializeDocument } from "./serialization";
import type { Stroke } from "./types";
import { identityTransform, newStrokeId } from "./types";

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

describe("explode / collapse", () => {
  it("pushes parts apart and stores reversible offsets", () => {
    const { doc, a, b } = docWithTwoParts();
    const undo = new UndoStack();

    undo.push(explodeCommand(doc));
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
    undo.push(explodeCommand(doc));

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
    undo.push(explodeCommand(doc));
    undo.undo();
    expect(doc.exploded).toBe(false);
    expect(doc.getStroke(a.id)!.transform.position.x).toBeCloseTo(-2);
    expect(doc.getPart("p1")!.explodeOffset).toBeUndefined();
  });
});
