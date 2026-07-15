import { describe, expect, it } from "vitest";
import { SketchDocument } from "./SketchDocument";
import type { DocumentEvent } from "./SketchDocument";
import {
  UndoStack,
  addStrokeCommand,
  removeStrokeCommand,
  setStrokeTransformCommand,
} from "./undo";
import type { Stroke } from "./types";
import { identityTransform, newStrokeId } from "./types";

function makeStroke(overrides: Partial<Stroke> = {}): Stroke {
  return {
    id: newStrokeId(),
    points: new Float32Array([0, 0, 0, 1, 0, 0, 2, 1, 0]),
    pressure: new Float32Array([0, 0.5, 1]),
    style: { visible: true, color: "#1c1c1e", width: 0.01 },
    fill: { visible: false, color: "#1c1c1e" },
    transform: identityTransform(),
    surface: { shape: "plane", transform: identityTransform() },
    ...overrides,
  };
}

describe("SketchDocument", () => {
  it("adds, gets and removes strokes, emitting events", () => {
    const doc = new SketchDocument();
    const events: DocumentEvent[] = [];
    doc.subscribe((e) => events.push(e));

    const stroke = makeStroke();
    doc.addStroke(stroke);
    expect(doc.strokeCount).toBe(1);
    expect(doc.getStroke(stroke.id)).toBe(stroke);

    doc.removeStroke(stroke.id);
    expect(doc.strokeCount).toBe(0);

    expect(events.map((e) => e.type)).toEqual(["strokeAdded", "strokeRemoved"]);
  });

  it("rejects duplicate ids", () => {
    const doc = new SketchDocument();
    const stroke = makeStroke();
    doc.addStroke(stroke);
    expect(() => doc.addStroke(stroke)).toThrow(/duplicate/);
  });

  it("unsubscribe stops events", () => {
    const doc = new SketchDocument();
    let count = 0;
    const unsubscribe = doc.subscribe(() => count++);
    doc.addStroke(makeStroke());
    unsubscribe();
    doc.addStroke(makeStroke());
    expect(count).toBe(1);
  });
});

describe("UndoStack", () => {
  it("undoes and redoes stroke addition", () => {
    const doc = new SketchDocument();
    const undo = new UndoStack();
    const stroke = makeStroke();

    undo.push(addStrokeCommand(doc, stroke));
    expect(doc.strokeCount).toBe(1);

    undo.undo();
    expect(doc.strokeCount).toBe(0);
    expect(undo.canRedo).toBe(true);

    undo.redo();
    expect(doc.strokeCount).toBe(1);
    expect(doc.getStroke(stroke.id)).toBeDefined();
  });

  it("undoes removal by restoring the same stroke", () => {
    const doc = new SketchDocument();
    const undo = new UndoStack();
    const stroke = makeStroke();
    doc.addStroke(stroke);

    undo.push(removeStrokeCommand(doc, stroke.id));
    expect(doc.strokeCount).toBe(0);
    undo.undo();
    expect(doc.getStroke(stroke.id)).toBe(stroke);
  });

  it("undoes transforms back to the prior transform", () => {
    const doc = new SketchDocument();
    const undo = new UndoStack();
    const stroke = makeStroke();
    doc.addStroke(stroke);

    const moved = identityTransform();
    moved.position.x = 5;
    undo.push(setStrokeTransformCommand(doc, stroke.id, moved));
    expect(doc.getStroke(stroke.id)!.transform.position.x).toBe(5);

    undo.undo();
    expect(doc.getStroke(stroke.id)!.transform.position.x).toBe(0);
  });

  it("a new command clears the redo stack", () => {
    const doc = new SketchDocument();
    const undo = new UndoStack();
    undo.push(addStrokeCommand(doc, makeStroke()));
    undo.undo();
    undo.push(addStrokeCommand(doc, makeStroke()));
    expect(undo.canRedo).toBe(false);
  });
});
