import type { SketchDocument } from "./SketchDocument";
import type { Pose, Stroke, Transform, Vec3 } from "./types";
import { cloneTransform } from "./types";
import { computeExplodeOffsets } from "./explode";

export interface Command {
  label: string;
  execute(): void;
  undo(): void;
}

/**
 * Classic command-pattern undo. `push` executes the command and records it;
 * undoing/redoing replays them. Commands close over the document, so the
 * stack itself is document-agnostic.
 */
export class UndoStack {
  private undos: Command[] = [];
  private redos: Command[] = [];
  /** Called after every push/undo/redo/clear, e.g. to refresh UI state. */
  onChange?: () => void;

  constructor(private readonly limit = 200) {}

  get canUndo(): boolean {
    return this.undos.length > 0;
  }

  get canRedo(): boolean {
    return this.redos.length > 0;
  }

  push(command: Command): void {
    command.execute();
    this.undos.push(command);
    if (this.undos.length > this.limit) this.undos.shift();
    this.redos = [];
    this.onChange?.();
  }

  undo(): void {
    const command = this.undos.pop();
    if (!command) return;
    command.undo();
    this.redos.push(command);
    this.onChange?.();
  }

  redo(): void {
    const command = this.redos.pop();
    if (!command) return;
    command.execute();
    this.undos.push(command);
    this.onChange?.();
  }

  clear(): void {
    this.undos = [];
    this.redos = [];
    this.onChange?.();
  }
}

export function addStrokeCommand(doc: SketchDocument, stroke: Stroke): Command {
  return {
    label: "Draw stroke",
    execute: () => doc.addStroke(stroke),
    undo: () => doc.removeStroke(stroke.id),
  };
}

export function removeStrokeCommand(doc: SketchDocument, id: string): Command {
  const stroke = doc.getStroke(id);
  if (!stroke) throw new Error(`no such stroke: ${id}`);
  return {
    label: "Erase stroke",
    execute: () => doc.removeStroke(id),
    undo: () => doc.addStroke(stroke),
  };
}

/** Assign/unassign a batch of strokes to a part in one undo step (one
 *  segmentation-pen drag). `before` records each stroke's prior part. */
export function setStrokesPartCommand(
  doc: SketchDocument,
  changes: { strokeId: string; before?: string; after?: string }[],
  label = "Segment strokes",
): Command {
  return {
    label,
    execute: () =>
      changes.forEach((c) => doc.setStrokePart(c.strokeId, c.after)),
    undo: () =>
      changes.forEach((c) => doc.setStrokePart(c.strokeId, c.before)),
  };
}

function shiftStroke(doc: SketchDocument, id: string, offset: Vec3, sign: 1 | -1) {
  const stroke = doc.getStroke(id);
  if (!stroke) return;
  const t = cloneTransform(stroke.transform);
  t.position.x += offset.x * sign;
  t.position.y += offset.y * sign;
  t.position.z += offset.z * sign;
  doc.setStrokeTransform(id, t);
}

/** Push all parts outward (exploded view). Offsets are stored on the parts
 *  so the collapse — and strokes segmented in the meantime — reverses them
 *  exactly. */
export function explodeCommand(doc: SketchDocument): Command {
  const offsets = computeExplodeOffsets(doc);
  const strokeIds = new Map<string, string[]>();
  for (const partId of offsets.keys()) {
    strokeIds.set(
      partId,
      doc.strokesInPart(partId).map((s) => s.id),
    );
  }
  const apply = (sign: 1 | -1) => {
    for (const [partId, offset] of offsets) {
      for (const id of strokeIds.get(partId)!) {
        shiftStroke(doc, id, offset, sign);
      }
      doc.setPartExplodeOffset(partId, sign === 1 ? offset : undefined);
    }
    doc.setExploded(sign === 1);
  };
  return {
    label: "Explode parts",
    execute: () => apply(1),
    undo: () => apply(-1),
  };
}

/** Reverse the stored explode offsets. Built at collapse time so strokes
 *  added to a part while exploded come along. */
export function collapseCommand(doc: SketchDocument): Command {
  const offsets = new Map<string, Vec3>();
  const strokeIds = new Map<string, string[]>();
  for (const part of doc.allParts()) {
    if (!part.explodeOffset) continue;
    offsets.set(part.id, { ...part.explodeOffset });
    strokeIds.set(
      part.id,
      doc.strokesInPart(part.id).map((s) => s.id),
    );
  }
  const apply = (sign: 1 | -1) => {
    for (const [partId, offset] of offsets) {
      for (const id of strokeIds.get(partId)!) {
        shiftStroke(doc, id, offset, sign);
      }
      doc.setPartExplodeOffset(partId, sign === 1 ? offset : undefined);
    }
    doc.setExploded(sign === 1);
  };
  return {
    label: "Collapse parts",
    execute: () => apply(-1),
    undo: () => apply(1),
  };
}

/** Apply a saved pose to every stroke it covers that still exists. */
export function applyPoseCommand(doc: SketchDocument, pose: Pose): Command {
  const changes: { id: string; before: Transform; after: Transform }[] = [];
  for (const [id, after] of Object.entries(pose.transforms)) {
    const stroke = doc.getStroke(id);
    if (stroke) {
      changes.push({
        id,
        before: cloneTransform(stroke.transform),
        after: cloneTransform(after),
      });
    }
  }
  return {
    label: `Apply pose ${pose.name}`,
    execute: () => changes.forEach((c) => doc.setStrokeTransform(c.id, c.after)),
    undo: () => changes.forEach((c) => doc.setStrokeTransform(c.id, c.before)),
  };
}

export function setStrokeTransformCommand(
  doc: SketchDocument,
  id: string,
  transform: Transform,
): Command {
  const stroke = doc.getStroke(id);
  if (!stroke) throw new Error(`no such stroke: ${id}`);
  const before = cloneTransform(stroke.transform);
  const after = cloneTransform(transform);
  return {
    label: "Transform stroke",
    execute: () => doc.setStrokeTransform(id, after),
    undo: () => doc.setStrokeTransform(id, before),
  };
}
