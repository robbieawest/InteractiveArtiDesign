import type { SketchDocument } from "./SketchDocument";
import type { Stroke, Transform } from "./types";
import { cloneTransform } from "./types";

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
