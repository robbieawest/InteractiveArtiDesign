import type { Stroke, Transform } from "./types";
import { cloneTransform } from "./types";

export type DocumentEvent =
  | { type: "strokeAdded"; stroke: Stroke }
  | { type: "strokeRemoved"; stroke: Stroke }
  | { type: "strokeChanged"; stroke: Stroke }
  | { type: "cleared" };

export type DocumentListener = (event: DocumentEvent) => void;

/**
 * The source of truth for a sketch. Holds plain data (no three.js) and
 * notifies subscribers of every change so the engine can keep the rendered
 * scene in sync. All mutations go through methods on this class — commands
 * (undo/redo) call these, never the other way around.
 */
export class SketchDocument {
  private strokes = new Map<string, Stroke>();
  private listeners = new Set<DocumentListener>();

  subscribe(listener: DocumentListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  get strokeCount(): number {
    return this.strokes.size;
  }

  getStroke(id: string): Stroke | undefined {
    return this.strokes.get(id);
  }

  allStrokes(): Stroke[] {
    return [...this.strokes.values()];
  }

  addStroke(stroke: Stroke): void {
    if (this.strokes.has(stroke.id)) {
      throw new Error(`duplicate stroke id: ${stroke.id}`);
    }
    this.strokes.set(stroke.id, stroke);
    this.emit({ type: "strokeAdded", stroke });
  }

  removeStroke(id: string): Stroke | undefined {
    const stroke = this.strokes.get(id);
    if (stroke) {
      this.strokes.delete(id);
      this.emit({ type: "strokeRemoved", stroke });
    }
    return stroke;
  }

  setStrokeTransform(id: string, transform: Transform): void {
    const stroke = this.strokes.get(id);
    if (!stroke) throw new Error(`no such stroke: ${id}`);
    stroke.transform = cloneTransform(transform);
    this.emit({ type: "strokeChanged", stroke });
  }

  clear(): void {
    this.strokes.clear();
    this.emit({ type: "cleared" });
  }

  private emit(event: DocumentEvent): void {
    for (const listener of this.listeners) listener(event);
  }
}
