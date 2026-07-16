import * as THREE from "three";
import type { SketchDocument } from "../core/SketchDocument";
import type { UndoStack } from "../core/undo";
import { setStrokesPartCommand } from "../core/undo";
import type { Viewport } from "../engine/Viewport";
import type { StrokeRenderer } from "../engine/StrokeRenderer";
import { collectPickables, pickAllStrokes } from "../engine/picking";

export type SegmentMode = "add" | "remove";

/**
 * The segmentation pen: left-drag an invisible brush over the sketch and
 * every stroke it crosses — through-collision, front to back, not just the
 * nearest — is added to (or removed from) the active part. One drag is one
 * undo step. Strokes of the active part stay highlighted while the tool is
 * active, so painting gives immediate feedback.
 */
export class SegmentTool {
  mode: SegmentMode = "add";
  private activePartId?: string;
  private painting = false;
  /** Original partId of every stroke changed during the current drag. */
  private dragChanges = new Map<string, string | undefined>();
  private unsubscribe?: () => void;

  constructor(
    private readonly viewport: Viewport,
    private readonly doc: SketchDocument,
    private readonly undo: UndoStack,
    private readonly strokeRenderer: StrokeRenderer,
  ) {}

  setActivePart(partId: string | undefined): void {
    this.activePartId = partId;
    this.refreshHighlights();
  }

  attach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.addEventListener("pointerdown", this.onPointerDown);
    dom.addEventListener("pointermove", this.onPointerMove);
    dom.addEventListener("pointerup", this.onPointerUp);
    dom.addEventListener("pointercancel", this.onPointerUp);
    // keep highlights current when strokes change under the tool (undo etc.)
    this.unsubscribe = this.doc.subscribe(() => this.refreshHighlights());
    this.refreshHighlights();
  }

  detach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.removeEventListener("pointerdown", this.onPointerDown);
    dom.removeEventListener("pointermove", this.onPointerMove);
    dom.removeEventListener("pointerup", this.onPointerUp);
    dom.removeEventListener("pointercancel", this.onPointerUp);
    this.unsubscribe?.();
    this.unsubscribe = undefined;
    this.painting = false;
    this.strokeRenderer.clearHighlights();
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.viewport.cameraOwnsPointer) return;
    if (!this.activePartId) return;
    this.painting = true;
    this.dragChanges.clear();
    this.paintAt(event);
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (this.painting) this.paintAt(event);
  };

  private onPointerUp = (): void => {
    if (!this.painting) return;
    this.painting = false;
    if (this.dragChanges.size === 0) return;

    // Changes were applied live during the drag; record them as one command
    // (its execute is idempotent, so the stack's initial execute is a no-op
    // re-application).
    const after = this.mode === "add" ? this.activePartId : undefined;
    this.undo.push(
      setStrokesPartCommand(
        this.doc,
        [...this.dragChanges].map(([strokeId, before]) => ({
          strokeId,
          before,
          after,
        })),
        this.mode === "add" ? "Add strokes to part" : "Remove strokes from part",
      ),
    );
    this.dragChanges.clear();
  };

  private paintAt(event: PointerEvent): void {
    if (!this.activePartId) return;
    const dom = this.viewport.renderer.domElement;
    const rect = dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, this.viewport.camera);

    const hits = pickAllStrokes(
      raycaster.ray,
      collectPickables(this.viewport, this.doc, this.strokeRenderer),
      0.1,
    );

    for (const id of hits) {
      const stroke = this.doc.getStroke(id);
      if (!stroke) continue;
      if (this.mode === "add" && stroke.partId !== this.activePartId) {
        if (!this.dragChanges.has(id)) this.dragChanges.set(id, stroke.partId);
        this.doc.setStrokePart(id, this.activePartId);
      } else if (this.mode === "remove" && stroke.partId === this.activePartId) {
        if (!this.dragChanges.has(id)) this.dragChanges.set(id, stroke.partId);
        this.doc.setStrokePart(id, undefined);
      }
    }
  }

  private refreshHighlights(): void {
    for (const stroke of this.doc.allStrokes()) {
      this.strokeRenderer.setHighlight(
        stroke.id,
        stroke.partId !== undefined && stroke.partId === this.activePartId,
      );
    }
  }
}
