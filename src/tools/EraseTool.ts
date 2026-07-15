import * as THREE from "three";
import type { SketchDocument } from "../core/SketchDocument";
import type { UndoStack } from "../core/undo";
import { removeStrokeCommand } from "../core/undo";
import type { Viewport } from "../engine/Viewport";
import type { StrokeRenderer } from "../engine/StrokeRenderer";
import { pickStrokeAtCursor } from "../engine/picking";

/** Left-drag erases every stroke the cursor passes over. Each removal is
 *  its own undoable command, like Penzil. */
export class EraseTool {
  private erasing = false;

  constructor(
    private readonly viewport: Viewport,
    private readonly doc: SketchDocument,
    private readonly undo: UndoStack,
    private readonly strokeRenderer: StrokeRenderer,
  ) {}

  attach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.addEventListener("pointerdown", this.onPointerDown);
    dom.addEventListener("pointermove", this.onPointerMove);
    dom.addEventListener("pointerup", this.onPointerUp);
    dom.addEventListener("pointercancel", this.onPointerUp);
  }

  detach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.removeEventListener("pointerdown", this.onPointerDown);
    dom.removeEventListener("pointermove", this.onPointerMove);
    dom.removeEventListener("pointerup", this.onPointerUp);
    dom.removeEventListener("pointercancel", this.onPointerUp);
    this.erasing = false;
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.viewport.cameraOwnsPointer) return;
    this.erasing = true;
    this.eraseAt(event);
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (this.erasing) this.eraseAt(event);
  };

  private onPointerUp = (): void => {
    this.erasing = false;
  };

  private eraseAt(event: PointerEvent): void {
    const dom = this.viewport.renderer.domElement;
    const rect = dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const hitId = pickStrokeAtCursor(
      ndc,
      this.viewport,
      this.doc,
      this.strokeRenderer,
    );
    if (hitId) {
      this.undo.push(removeStrokeCommand(this.doc, hitId));
    }
  }
}
