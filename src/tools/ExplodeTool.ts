import * as THREE from "three";
import type { SketchDocument } from "../core/SketchDocument";
import type { UndoStack } from "../core/undo";
import {
  collapseCommand,
  explodeStateCommand,
  type PartOffsetChange,
  type StrokeTransformChange,
} from "../core/undo";
import { computeExplodeLayout } from "../core/explode";
import type { Transform, Vec3 } from "../core/types";
import { cloneTransform } from "../core/types";
import type { Viewport } from "../engine/Viewport";

/** Upper bound on the explode factor (1 = the standard spread). */
const MAX_FACTOR = 4;
/** Dragging this fraction of the viewport's smaller dimension away from the
 *  model center adds one unit of explode factor. */
const DRAG_FRACTION = 0.2;

/**
 * Drag-to-explode: while active, left-dragging away from the model's center
 * pushes every part outward along its explode direction; dragging back in
 * reduces the spread, but never past the original pose (the factor is
 * clamped to [0, MAX_FACTOR] — no imploding). Each drag is one undoable
 * command; deactivating the tool collapses everything back to the original
 * pose exactly, including strokes added to parts while exploded.
 */
export class ExplodeTool {
  /** Per-part outward offset at factor 1, computed from the rest pose. */
  private baseOffsets = new Map<string, Vec3>();
  private center: Vec3 = { x: 0, y: 0, z: 0 };
  private dragging = false;
  private startFactor = 0;
  private startDistance = 0;
  private factor = 0;
  /** Stroke transforms and part offsets captured at drag start; each frame
   *  re-derives from these so nothing accumulates within a drag. */
  private baselineTransforms = new Map<string, Transform>();
  private strokeParts = new Map<string, string>();
  private baselineOffsets = new Map<string, Vec3 | undefined>();

  constructor(
    private readonly viewport: Viewport,
    private readonly doc: SketchDocument,
    private readonly undo: UndoStack,
  ) {}

  attach(): void {
    // a document loaded mid-explosion starts from its original pose so the
    // explode directions are computed at rest
    if (this.doc.allParts().some((p) => p.explodeOffset)) {
      this.undo.push(collapseCommand(this.doc));
    }
    const layout = computeExplodeLayout(this.doc);
    this.baseOffsets = layout.offsets;
    this.center = layout.center;
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
    if (this.dragging) this.endDrag();
    // turning explode off reverts to the original pose
    if (this.doc.allParts().some((p) => p.explodeOffset)) {
      this.undo.push(collapseCommand(this.doc));
    }
    this.baseOffsets.clear();
    this.viewport.invalidate();
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.viewport.cameraOwnsPointer) return;
    if (this.baseOffsets.size === 0) return;
    this.dragging = true;
    this.viewport.renderer.domElement.setPointerCapture(event.pointerId);
    // derive the factor from the stored offsets so undo/redo between drags
    // can't leave the tool out of sync with the document
    this.startFactor = this.currentFactor();
    this.factor = this.startFactor;
    this.startDistance = this.cursorDistance(event);

    this.baselineTransforms.clear();
    this.strokeParts.clear();
    this.baselineOffsets.clear();
    for (const partId of this.baseOffsets.keys()) {
      const offset = this.doc.getPart(partId)?.explodeOffset;
      this.baselineOffsets.set(partId, offset ? { ...offset } : undefined);
      for (const stroke of this.doc.strokesInPart(partId)) {
        this.baselineTransforms.set(stroke.id, cloneTransform(stroke.transform));
        this.strokeParts.set(stroke.id, partId);
      }
    }
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (!this.dragging) return;
    const rect = this.viewport.renderer.domElement.getBoundingClientRect();
    const scale = DRAG_FRACTION * Math.min(rect.width, rect.height);
    const raw =
      this.startFactor + (this.cursorDistance(event) - this.startDistance) / scale;
    this.applyFactor(Math.min(Math.max(raw, 0), MAX_FACTOR));
  };

  private onPointerUp = (): void => {
    if (this.dragging) this.endDrag();
  };

  /** Screen-space distance (px) between the cursor and the model center. */
  private cursorDistance(event: PointerEvent): number {
    const rect = this.viewport.renderer.domElement.getBoundingClientRect();
    const projected = new THREE.Vector3(
      this.center.x,
      this.center.y,
      this.center.z,
    ).project(this.viewport.camera);
    const cx = rect.left + ((projected.x + 1) / 2) * rect.width;
    const cy = rect.top + ((1 - projected.y) / 2) * rect.height;
    return Math.hypot(event.clientX - cx, event.clientY - cy);
  }

  private currentFactor(): number {
    for (const [partId, base] of this.baseOffsets) {
      const length = Math.hypot(base.x, base.y, base.z);
      if (length < 1e-9) continue;
      const offset = this.doc.getPart(partId)?.explodeOffset;
      return offset ? Math.hypot(offset.x, offset.y, offset.z) / length : 0;
    }
    return 0;
  }

  private applyFactor(factor: number): void {
    this.factor = factor;
    const delta = factor - this.startFactor;
    for (const [strokeId, baseline] of this.baselineTransforms) {
      const base = this.baseOffsets.get(this.strokeParts.get(strokeId)!)!;
      const t = cloneTransform(baseline);
      t.position.x += base.x * delta;
      t.position.y += base.y * delta;
      t.position.z += base.z * delta;
      this.doc.setStrokeTransform(strokeId, t);
    }
    for (const [partId, base] of this.baseOffsets) {
      this.doc.setPartExplodeOffset(
        partId,
        factor > 1e-9
          ? { x: base.x * factor, y: base.y * factor, z: base.z * factor }
          : undefined,
      );
    }
    this.doc.setExploded(factor > 1e-9);
    this.viewport.invalidate();
  }

  private endDrag(): void {
    this.dragging = false;
    if (this.factor === this.startFactor) return;

    const strokeChanges: StrokeTransformChange[] = [];
    for (const [id, before] of this.baselineTransforms) {
      const stroke = this.doc.getStroke(id);
      if (stroke) {
        strokeChanges.push({ id, before, after: cloneTransform(stroke.transform) });
      }
    }
    const offsetChanges: PartOffsetChange[] = [];
    for (const [partId, before] of this.baselineOffsets) {
      const after = this.doc.getPart(partId)?.explodeOffset;
      offsetChanges.push({ partId, before, after: after ? { ...after } : undefined });
    }
    this.baselineTransforms.clear();
    this.strokeParts.clear();
    this.baselineOffsets.clear();
    this.undo.push(
      explodeStateCommand(this.doc, "Adjust explode", strokeChanges, offsetChanges),
    );
  }
}
