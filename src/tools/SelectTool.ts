import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { SketchDocument } from "../core/SketchDocument";
import type { UndoStack, Command } from "../core/undo";
import type { Transform } from "../core/types";
import { cloneTransform } from "../core/types";
import type { Viewport } from "../engine/Viewport";
import type { StrokeRenderer } from "../engine/StrokeRenderer";
import type { GizmoMode } from "../engine/CanvasSurface";
import { pickStrokeAtCursor } from "../engine/picking";

/**
 * Click a stroke to select it; double-click to select its whole part.
 * Selected strokes are highlighted and share one transform gizmo, attached
 * to a pivot at the selection's center: while dragging, the stroke groups
 * are temporarily parented under the pivot so they follow it rigidly, and
 * on release every stroke's new transform is committed as a single undo
 * step. Delete removes the selection; Escape or empty-click deselects.
 */
export class SelectTool {
  private selection: string[] = [];
  /** Whether this tool currently owns the pointer/highlights. Selection can
   *  change while detached (e.g. clicking a part in the parts panel), in
   *  which case only the logical state updates — no visuals. */
  private attached = false;
  /** Set when the selection is exactly one whole part (double-click) —
   *  such selections glow purple and drawing auto-assigns to the part. */
  private partId: string | null = null;
  /** Fires when a selection becomes/stops being a part selection. */
  onPartSelectionChanged?: (partId: string | null) => void;
  private beforeDrag = new Map<string, Transform>();
  private pivot?: THREE.Object3D;
  private controls?: TransformControls;
  private gizmoMode: GizmoMode = "translate";

  constructor(
    private readonly viewport: Viewport,
    private readonly doc: SketchDocument,
    private readonly undo: UndoStack,
    private readonly strokeRenderer: StrokeRenderer,
  ) {}

  attach(): void {
    this.attached = true;
    const dom = this.viewport.renderer.domElement;
    dom.addEventListener("pointerdown", this.onPointerDown);
    dom.addEventListener("dblclick", this.onDoubleClick);
    window.addEventListener("keydown", this.onKeyDown);
    // selection survives tool switches; restore its visuals. A part
    // selection re-collects its strokes so ones drawn in the meantime are
    // included.
    if (this.partId && !this.doc.getPart(this.partId)) {
      this.setPartId(null); // part was deleted while another tool was active
    }
    if (this.partId) {
      this.selection = this.doc.strokesInPart(this.partId).map((s) => s.id);
    }
    this.selection = this.selection.filter((id) => this.doc.getStroke(id));
    if (this.selection.length > 0) {
      this.applyHighlights(true);
      this.buildGizmo();
    }
  }

  detach(): void {
    this.attached = false;
    const dom = this.viewport.renderer.domElement;
    dom.removeEventListener("pointerdown", this.onPointerDown);
    dom.removeEventListener("dblclick", this.onDoubleClick);
    window.removeEventListener("keydown", this.onKeyDown);
    // keep the logical selection (and part flag) but drop the visuals; the
    // next tool owns highlights from here
    this.destroyGizmo();
    this.applyHighlights(false);
  }

  get partSelectionId(): string | null {
    return this.partId;
  }

  setGizmoMode(mode: GizmoMode): void {
    this.gizmoMode = mode;
    this.controls?.setMode(mode);
    this.viewport.invalidate();
  }

  /** Hide the selection gizmo without deselecting (used while the camera
   *  is moving). */
  suppressGizmo(on: boolean): void {
    if (!this.controls) return;
    this.controls.enabled = !on;
    this.controls.getHelper().visible = !on;
    this.viewport.invalidate();
  }

  private pickAt(event: MouseEvent): string | undefined {
    const dom = this.viewport.renderer.domElement;
    const rect = dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    return pickStrokeAtCursor(ndc, this.viewport, this.doc, this.strokeRenderer);
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.viewport.cameraOwnsPointer) return;
    if (this.controls && (this.controls.dragging || this.controls.axis)) {
      return; // gizmo click
    }
    const hitId = this.pickAt(event);
    if (event.ctrlKey || event.metaKey) {
      // ctrl+click toggles strokes in/out; the result is a plain (orange)
      // multi-selection even if it started as a part selection
      if (!hitId) return;
      const ids = new Set(this.selection);
      if (ids.has(hitId)) {
        ids.delete(hitId);
      } else {
        ids.add(hitId);
      }
      this.select([...ids], null);
      return;
    }
    if (hitId) {
      if (!this.selection.includes(hitId)) this.select([hitId], null);
    } else {
      this.deselect();
    }
  };

  private onDoubleClick = (event: MouseEvent): void => {
    if (this.viewport.cameraOwnsPointer) return;
    if (this.controls && (this.controls.dragging || this.controls.axis)) return;
    const hitId = this.pickAt(event);
    if (!hitId) return;
    const partId = this.doc.getStroke(hitId)?.partId;
    if (partId) {
      this.select(
        this.doc.strokesInPart(partId).map((s) => s.id),
        partId,
      );
    } else {
      this.select([hitId], null);
    }
  };

  private onKeyDown = (event: KeyboardEvent): void => {
    if (event.target instanceof HTMLElement && event.target.tagName !== "BODY")
      return;
    if (event.code === "Escape") {
      this.deselect();
    } else if (
      (event.code === "Delete" || event.code === "Backspace") &&
      this.selection.length > 0
    ) {
      const doc = this.doc;
      const strokes = this.selection
        .map((id) => doc.getStroke(id))
        .filter((s) => s !== undefined);
      this.deselect();
      this.undo.push({
        label: "Delete strokes",
        execute: () => strokes.forEach((s) => doc.removeStroke(s.id)),
        undo: () => strokes.forEach((s) => doc.addStroke(s)),
      });
    }
  };

  select(ids: string[], partId: string | null = null): void {
    if (this.attached) {
      this.applyHighlights(false);
      this.destroyGizmo();
    }
    this.selection = ids.filter((id) => this.doc.getStroke(id));
    // an empty part selection is allowed: selecting a brand-new part means
    // "draw into this part" even before it has strokes
    this.setPartId(partId);
    if (this.selection.length === 0 || !this.attached) return;

    this.applyHighlights(true);
    this.buildGizmo();
  }

  deselect(): void {
    if (this.attached) this.applyHighlights(false);
    this.selection = [];
    this.setPartId(null);
    this.destroyGizmo();
    this.viewport.invalidate();
  }

  private setPartId(partId: string | null): void {
    if (partId === this.partId) return;
    this.partId = partId;
    this.onPartSelectionChanged?.(partId);
  }

  private applyHighlights(on: boolean): void {
    const kind = this.partId ? "part" : "stroke";
    for (const id of this.selection) {
      this.strokeRenderer.setHighlight(id, on, kind);
    }
  }

  private buildGizmo(): void {
    const groups = this.selectionGroups();
    if (groups.length === 0) return;

    // pivot at the selection's center so rotation/scale feel natural
    const center = new THREE.Vector3();
    for (const group of groups) center.add(group.position);
    center.divideScalar(groups.length);

    this.pivot = new THREE.Object3D();
    this.pivot.position.copy(center);
    this.viewport.scene.add(this.pivot);

    this.controls = new TransformControls(
      this.viewport.camera,
      this.viewport.renderer.domElement,
    );
    this.controls.setSize(0.5);
    this.controls.setMode(this.gizmoMode);
    this.controls.addEventListener("change", () => this.viewport.invalidate());
    this.controls.addEventListener("dragging-changed", (event) => {
      if (event.value) {
        this.onDragStart();
      } else {
        this.onDragEnd();
      }
    });
    this.viewport.scene.add(this.controls.getHelper());
    this.controls.attach(this.pivot);
    this.viewport.invalidate();
  }

  private destroyGizmo(): void {
    if (this.controls) {
      this.controls.detach();
      this.viewport.scene.remove(this.controls.getHelper());
      this.controls.dispose();
      this.controls = undefined;
    }
    if (this.pivot) {
      this.viewport.scene.remove(this.pivot);
      this.pivot = undefined;
    }
  }

  private selectionGroups(): THREE.Object3D[] {
    return this.selection
      .map((id) => this.strokeRenderer.groupFor(id))
      .filter((g) => g !== undefined);
  }

  private onDragStart(): void {
    if (!this.pivot) return;
    this.beforeDrag.clear();
    for (const id of this.selection) {
      const stroke = this.doc.getStroke(id);
      if (stroke) this.beforeDrag.set(id, cloneTransform(stroke.transform));
    }
    // parent the stroke groups under the pivot (keeping world transforms)
    // so the gizmo drives them all rigidly
    this.viewport.scene.updateMatrixWorld();
    for (const group of this.selectionGroups()) {
      this.pivot.attach(group);
    }
  }

  private onDragEnd(): void {
    if (!this.pivot) return;
    // hand the groups back to the scene, then commit their new transforms
    this.viewport.scene.updateMatrixWorld();
    const changes: { id: string; before: Transform; after: Transform }[] = [];
    for (const id of this.selection) {
      const group = this.strokeRenderer.groupFor(id);
      const before = this.beforeDrag.get(id);
      if (!group || !before) continue;
      this.viewport.scene.attach(group);
      changes.push({
        id,
        before,
        after: {
          position: { x: group.position.x, y: group.position.y, z: group.position.z },
          quaternion: {
            x: group.quaternion.x,
            y: group.quaternion.y,
            z: group.quaternion.z,
            w: group.quaternion.w,
          },
          scale: { x: group.scale.x, y: group.scale.y, z: group.scale.z },
        },
      });
    }
    this.beforeDrag.clear();
    if (changes.length === 0) return;

    const doc = this.doc;
    const command: Command = {
      label: "Transform strokes",
      execute: () => changes.forEach((c) => doc.setStrokeTransform(c.id, c.after)),
      undo: () => changes.forEach((c) => doc.setStrokeTransform(c.id, c.before)),
    };
    this.undo.push(command);

    // re-center the gizmo pivot on the moved selection
    const selection = [...this.selection];
    this.destroyGizmo();
    this.selection = selection;
    this.buildGizmo();
  }
}
