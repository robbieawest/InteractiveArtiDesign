import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { SketchDocument } from "../core/SketchDocument";
import type { UndoStack, Command } from "../core/undo";
import { removeStrokeCommand } from "../core/undo";
import type { Transform } from "../core/types";
import { cloneTransform } from "../core/types";
import type { Viewport } from "../engine/Viewport";
import type { StrokeRenderer } from "../engine/StrokeRenderer";
import type { GizmoMode } from "../engine/CanvasSurface";
import { pickStrokeAtCursor } from "../engine/picking";

/**
 * Click a stroke to select it; a transform gizmo attaches to its group.
 * Dragging the gizmo moves the live mesh; on release the whole drag is
 * committed to the document as one undoable transform command. Click empty
 * space or press Escape to deselect; Delete removes the selection.
 */
export class SelectTool {
  private selectedId?: string;
  private beforeDrag?: Transform;
  private controls?: TransformControls;
  private gizmoMode: GizmoMode = "translate";

  constructor(
    private readonly viewport: Viewport,
    private readonly doc: SketchDocument,
    private readonly undo: UndoStack,
    private readonly strokeRenderer: StrokeRenderer,
  ) {}

  attach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.addEventListener("pointerdown", this.onPointerDown);
    window.addEventListener("keydown", this.onKeyDown);
  }

  detach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.removeEventListener("pointerdown", this.onPointerDown);
    window.removeEventListener("keydown", this.onKeyDown);
    this.deselect();
  }

  setGizmoMode(mode: GizmoMode): void {
    this.gizmoMode = mode;
    this.controls?.setMode(mode);
    this.viewport.invalidate();
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.viewport.cameraOwnsPointer) return;
    // clicks on the gizmo belong to the gizmo
    if (this.controls && (this.controls.dragging || this.controls.axis)) {
      return;
    }

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
      this.select(hitId);
    } else {
      this.deselect();
    }
  };

  private onKeyDown = (event: KeyboardEvent): void => {
    if (event.target instanceof HTMLElement && event.target.tagName !== "BODY")
      return;
    if (event.code === "Escape") {
      this.deselect();
    } else if (
      (event.code === "Delete" || event.code === "Backspace") &&
      this.selectedId
    ) {
      const id = this.selectedId;
      this.deselect();
      this.undo.push(removeStrokeCommand(this.doc, id));
    }
  };

  private select(id: string): void {
    if (this.selectedId === id) return;
    this.deselect();
    const group = this.strokeRenderer.groupFor(id);
    if (!group) return;

    this.selectedId = id;
    this.controls = new TransformControls(
      this.viewport.camera,
      this.viewport.renderer.domElement,
    );
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
    this.controls.attach(group);
    this.viewport.invalidate();
  }

  private deselect(): void {
    if (!this.controls) return;
    this.controls.detach();
    this.viewport.scene.remove(this.controls.getHelper());
    this.controls.dispose();
    this.controls = undefined;
    this.selectedId = undefined;
    this.viewport.invalidate();
  }

  private onDragStart(): void {
    const stroke = this.selectedId && this.doc.getStroke(this.selectedId);
    if (stroke) this.beforeDrag = cloneTransform(stroke.transform);
  }

  private onDragEnd(): void {
    const id = this.selectedId;
    const before = this.beforeDrag;
    this.beforeDrag = undefined;
    if (!id || !before) return;
    const group = this.strokeRenderer.groupFor(id);
    if (!group) return;

    const after: Transform = {
      position: { x: group.position.x, y: group.position.y, z: group.position.z },
      quaternion: {
        x: group.quaternion.x,
        y: group.quaternion.y,
        z: group.quaternion.z,
        w: group.quaternion.w,
      },
      scale: { x: group.scale.x, y: group.scale.y, z: group.scale.z },
    };

    const doc = this.doc;
    const command: Command = {
      label: "Transform stroke",
      execute: () => doc.setStrokeTransform(id, after),
      undo: () => doc.setStrokeTransform(id, before),
    };
    this.undo.push(command);
  }
}
