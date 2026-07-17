import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { SketchDocument } from "../core/SketchDocument";
import type { UndoStack } from "../core/undo";
import {
  addJointCommand,
  compoundCommand,
  removeJointCommand,
  resetArticulationCommand,
  updateJointCommand,
} from "../core/undo";
import type { Joint, JointDofName, Transform, Vec3 } from "../core/types";
import {
  JOINT_DOF_NAMES,
  cloneJoint,
  cloneTransform,
  jointPosed,
  lockedDofs,
} from "../core/types";
import {
  computePartDeltas,
  partsInSubtree,
  valuesOfPose,
  type JointValues,
} from "../core/articulation";
import {
  applyRigidToTransform,
  identityRigid,
  rigidInvert,
  rigidMultiply,
} from "../core/rigid";
import type { Viewport } from "../engine/Viewport";
import type { StrokeRenderer } from "../engine/StrokeRenderer";
import { pickStrokeAtCursor } from "../engine/picking";
import { dofRangeVisual } from "../engine/jointRangeVisual";
import { posedJointFrame, twistAbout } from "./ArticulateTool";

export interface JointToolState {
  /** Joint currently being edited (its axis gizmo is showing). */
  jointId: string | null;
  /** DoF being demonstrated, or null when placing the axis. */
  armedDof: JointDofName | null;
  mirror: boolean;
}

const RING_DOF: Record<string, JointDofName> = {
  X: "twist",
  Y: "swingU",
  Z: "swingV",
};

/**
 * Authoring and editing are one process: joints are "filled in" rather
 * than wizard-stepped.
 *
 * - Drag from one part to another to create a joint between them (parent →
 *   child); the new joint starts fully locked with its axis at the child's
 *   centroid, pointing from parent to child.
 * - Click a part to edit the joint that drives it. The gizmo places the
 *   axis: T translates the pivot, R aims the direction (rolling about the
 *   axis is meaningless, so that ring is hidden).
 * - Arm a DoF in the Articulations panel and drag the gizmo to demonstrate
 *   its range: the child moves live and the extremes you reach become the
 *   range ([min, max], always containing 0). Each armed session starts
 *   fresh, so re-demonstrating can shrink a range. With mirror on, the
 *   committed range is symmetrized. On release the part snaps back to rest.
 *
 * The selected joint's parent part highlights purple, the child yellow.
 * The tool always works at the rest pose: selecting or creating first
 * drives any posed joints back to zero (one undoable step).
 */
export class JointTool {
  private jointId: string | null = null;
  private armedDof: JointDofName | null = null;
  private mirror = false;

  /** Connect gesture in progress: parent part + candidate child. */
  private connectParent: string | null = null;
  private connectCandidate: string | null = null;
  private connectMoved = false;
  private downPos = new THREE.Vector2();

  private proxy?: THREE.Group;
  /** Range fills, fixed at the rest frame so they hold still while the
   *  child part moves during a demonstration. */
  private rangeGroup?: THREE.Group;
  private controls?: TransformControls;
  private pivotMode: "translate" | "rotate" = "translate";
  private dragging = false;
  private dragStartQuat = new THREE.Quaternion();
  private dragStartPos = new THREE.Vector3();
  /** Demonstration session state (reset when a DoF is armed). */
  private sessionMin = 0;
  private sessionMax = 0;
  private baselineTransforms = new Map<string, Transform>();

  onStateChanged?: (state: JointToolState) => void;

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
    window.addEventListener("keydown", this.onKeyDown);
  }

  detach(): void {
    const dom = this.viewport.renderer.domElement;
    dom.removeEventListener("pointerdown", this.onPointerDown);
    dom.removeEventListener("pointermove", this.onPointerMove);
    dom.removeEventListener("pointerup", this.onPointerUp);
    dom.removeEventListener("pointercancel", this.onPointerUp);
    window.removeEventListener("keydown", this.onKeyDown);
    this.deselect();
  }

  suppressGizmo(on: boolean): void {
    if (!this.controls) return;
    this.controls.enabled = !on;
    this.controls.getHelper().visible = !on;
    this.viewport.invalidate();
  }

  deselect(): void {
    this.setHighlights(false);
    this.jointId = null;
    this.armedDof = null;
    this.destroyGizmo();
    this.notify();
    this.viewport.invalidate();
  }

  /** Edit an existing joint (e.g. from the Articulations panel). */
  selectJoint(jointId: string): void {
    const joint = this.doc.getJoint(jointId);
    if (!joint) return;
    this.deselect();
    this.ensureRestPose();
    this.jointId = jointId;
    this.armedDof = null;
    this.setHighlights(true);
    this.buildGizmo();
    this.notify();
  }

  /** Arm a DoF for range demonstration (null returns to axis placement). */
  armDof(dof: JointDofName | null): void {
    if (!this.jointId || this.doc.exploded) return;
    this.armedDof = this.armedDof === dof ? null : dof;
    this.sessionMin = 0;
    this.sessionMax = 0;
    this.rebuildGizmo();
    this.notify();
  }

  setMirror(on: boolean): void {
    this.mirror = on;
    this.notify();
  }

  deleteJoint(jointId: string): void {
    const joint = this.doc.getJoint(jointId);
    if (!joint) return;
    if (this.jointId === jointId) this.deselect();
    const commands = [];
    if (this.doc.allJoints().some(jointPosed)) {
      // deleting a posed joint would orphan its motion; go to rest first
      commands.push(resetArticulationCommand(this.doc));
    }
    commands.push(removeJointCommand(this.doc, jointId));
    this.undo.push(compoundCommand("Delete joint", commands));
  }

  private notify(): void {
    this.onStateChanged?.({
      jointId: this.jointId,
      armedDof: this.armedDof,
      mirror: this.mirror,
    });
  }

  /** The Joint tool always works at the rest pose. */
  private ensureRestPose(): void {
    if (this.doc.allJoints().some(jointPosed)) {
      this.undo.push(resetArticulationCommand(this.doc));
    }
  }

  // --- pointer: click = edit, drag part→part = create ---

  private pointerDownEmpty = false;

  private onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.viewport.cameraOwnsPointer) return;
    if (this.controls && (this.controls.dragging || this.controls.axis)) {
      return; // gizmo interaction
    }
    const partId = this.partAt(event);
    this.downPos.set(event.clientX, event.clientY);
    this.connectMoved = false;
    if (!partId) {
      this.pointerDownEmpty = true;
      return;
    }
    this.pointerDownEmpty = false;
    this.connectParent = partId;
    this.connectCandidate = null;
    this.highlightPart(partId, true, "part");
  };

  private onPointerMove = (event: PointerEvent): void => {
    if (!this.connectParent) return;
    if (
      Math.hypot(event.clientX - this.downPos.x, event.clientY - this.downPos.y) > 5
    ) {
      this.connectMoved = true;
    }
    if (!this.connectMoved) return;
    const over = this.partAt(event);
    const candidate = over && over !== this.connectParent ? over : null;
    if (candidate !== this.connectCandidate) {
      if (this.connectCandidate) {
        this.highlightPart(this.connectCandidate, false, "jointChild");
      }
      this.connectCandidate = candidate;
      if (candidate) this.highlightPart(candidate, true, "jointChild");
    }
  };

  private onPointerUp = (): void => {
    if (this.pointerDownEmpty) {
      this.pointerDownEmpty = false;
      // a plain click on empty space clears the selection
      if (!this.dragging) this.deselect();
      return;
    }
    if (!this.connectParent) return;
    const parent = this.connectParent;
    const candidate = this.connectCandidate;
    this.highlightPart(parent, false, "part");
    if (candidate) this.highlightPart(candidate, false, "jointChild");
    this.connectParent = null;
    this.connectCandidate = null;

    if (this.connectMoved && candidate) {
      this.createJoint(parent, candidate);
    } else if (!this.connectMoved) {
      // plain click: edit the joint driving this part, if any
      const driver = this.doc.allJoints().find((j) => j.childPartId === parent);
      if (driver) {
        this.selectJoint(driver.id);
      } else {
        this.deselect();
      }
    }
    // reapply the editing highlights the gesture may have overwritten
    this.setHighlights(true);
  };

  private onKeyDown = (event: KeyboardEvent): void => {
    if (event.target instanceof HTMLElement && event.target.tagName !== "BODY")
      return;
    if (event.code === "Escape") {
      if (this.armedDof) {
        this.armDof(null);
      } else {
        this.deselect();
      }
    } else if (this.jointId && !this.armedDof && !this.dragging) {
      if (event.code === "KeyT" && this.pivotMode !== "translate") {
        this.pivotMode = "translate";
        this.rebuildGizmo();
      } else if (event.code === "KeyR" && this.pivotMode !== "rotate") {
        this.pivotMode = "rotate";
        this.rebuildGizmo();
      }
    }
  };

  private partAt(event: PointerEvent): string | undefined {
    const dom = this.viewport.renderer.domElement;
    const rect = dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const hitId = pickStrokeAtCursor(ndc, this.viewport, this.doc, this.strokeRenderer);
    return hitId ? this.doc.getStroke(hitId)?.partId : undefined;
  }

  // --- creation ---

  private createJoint(parentPartId: string, childPartId: string): void {
    const joints = this.doc.allJoints();
    if (partsInSubtree(joints, childPartId).has(parentPartId)) {
      alert("That connection would create a loop in the part hierarchy.");
      return;
    }
    this.ensureRestPose();

    const parentCenter = this.restCentroid(parentPartId);
    const childCenter = this.restCentroid(childPartId);
    if (!parentCenter || !childCenter) return;
    const dir = {
      x: childCenter.x - parentCenter.x,
      y: childCenter.y - parentCenter.y,
      z: childCenter.z - parentCenter.z,
    };
    const len = Math.hypot(dir.x, dir.y, dir.z);
    const axis: Vec3 =
      len > 1e-6
        ? { x: dir.x / len, y: dir.y / len, z: dir.z / len }
        : { x: 0, y: 1, z: 0 };

    const joint: Joint = {
      id: crypto.randomUUID(),
      name: `Joint ${this.doc.allJoints().length + 1}`,
      parentPartId,
      childPartId,
      pivot: childCenter,
      axis,
      dofs: lockedDofs(),
    };
    const replaced = joints.find((j) => j.childPartId === childPartId);
    this.undo.push(addJointCommand(this.doc, joint, replaced));
    this.selectJoint(joint.id);
  }

  /** A part's centroid at rest (explode offsets subtracted). */
  private restCentroid(partId: string): Vec3 | undefined {
    const strokes = this.doc.strokesInPart(partId);
    if (strokes.length === 0) return undefined;
    const c = { x: 0, y: 0, z: 0 };
    for (const s of strokes) {
      c.x += s.transform.position.x;
      c.y += s.transform.position.y;
      c.z += s.transform.position.z;
    }
    const offset = this.doc.getPart(partId)?.explodeOffset ?? { x: 0, y: 0, z: 0 };
    return {
      x: c.x / strokes.length - offset.x,
      y: c.y / strokes.length - offset.y,
      z: c.z / strokes.length - offset.z,
    };
  }

  // --- highlights ---

  private setHighlights(on: boolean): void {
    const joint = this.jointId ? this.doc.getJoint(this.jointId) : undefined;
    if (!joint) return;
    this.highlightPart(joint.parentPartId, on, "part");
    this.highlightPart(joint.childPartId, on, "jointChild");
  }

  private highlightPart(
    partId: string,
    on: boolean,
    kind: "part" | "jointChild",
  ): void {
    for (const stroke of this.doc.strokesInPart(partId)) {
      this.strokeRenderer.setHighlight(stroke.id, on, kind);
    }
  }

  // --- gizmo ---

  private rebuildGizmo(): void {
    this.destroyGizmo();
    this.buildGizmo();
  }

  private buildGizmo(): void {
    const joint = this.jointId ? this.doc.getJoint(this.jointId) : undefined;
    if (!joint) return;

    this.proxy = new THREE.Group();
    // while exploded, show the axis where the child currently sits
    const offset = this.doc.getPart(joint.childPartId)?.explodeOffset ?? {
      x: 0,
      y: 0,
      z: 0,
    };
    this.proxy.position.set(
      joint.pivot.x + offset.x,
      joint.pivot.y + offset.y,
      joint.pivot.z + offset.z,
    );
    this.proxy.quaternion.copy(
      posedJointFrame(joint, { x: 0, y: 0, z: 0, w: 1 }),
    );
    this.proxy.add(...axisVisual(this.axisLength(joint)));
    this.viewport.scene.add(this.proxy);

    this.rangeGroup = new THREE.Group();
    this.rangeGroup.position.copy(this.proxy.position);
    this.rangeGroup.quaternion.copy(this.proxy.quaternion);
    this.viewport.scene.add(this.rangeGroup);
    this.refreshRangeVisuals();

    this.controls = new TransformControls(
      this.viewport.camera,
      this.viewport.renderer.domElement,
    );
    this.controls.setSize(0.6);
    this.controls.setSpace("local");
    if (this.armedDof === null) {
      // placing the axis: T moves the pivot, R aims the direction
      this.controls.setMode(this.pivotMode);
      if (this.pivotMode === "rotate") {
        this.controls.showX = false; // roll about the axis is meaningless
        this.controls.showY = true;
        this.controls.showZ = true;
      }
    } else if (this.armedDof === "translation") {
      this.controls.setMode("translate");
      this.controls.showX = true;
      this.controls.showY = false;
      this.controls.showZ = false;
    } else {
      this.controls.setMode("rotate");
      this.controls.showX = this.armedDof === "twist";
      this.controls.showY = this.armedDof === "swingU";
      this.controls.showZ = this.armedDof === "swingV";
    }
    this.controls.addEventListener("change", () => this.viewport.invalidate());
    this.controls.addEventListener("objectChange", () => this.onGizmoChange());
    this.controls.addEventListener("dragging-changed", (event) => {
      if (event.value) {
        this.onDragStart();
      } else {
        this.onDragEnd();
      }
    });
    this.viewport.scene.add(this.controls.getHelper());
    this.controls.attach(this.proxy);
    this.viewport.invalidate();
  }

  private destroyGizmo(): void {
    if (this.controls) {
      this.controls.detach();
      this.viewport.scene.remove(this.controls.getHelper());
      this.controls.dispose();
      this.controls = undefined;
    }
    if (this.proxy) {
      this.viewport.scene.remove(this.proxy);
      this.proxy.traverse((object) => {
        if (object instanceof THREE.Line) {
          object.geometry.dispose();
          (object.material as THREE.Material).dispose();
        }
      });
      this.proxy = undefined;
    }
    if (this.rangeGroup) {
      this.clearRangeVisuals();
      this.viewport.scene.remove(this.rangeGroup);
      this.rangeGroup = undefined;
    }
  }

  private clearRangeVisuals(): void {
    if (!this.rangeGroup) return;
    for (const child of [...this.rangeGroup.children]) {
      this.rangeGroup.remove(child);
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }
  }

  /** Rebuild the range fills: while a DoF is armed only that DoF shows
   *  (`preview` overrides its stored range mid-demonstration); otherwise
   *  every unlocked DoF's committed range shows. */
  private refreshRangeVisuals(preview?: [number, number]): void {
    const joint = this.jointId ? this.doc.getJoint(this.jointId) : undefined;
    if (!this.rangeGroup || !joint) return;
    this.clearRangeVisuals();
    const length = this.axisLength(joint);
    const dofs = this.armedDof ? [this.armedDof] : JOINT_DOF_NAMES;
    for (const dof of dofs) {
      const range =
        preview && dof === this.armedDof ? preview : joint.dofs[dof].range;
      const mesh = dofRangeVisual(dof, range, length);
      if (mesh) this.rangeGroup.add(mesh);
    }
    this.viewport.invalidate();
  }

  /** Axis visual length, scaled to the child part's size. */
  private axisLength(joint: Joint): number {
    const strokes = this.doc.strokesInPart(joint.childPartId);
    let radius = 0;
    for (const s of strokes) {
      radius = Math.max(
        radius,
        Math.hypot(
          s.transform.position.x - joint.pivot.x,
          s.transform.position.y - joint.pivot.y,
          s.transform.position.z - joint.pivot.z,
        ),
      );
    }
    return Math.max(0.5, radius * 0.9);
  }

  // --- dragging (axis placement or range demonstration) ---

  private onDragStart(): void {
    if (!this.proxy) return;
    this.dragging = true;
    this.dragStartQuat.copy(this.proxy.quaternion);
    this.dragStartPos.copy(this.proxy.position);
    if (this.armedDof) {
      const joint = this.doc.getJoint(this.jointId!);
      if (!joint) return;
      this.baselineTransforms.clear();
      for (const partId of partsInSubtree(this.doc.allJoints(), joint.childPartId)) {
        for (const stroke of this.doc.strokesInPart(partId)) {
          this.baselineTransforms.set(stroke.id, cloneTransform(stroke.transform));
        }
      }
    }
  }

  private onGizmoChange(): void {
    if (!this.dragging || !this.proxy) return;
    if (!this.armedDof) {
      // axis placement commits at release; the range fills ride along
      if (this.rangeGroup) {
        this.rangeGroup.position.copy(this.proxy.position);
        this.rangeGroup.quaternion.copy(this.proxy.quaternion);
      }
      return;
    }
    const joint = this.doc.getJoint(this.jointId!);
    if (!joint) return;

    let raw: number;
    if (this.armedDof === "translation") {
      const axis = new THREE.Vector3(1, 0, 0).applyQuaternion(this.dragStartQuat);
      raw = new THREE.Vector3()
        .subVectors(this.proxy.position, this.dragStartPos)
        .dot(axis);
    } else {
      const ring = this.controls?.axis;
      if (!ring || RING_DOF[ring] !== this.armedDof) return;
      raw = twistAbout(this.dragStartQuat, this.proxy.quaternion, ring);
    }
    this.sessionMin = Math.min(this.sessionMin, raw);
    this.sessionMax = Math.max(this.sessionMax, raw);
    this.refreshRangeVisuals(this.sessionRange());
    this.applyDemoValue(joint, this.armedDof, raw);
  }

  /** The range this demonstration session would commit right now. */
  private sessionRange(): [number, number] {
    if (this.mirror) {
      const extreme = Math.max(-this.sessionMin, this.sessionMax);
      return [-extreme, extreme];
    }
    return [Math.min(0, this.sessionMin), Math.max(0, this.sessionMax)];
  }

  /** Move the child subtree live during a demonstration (unclamped). */
  private applyDemoValue(joint: Joint, dof: JointDofName, value: number): void {
    const joints = this.doc.allJoints();
    // the tool guarantees rest pose, so the baseline is all-zero values
    const before: JointValues = valuesOfPose(new Map());
    const after: JointValues = (j, d) =>
      j.id === joint.id && d === dof ? value : 0;
    const baseDeltas = computePartDeltas(joints, before);
    const nowDeltas = computePartDeltas(joints, after);
    for (const [strokeId, transform] of this.baselineTransforms) {
      const stroke = this.doc.getStroke(strokeId);
      if (!stroke?.partId) continue;
      const base = baseDeltas.get(stroke.partId) ?? identityRigid();
      const current = nowDeltas.get(stroke.partId) ?? identityRigid();
      const patch = rigidMultiply(current, rigidInvert(base));
      this.doc.setStrokeTransform(strokeId, applyRigidToTransform(patch, transform));
    }
    this.doc.setJointValue(joint.id, dof, value);
  }

  private onDragEnd(): void {
    if (!this.dragging) return;
    this.dragging = false;
    const joint = this.jointId ? this.doc.getJoint(this.jointId) : undefined;
    if (!joint || !this.proxy) return;

    if (this.armedDof) {
      // snap the part back to rest and commit the demonstrated range
      for (const [strokeId, transform] of this.baselineTransforms) {
        if (this.doc.getStroke(strokeId)) {
          this.doc.setStrokeTransform(strokeId, transform);
        }
      }
      this.doc.setJointValue(joint.id, this.armedDof, 0);
      this.baselineTransforms.clear();

      const range = this.sessionRange();
      const before = cloneJoint(joint);
      const after = cloneJoint(joint);
      after.dofs[this.armedDof].range = range;
      if (
        before.dofs[this.armedDof].range[0] !== range[0] ||
        before.dofs[this.armedDof].range[1] !== range[1]
      ) {
        this.undo.push(updateJointCommand(this.doc, before, after, "Set joint range"));
      }
      this.rebuildGizmo();
      return;
    }

    // axis placement: read the proxy back into pivot + axis
    const offset = this.doc.getPart(joint.childPartId)?.explodeOffset ?? {
      x: 0,
      y: 0,
      z: 0,
    };
    const pivot: Vec3 = {
      x: this.proxy.position.x - offset.x,
      y: this.proxy.position.y - offset.y,
      z: this.proxy.position.z - offset.z,
    };
    const axisV = new THREE.Vector3(1, 0, 0)
      .applyQuaternion(this.proxy.quaternion)
      .normalize();
    const before = cloneJoint(joint);
    const after = cloneJoint(joint);
    after.pivot = pivot;
    after.axis = { x: axisV.x, y: axisV.y, z: axisV.z };
    const changed =
      Math.hypot(
        after.pivot.x - before.pivot.x,
        after.pivot.y - before.pivot.y,
        after.pivot.z - before.pivot.z,
      ) > 1e-9 ||
      Math.hypot(
        after.axis.x - before.axis.x,
        after.axis.y - before.axis.y,
        after.axis.z - before.axis.z,
      ) > 1e-9;
    if (changed) {
      this.undo.push(updateJointCommand(this.doc, before, after, "Place joint axis"));
    }
    // re-orthonormalize the proxy's basis (U/V are derived from the axis)
    this.rebuildGizmo();
  }
}

/** Line visuals for the joint frame: a long two-sided axis (local X, red)
 *  and shorter U/V reference axes (green/blue), drawn through walls. */
function axisVisual(length: number): THREE.Object3D[] {
  const make = (to: Vec3, from: Vec3, color: number): THREE.Line => {
    const geometry = new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.Float32BufferAttribute([from.x, from.y, from.z, to.x, to.y, to.z], 3),
    );
    const line = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color, depthTest: false }),
    );
    line.renderOrder = 2;
    return line;
  };
  const side = length * 0.35;
  return [
    make({ x: length, y: 0, z: 0 }, { x: -length, y: 0, z: 0 }, 0xdd3333),
    make({ x: 0, y: side, z: 0 }, { x: 0, y: 0, z: 0 }, 0x33aa33),
    make({ x: 0, y: 0, z: side }, { x: 0, y: 0, z: 0 }, 0x3355dd),
  ];
}
