import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { SketchDocument } from "../core/SketchDocument";
import type { UndoStack } from "../core/undo";
import {
  articulateCommand,
  type JointValueChange,
  type StrokeTransformChange,
} from "../core/undo";
import type { Joint, JointDofName, Transform, Vec3 } from "../core/types";
import { JOINT_DOF_NAMES, cloneTransform, dofUnlocked } from "../core/types";
import {
  computePartDeltas,
  jointBasis,
  jointChainTo,
  partsInSubtree,
  poseFrom,
  solveIK,
  valuesOfPose,
  type JointPose,
  type JointValues,
} from "../core/articulation";
import {
  applyRigidToTransform,
  identityRigid,
  rigidApplyPoint,
  rigidInvert,
  rigidMultiply,
  rotateVec,
} from "../core/rigid";
import type { Viewport } from "../engine/Viewport";
import type { StrokeRenderer } from "../engine/StrokeRenderer";
import { pickStrokeAtCursor } from "../engine/picking";

/** Which DoF each rotation-gizmo ring drives: the proxy's local X is the
 *  joint axis (twist), Y is the U reference axis, Z is V. */
const RING_DOF: Record<string, JointDofName> = {
  X: "twist",
  Y: "swingU",
  Z: "swingV",
};

/**
 * Click a part to articulate it along the screw joint that drives it: the
 * rotation gizmo shows one ring per unlocked rotational DoF (twist about
 * the axis, swings about the perpendicular U/V axes) and the translate
 * gizmo an arrow along the axis when the slide DoF is unlocked; T/R switch
 * between them when a joint has both. With IK enabled the gizmo becomes a
 * free translation handle and the whole joint chain from the root solves
 * backwards (CCD) to follow it.
 *
 * Strokes always hold absolute transforms; a drag re-derives every affected
 * stroke from its drag-start transform via Δ(values now) ∘ Δ(values at drag
 * start)⁻¹, so nothing accumulates across the drag, and one undoable
 * command records the whole motion at release.
 */
export class ArticulateTool {
  private partId: string | null = null;
  /** Joint driving the selected part (FK mode). */
  private joint: Joint | null = null;
  /** Root-first joint chain to the selected part (IK mode). */
  private chain: Joint[] = [];
  private ik = false;
  /** FK gizmo mode when the joint has both sliding and rotating DoFs. */
  private gizmoMode: "rotate" | "translate" = "rotate";

  private proxy?: THREE.Object3D;
  private controls?: TransformControls;
  private dragStartQuat = new THREE.Quaternion();
  private dragStartPos = new THREE.Vector3();
  /** Joint values / affected stroke transforms captured at drag start. */
  private baselineValues: JointPose = new Map();
  private baselineTransforms = new Map<string, Transform>();
  /** Where the IK grab point rests in world rest space. */
  private ikRestPoint: Vec3 = { x: 0, y: 0, z: 0 };
  private dragging = false;
  /** Notified with the id of the joint whose gizmo is showing (null when
   *  nothing is selected), so the Articulations panel can highlight it. */
  onJointSelected?: (jointId: string | null) => void;

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

  setIkMode(on: boolean): void {
    if (this.ik === on) return;
    this.ik = on;
    // re-resolve the current selection under the new mode
    const partId = this.partId;
    this.deselect();
    if (partId) this.selectPart(partId);
  }

  suppressGizmo(on: boolean): void {
    if (!this.controls) return;
    this.controls.enabled = !on;
    this.controls.getHelper().visible = !on;
    this.viewport.invalidate();
  }

  deselect(): void {
    this.setHighlights(false);
    this.partId = null;
    this.joint = null;
    this.chain = [];
    this.destroyGizmo();
    this.onJointSelected?.(null);
    this.viewport.invalidate();
  }

  /** Select a joint from the Articulations panel: shows the gizmo on the
   *  part it drives, exactly as if that part had been clicked. */
  selectJoint(jointId: string): void {
    if (this.doc.exploded) return; // rest pivots don't apply while exploded
    const joint = this.doc.getJoint(jointId);
    if (!joint) return;
    this.deselect();
    this.selectPart(joint.childPartId);
  }

  private onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.viewport.cameraOwnsPointer) return;
    if (this.controls && (this.controls.dragging || this.controls.axis)) {
      return; // gizmo click
    }
    if (this.doc.exploded) return; // rest pivots don't apply while exploded
    const dom = this.viewport.renderer.domElement;
    const rect = dom.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1,
    );
    const hitId = pickStrokeAtCursor(ndc, this.viewport, this.doc, this.strokeRenderer);
    const partId = hitId ? this.doc.getStroke(hitId)?.partId : undefined;
    if (!partId) {
      this.deselect();
      return;
    }
    this.deselect();
    this.selectPart(partId);
  };

  private onKeyDown = (event: KeyboardEvent): void => {
    if (event.target instanceof HTMLElement && event.target.tagName !== "BODY")
      return;
    if (event.code === "Escape") {
      this.deselect();
    } else if (!this.ik && this.joint && !this.dragging) {
      // T/R switch between the slide arrow and the rotation rings when the
      // joint has both kinds of DoF
      const rotates = this.jointRotates(this.joint);
      const slides = dofUnlocked(this.joint.dofs.translation);
      if (event.code === "KeyT" && slides && this.gizmoMode !== "translate") {
        this.gizmoMode = "translate";
        this.rebuildGizmo();
      } else if (event.code === "KeyR" && rotates && this.gizmoMode !== "rotate") {
        this.gizmoMode = "rotate";
        this.rebuildGizmo();
      }
    }
  };

  private jointRotates(joint: Joint): boolean {
    return (
      dofUnlocked(joint.dofs.twist) ||
      dofUnlocked(joint.dofs.swingU) ||
      dofUnlocked(joint.dofs.swingV)
    );
  }

  private selectPart(partId: string): void {
    const joints = this.doc.allJoints();
    const unlocked = (j: Joint) => JOINT_DOF_NAMES.some((d) => dofUnlocked(j.dofs[d]));
    if (this.ik) {
      this.chain = jointChainTo(joints, partId).filter(unlocked);
      if (this.chain.length === 0) return;
    } else {
      const joint = joints.find((j) => j.childPartId === partId);
      if (!joint || !unlocked(joint)) return;
      this.joint = joint;
      this.gizmoMode = this.jointRotates(joint) ? "rotate" : "translate";
    }
    this.partId = partId;
    this.setHighlights(true);
    this.buildGizmo();
    const driver = this.ik ? this.chain[this.chain.length - 1] : this.joint;
    this.onJointSelected?.(driver?.id ?? null);
  }

  private setHighlights(on: boolean): void {
    if (!this.partId) return;
    for (const stroke of this.doc.strokesInPart(this.partId)) {
      this.strokeRenderer.setHighlight(stroke.id, on, "part");
    }
  }

  // --- gizmo ---

  private rebuildGizmo(): void {
    this.destroyGizmo();
    this.buildGizmo();
  }

  private buildGizmo(): void {
    if (!this.partId) return;
    const deltas = computePartDeltas(this.doc.allJoints());

    this.proxy = new THREE.Object3D();
    if (this.ik) {
      const grab = this.partCentroid(this.partId);
      if (!grab) return;
      this.proxy.position.set(grab.x, grab.y, grab.z);
      const delta = deltas.get(this.partId) ?? identityRigid();
      this.ikRestPoint = rigidApplyPoint(rigidInvert(delta), grab);
    } else if (this.joint) {
      const delta = deltas.get(this.joint.childPartId) ?? identityRigid();
      const pivot = rigidApplyPoint(delta, this.joint.pivot);
      this.proxy.position.set(pivot.x, pivot.y, pivot.z);
      this.proxy.quaternion.copy(posedJointFrame(this.joint, delta.q));
    }
    this.viewport.scene.add(this.proxy);

    this.controls = new TransformControls(
      this.viewport.camera,
      this.viewport.renderer.domElement,
    );
    this.controls.setSize(0.6);
    if (this.ik) {
      this.controls.setMode("translate");
    } else {
      // constrain the gizmo to the joint's DoFs: the proxy's local X is the
      // joint axis, Y and Z the swing reference axes U and V
      const joint = this.joint!;
      this.controls.setMode(this.gizmoMode);
      this.controls.setSpace("local");
      if (this.gizmoMode === "rotate") {
        this.controls.showX = dofUnlocked(joint.dofs.twist);
        this.controls.showY = dofUnlocked(joint.dofs.swingU);
        this.controls.showZ = dofUnlocked(joint.dofs.swingV);
      } else {
        this.controls.showX = true;
        this.controls.showY = false;
        this.controls.showZ = false;
      }
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
      this.proxy = undefined;
    }
  }

  /** Mean of the part's (posed) stroke pivots — the IK grab point. */
  private partCentroid(partId: string): Vec3 | undefined {
    const strokes = this.doc.strokesInPart(partId);
    if (strokes.length === 0) return undefined;
    const c = { x: 0, y: 0, z: 0 };
    for (const s of strokes) {
      c.x += s.transform.position.x;
      c.y += s.transform.position.y;
      c.z += s.transform.position.z;
    }
    return {
      x: c.x / strokes.length,
      y: c.y / strokes.length,
      z: c.z / strokes.length,
    };
  }

  // --- dragging ---

  private onDragStart(): void {
    if (!this.proxy) return;
    this.dragging = true;
    this.dragStartQuat.copy(this.proxy.quaternion);
    this.dragStartPos.copy(this.proxy.position);

    this.baselineValues = poseFrom(this.doc.allJoints());

    // every stroke that can move: the subtree below the driven joint (FK)
    // or below the chain's topmost joint (IK)
    const topPartId = this.ik ? this.chain[0].childPartId : this.joint!.childPartId;
    const affected = partsInSubtree(this.doc.allJoints(), topPartId);
    this.baselineTransforms.clear();
    for (const partId of affected) {
      for (const stroke of this.doc.strokesInPart(partId)) {
        this.baselineTransforms.set(stroke.id, cloneTransform(stroke.transform));
      }
    }
  }

  private onGizmoChange(): void {
    if (!this.dragging || !this.proxy) return;
    if (this.ik) {
      const target = {
        x: this.proxy.position.x,
        y: this.proxy.position.y,
        z: this.proxy.position.z,
      };
      const pose = solveIK(
        this.doc.allJoints(),
        this.chain,
        poseFrom(this.doc.allJoints()),
        this.ikRestPoint,
        target,
      );
      this.applyPose(pose);
      return;
    }

    const joint = this.joint!;
    let dof: JointDofName;
    let raw: number;
    if (this.gizmoMode === "translate") {
      dof = "translation";
      const axis = new THREE.Vector3(1, 0, 0).applyQuaternion(this.dragStartQuat);
      raw = new THREE.Vector3()
        .subVectors(this.proxy.position, this.dragStartPos)
        .dot(axis);
    } else {
      const ring = this.controls?.axis;
      const mapped = ring ? RING_DOF[ring] : undefined;
      if (!mapped) return; // free/screen-space ring: not a joint DoF
      dof = mapped;
      raw = twistAbout(this.dragStartQuat, this.proxy.quaternion, ring!);
    }
    const baseValue = this.baselineValues.get(joint.id)?.[dof] ?? 0;
    const value = Math.min(
      Math.max(baseValue + raw, joint.dofs[dof].range[0]),
      joint.dofs[dof].range[1],
    );
    const pose: JointPose = new Map();
    pose.set(joint.id, {
      ...this.baselineValues.get(joint.id)!,
      [dof]: value,
    });
    this.applyPose(pose);
  }

  /** Re-derive every affected stroke from its drag-start transform under
   *  the new joint pose (exact — nothing accumulates within a drag). */
  private applyPose(pose: JointPose): void {
    const joints = this.doc.allJoints();
    const baseline = valuesOfPose(this.baselineValues);
    const now: JointValues = (j, d) => pose.get(j.id)?.[d] ?? baseline(j, d);

    const baseDeltas = computePartDeltas(joints, baseline);
    const nowDeltas = computePartDeltas(joints, now);

    for (const [strokeId, transform] of this.baselineTransforms) {
      const stroke = this.doc.getStroke(strokeId);
      if (!stroke?.partId) continue;
      const base = baseDeltas.get(stroke.partId) ?? identityRigid();
      const current = nowDeltas.get(stroke.partId) ?? identityRigid();
      const patch = rigidMultiply(current, rigidInvert(base));
      this.doc.setStrokeTransform(strokeId, applyRigidToTransform(patch, transform));
    }
    for (const [jointId, values] of pose) {
      const joint = this.doc.getJoint(jointId);
      if (!joint) continue;
      for (const dof of JOINT_DOF_NAMES) {
        if (joint.dofs[dof].value !== values[dof]) {
          this.doc.setJointValue(jointId, dof, values[dof]);
        }
      }
    }
  }

  private onDragEnd(): void {
    if (!this.dragging) return;
    this.dragging = false;

    const jointChanges: JointValueChange[] = [];
    for (const [jointId, before] of this.baselineValues) {
      const joint = this.doc.getJoint(jointId);
      if (!joint) continue;
      for (const dof of JOINT_DOF_NAMES) {
        const after = joint.dofs[dof].value;
        if (after !== before[dof]) {
          jointChanges.push({ jointId, dof, before: before[dof], after });
        }
      }
    }
    const strokeChanges: StrokeTransformChange[] = [];
    for (const [id, before] of this.baselineTransforms) {
      const stroke = this.doc.getStroke(id);
      if (stroke) {
        strokeChanges.push({
          id,
          before,
          after: cloneTransform(stroke.transform),
        });
      }
    }
    this.baselineValues.clear();
    this.baselineTransforms.clear();
    if (jointChanges.length === 0) return;

    this.undo.push(
      articulateCommand(
        this.doc,
        this.ik ? "Articulate (IK)" : "Articulate joint",
        jointChanges,
        strokeChanges,
      ),
    );

    // reseat the gizmo on the moved part (pivots ride along, the IK handle
    // follows the part)
    const partId = this.partId;
    this.destroyGizmo();
    this.partId = partId;
    if (partId) this.buildGizmo();
  }
}

/** Orientation whose local X/Y/Z are the joint's posed axis/U/V. */
export function posedJointFrame(joint: Joint, q: { x: number; y: number; z: number; w: number }): THREE.Quaternion {
  const { u, v } = jointBasis(joint.axis);
  const x = rotateVec(q, joint.axis);
  const y = rotateVec(q, u);
  const z = rotateVec(q, v);
  const m = new THREE.Matrix4().makeBasis(
    new THREE.Vector3(x.x, x.y, x.z),
    new THREE.Vector3(y.x, y.y, y.z),
    new THREE.Vector3(z.x, z.y, z.z),
  );
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

/** Signed rotation of `now` relative to `start` about the start-local
 *  X/Y/Z axis (swing-twist decomposition, twist part). */
export function twistAbout(
  start: THREE.Quaternion,
  now: THREE.Quaternion,
  axis: string,
): number {
  const rel = start.clone().invert().multiply(now);
  const component = axis === "X" ? rel.x : axis === "Y" ? rel.y : rel.z;
  return 2 * Math.atan2(rel.w < 0 ? -component : component, Math.abs(rel.w));
}
