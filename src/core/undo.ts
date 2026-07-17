import type { SketchDocument } from "./SketchDocument";
import type { Joint, JointDofName, Pose, Stroke, Transform, Vec3 } from "./types";
import { JOINT_DOF_NAMES, cloneJoint, cloneTransform } from "./types";
import { computeArticulationPatch, currentValues } from "./articulation";
import { applyRigidToTransform } from "./rigid";

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

/** Several commands as one undo step; undo runs in reverse order. */
export function compoundCommand(label: string, commands: Command[]): Command {
  return {
    label,
    execute: () => commands.forEach((c) => c.execute()),
    undo: () => [...commands].reverse().forEach((c) => c.undo()),
  };
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

export interface PartOffsetChange {
  partId: string;
  before?: Vec3;
  after?: Vec3;
}

/**
 * One explode-tool adjustment: new per-part offsets plus the stroke
 * transforms they produced, both recorded as absolute before/after state so
 * execute/undo are exact and idempotent (the tool applies the motion live
 * during the drag and pushes this at release, like `articulateCommand`).
 * The document's exploded flag is kept in sync with whether any offsets
 * remain.
 */
export function explodeStateCommand(
  doc: SketchDocument,
  label: string,
  strokeChanges: StrokeTransformChange[],
  offsetChanges: PartOffsetChange[],
): Command {
  const syncExploded = () =>
    doc.setExploded(doc.allParts().some((p) => p.explodeOffset !== undefined));
  return {
    label,
    execute: () => {
      strokeChanges.forEach((c) => doc.setStrokeTransform(c.id, c.after));
      offsetChanges.forEach((c) => doc.setPartExplodeOffset(c.partId, c.after));
      syncExploded();
    },
    undo: () => {
      strokeChanges.forEach((c) => doc.setStrokeTransform(c.id, c.before));
      offsetChanges.forEach((c) => doc.setPartExplodeOffset(c.partId, c.before));
      syncExploded();
    },
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

export interface JointValueChange {
  jointId: string;
  dof: JointDofName;
  before: number;
  after: number;
}

export interface StrokeTransformChange {
  id: string;
  before: Transform;
  after: Transform;
}

/**
 * One articulation step: new joint values plus the stroke transforms they
 * produced. Both sides are recorded as absolute before/after state, so
 * execute/undo are exact and idempotent — the tool applies the motion live
 * during the drag and pushes this at release (execute() then re-applies the
 * already-current state, like the select tool's transform command).
 */
export function articulateCommand(
  doc: SketchDocument,
  label: string,
  jointChanges: JointValueChange[],
  strokeChanges: StrokeTransformChange[],
): Command {
  return {
    label,
    execute: () => {
      strokeChanges.forEach((c) => doc.setStrokeTransform(c.id, c.after));
      jointChanges.forEach((c) => doc.setJointValue(c.jointId, c.dof, c.after));
    },
    undo: () => {
      strokeChanges.forEach((c) => doc.setStrokeTransform(c.id, c.before));
      jointChanges.forEach((c) => doc.setJointValue(c.jointId, c.dof, c.before));
    },
  };
}

/** Drive every joint back to its rest pose (all DoF values 0), moving
 *  member strokes by the exact inverse of the current articulation. */
export function resetArticulationCommand(doc: SketchDocument): Command {
  const joints = doc.allJoints();
  const jointChanges: JointValueChange[] = [];
  for (const joint of joints) {
    for (const dof of JOINT_DOF_NAMES) {
      const value = joint.dofs[dof].value;
      if (value !== 0) {
        jointChanges.push({ jointId: joint.id, dof, before: value, after: 0 });
      }
    }
  }

  const patch = computeArticulationPatch(joints, currentValues, () => 0);
  const strokeChanges: StrokeTransformChange[] = [];
  for (const [partId, rigid] of patch) {
    const isIdentity =
      Math.abs(rigid.q.w) > 1 - 1e-12 &&
      Math.hypot(rigid.t.x, rigid.t.y, rigid.t.z) < 1e-12;
    if (isIdentity) continue;
    for (const stroke of doc.strokesInPart(partId)) {
      strokeChanges.push({
        id: stroke.id,
        before: cloneTransform(stroke.transform),
        after: applyRigidToTransform(rigid, stroke.transform),
      });
    }
  }
  return articulateCommand(doc, "Reset articulation", jointChanges, strokeChanges);
}

/** Drive a single joint's DoFs to rest (0), leaving the rest of the pose
 *  alone — used before editing a joint's axis so its pivot stays truthful. */
export function restJointCommand(doc: SketchDocument, jointId: string): Command {
  const joints = doc.allJoints();
  const target = doc.getJoint(jointId);
  if (!target) throw new Error(`no such joint: ${jointId}`);

  const jointChanges: JointValueChange[] = [];
  for (const dof of JOINT_DOF_NAMES) {
    const value = target.dofs[dof].value;
    if (value !== 0) {
      jointChanges.push({ jointId, dof, before: value, after: 0 });
    }
  }
  const patch = computeArticulationPatch(joints, currentValues, (j, dof) =>
    j.id === jointId ? 0 : j.dofs[dof].value,
  );
  const strokeChanges: StrokeTransformChange[] = [];
  for (const [partId, rigid] of patch) {
    const isIdentity =
      Math.abs(rigid.q.w) > 1 - 1e-12 &&
      Math.hypot(rigid.t.x, rigid.t.y, rigid.t.z) < 1e-12;
    if (isIdentity) continue;
    for (const stroke of doc.strokesInPart(partId)) {
      strokeChanges.push({
        id: stroke.id,
        before: cloneTransform(stroke.transform),
        after: applyRigidToTransform(rigid, stroke.transform),
      });
    }
  }
  return articulateCommand(doc, "Rest joint", jointChanges, strokeChanges);
}

/** Create a joint; if `replaced` is given (the child's previous driver),
 *  it is swapped out in the same undo step. */
export function addJointCommand(
  doc: SketchDocument,
  joint: Joint,
  replaced?: Joint,
): Command {
  const added = cloneJoint(joint);
  const removed = replaced ? cloneJoint(replaced) : undefined;
  return {
    label: "Create joint",
    execute: () => {
      if (removed) doc.removeJoint(removed.id);
      doc.addJoint(cloneJoint(added));
    },
    undo: () => {
      doc.removeJoint(added.id);
      if (removed) doc.addJoint(cloneJoint(removed));
    },
  };
}

export function removeJointCommand(doc: SketchDocument, jointId: string): Command {
  const joint = doc.getJoint(jointId);
  if (!joint) throw new Error(`no such joint: ${jointId}`);
  const snapshot = cloneJoint(joint);
  return {
    label: "Delete joint",
    execute: () => doc.removeJoint(snapshot.id),
    undo: () => doc.addJoint(cloneJoint(snapshot)),
  };
}

/** Replace a joint's definition (axis, pivot, ranges…) with absolute
 *  before/after snapshots, so execute/undo are exact and idempotent. */
export function updateJointCommand(
  doc: SketchDocument,
  before: Joint,
  after: Joint,
  label = "Edit joint",
): Command {
  const beforeSnapshot = cloneJoint(before);
  const afterSnapshot = cloneJoint(after);
  return {
    label,
    execute: () => doc.replaceJoint(afterSnapshot),
    undo: () => doc.replaceJoint(beforeSnapshot),
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
