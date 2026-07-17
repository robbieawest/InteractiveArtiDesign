import type { Joint, JointDofName, Part, Pose, Stroke, Transform, Vec3 } from "./types";
import { cloneJoint, cloneTransform } from "./types";

export type DocumentEvent =
  | { type: "strokeAdded"; stroke: Stroke }
  | { type: "strokeRemoved"; stroke: Stroke }
  | { type: "strokeChanged"; stroke: Stroke }
  | { type: "partsChanged" }
  | { type: "posesChanged" }
  | { type: "jointsChanged" }
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
  private parts = new Map<string, Part>();
  private poses = new Map<string, Pose>();
  private joints = new Map<string, Joint>();
  /** True while parts are pushed apart; offsets live on the parts. */
  exploded = false;
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

  // --- parts ---

  getPart(id: string): Part | undefined {
    return this.parts.get(id);
  }

  allParts(): Part[] {
    return [...this.parts.values()];
  }

  addPart(part: Part): void {
    if (this.parts.has(part.id)) throw new Error(`duplicate part id: ${part.id}`);
    this.parts.set(part.id, part);
    this.emit({ type: "partsChanged" });
  }

  removePart(id: string): void {
    if (!this.parts.delete(id)) return;
    for (const stroke of this.strokes.values()) {
      if (stroke.partId === id) {
        delete stroke.partId;
        this.emit({ type: "strokeChanged", stroke });
      }
    }
    // joints are edges between parts; they can't outlive an endpoint
    let jointsRemoved = false;
    for (const joint of [...this.joints.values()]) {
      if (joint.parentPartId === id || joint.childPartId === id) {
        this.joints.delete(joint.id);
        jointsRemoved = true;
      }
    }
    if (jointsRemoved) this.emit({ type: "jointsChanged" });
    this.emit({ type: "partsChanged" });
  }

  setStrokePart(strokeId: string, partId: string | undefined): void {
    const stroke = this.strokes.get(strokeId);
    if (!stroke) throw new Error(`no such stroke: ${strokeId}`);
    if (partId !== undefined && !this.parts.has(partId)) {
      throw new Error(`no such part: ${partId}`);
    }
    if (partId === undefined) {
      delete stroke.partId;
    } else {
      stroke.partId = partId;
    }
    this.emit({ type: "strokeChanged", stroke });
    this.emit({ type: "partsChanged" });
  }

  strokesInPart(partId: string): Stroke[] {
    return this.allStrokes().filter((s) => s.partId === partId);
  }

  setPartExplodeOffset(partId: string, offset: Vec3 | undefined): void {
    const part = this.parts.get(partId);
    if (!part) throw new Error(`no such part: ${partId}`);
    if (offset === undefined) {
      delete part.explodeOffset;
    } else {
      part.explodeOffset = { ...offset };
    }
    this.emit({ type: "partsChanged" });
  }

  setExploded(exploded: boolean): void {
    this.exploded = exploded;
    this.emit({ type: "partsChanged" });
  }

  // --- joints ---

  getJoint(id: string): Joint | undefined {
    return this.joints.get(id);
  }

  allJoints(): Joint[] {
    return [...this.joints.values()];
  }

  addJoint(joint: Joint): void {
    if (this.joints.has(joint.id)) {
      throw new Error(`duplicate joint id: ${joint.id}`);
    }
    this.joints.set(joint.id, joint);
    this.emit({ type: "jointsChanged" });
  }

  removeJoint(id: string): void {
    if (this.joints.delete(id)) this.emit({ type: "jointsChanged" });
  }

  setJointValue(id: string, dof: JointDofName, value: number): void {
    const joint = this.joints.get(id);
    if (!joint) throw new Error(`no such joint: ${id}`);
    joint.dofs[dof].value = value;
    this.emit({ type: "jointsChanged" });
  }

  /** Overwrite a joint's definition (pivot, axis, ranges…) in place. */
  replaceJoint(joint: Joint): void {
    if (!this.joints.has(joint.id)) throw new Error(`no such joint: ${joint.id}`);
    this.joints.set(joint.id, cloneJoint(joint));
    this.emit({ type: "jointsChanged" });
  }

  // --- poses ---

  getPose(id: string): Pose | undefined {
    return this.poses.get(id);
  }

  allPoses(): Pose[] {
    return [...this.poses.values()];
  }

  addPose(pose: Pose): void {
    if (this.poses.has(pose.id)) throw new Error(`duplicate pose id: ${pose.id}`);
    this.poses.set(pose.id, pose);
    this.emit({ type: "posesChanged" });
  }

  removePose(id: string): void {
    if (this.poses.delete(id)) this.emit({ type: "posesChanged" });
  }

  clear(): void {
    this.strokes.clear();
    this.parts.clear();
    this.poses.clear();
    this.joints.clear();
    this.exploded = false;
    this.emit({ type: "cleared" });
  }

  private emit(event: DocumentEvent): void {
    for (const listener of this.listeners) listener(event);
  }
}
