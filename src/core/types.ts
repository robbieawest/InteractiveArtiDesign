// Plain-data types for the document model. No three.js here: the engine
// converts these to/from THREE.Vector3 etc. at its own boundary.

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Quat {
  x: number;
  y: number;
  z: number;
  w: number;
}

export interface Transform {
  position: Vec3;
  quaternion: Quat;
  scale: Vec3;
}

export function identityTransform(): Transform {
  return {
    position: { x: 0, y: 0, z: 0 },
    quaternion: { x: 0, y: 0, z: 0, w: 1 },
    scale: { x: 1, y: 1, z: 1 },
  };
}

export function cloneTransform(t: Transform): Transform {
  return {
    position: { ...t.position },
    quaternion: { ...t.quaternion },
    scale: { ...t.scale },
  };
}

/** The shape of the canvas surface a stroke was drawn on. */
export type SurfaceShape = "plane" | "cube" | "cylinder" | "sphere";

export interface StrokeStyle {
  visible: boolean;
  color: string; // CSS color
  /** Base line width in world units (matches legacy Penzil's lineWidth). */
  width: number;
}

export interface FillStyle {
  visible: boolean;
  color: string;
}

export interface Stroke {
  id: string;
  /** Centerline as xyz triplets, local to `transform`. */
  points: Float32Array;
  /** One weight per centerline point, 0..1. Modulates width along the line. */
  pressure: Float32Array;
  style: StrokeStyle;
  fill: FillStyle;
  transform: Transform;
  /** The surface the stroke was drawn on, so it can be restored later. */
  surface: { shape: SurfaceShape; transform: Transform };
  /** Reserved for segmentation (phase 6). */
  partId?: string;
}

export function newStrokeId(): string {
  return crypto.randomUUID();
}

/** A named group of strokes (via `stroke.partId`) that poses and explodes
 *  as a unit. */
export interface Part {
  id: string;
  name: string;
  /** While the document is exploded: the world offset this part was moved
   *  by, so collapsing (including strokes segmented in the meantime) is the
   *  exact reverse. */
  explodeOffset?: Vec3;
}

/**
 * The degrees of freedom of a screw joint, all relative to its single axis:
 * slide along it, twist about it, and swing about the two perpendicular
 * reference axes U/V (derived deterministically from the axis — see
 * `jointBasis`). A revolute joint is just a screw with only twist unlocked,
 * a prismatic one only translation, a ball joint has the swings too.
 */
export type JointDofName = "translation" | "twist" | "swingU" | "swingV";

export const JOINT_DOF_NAMES: readonly JointDofName[] = [
  "translation",
  "twist",
  "swingU",
  "swingV",
];

export interface JointDof {
  /** [min, max] joint value; radians for the rotational DoFs, world units
   *  for translation. Always contains 0, the rest value. [0, 0] = locked. */
  range: [number, number];
  /** Current value; 0 = rest. */
  value: number;
}

export function lockedDof(): JointDof {
  return { range: [0, 0], value: 0 };
}

export function lockedDofs(): Record<JointDofName, JointDof> {
  return {
    translation: lockedDof(),
    twist: lockedDof(),
    swingU: lockedDof(),
    swingV: lockedDof(),
  };
}

export function dofUnlocked(dof: JointDof): boolean {
  return dof.range[0] < 0 || dof.range[1] > 0;
}

/**
 * A joint edge between two parts: a screw — one axis (a line in Plücker
 * terms: pivot point + unit direction) with independent, individually
 * ranged DoFs. Pivot and axis are world-space *at rest pose* (all DoF
 * values zero); strokes themselves always store absolute transforms, so
 * posing works by patching member strokes with the delta between two joint
 * configurations — never by re-parenting coordinates.
 */
export interface Joint {
  id: string;
  name: string;
  /** Part on the root side of the edge. */
  parentPartId: string;
  /** Part this joint drives (and, transitively, everything below it). */
  childPartId: string;
  /** Pivot point (a point on the axis), world space at rest. */
  pivot: Vec3;
  /** Unit axis direction, world space at rest. */
  axis: Vec3;
  dofs: Record<JointDofName, JointDof>;
}

export function cloneJoint(joint: Joint): Joint {
  return {
    ...joint,
    pivot: { ...joint.pivot },
    axis: { ...joint.axis },
    dofs: {
      translation: { range: [...joint.dofs.translation.range], value: joint.dofs.translation.value },
      twist: { range: [...joint.dofs.twist.range], value: joint.dofs.twist.value },
      swingU: { range: [...joint.dofs.swingU.range], value: joint.dofs.swingU.value },
      swingV: { range: [...joint.dofs.swingV.range], value: joint.dofs.swingV.value },
    },
  };
}

/** True if any DoF is away from its rest value. */
export function jointPosed(joint: Joint): boolean {
  return JOINT_DOF_NAMES.some((dof) => joint.dofs[dof].value !== 0);
}

/** Human label for the joint's effective kind, derived from which DoFs are
 *  unlocked. */
export function jointKindLabel(joint: Joint): string {
  const slide = dofUnlocked(joint.dofs.translation);
  const twist = dofUnlocked(joint.dofs.twist);
  const swing = dofUnlocked(joint.dofs.swingU) || dofUnlocked(joint.dofs.swingV);
  if (swing) return "ball";
  if (slide && twist) return "cylindrical";
  if (twist) return "revolute";
  if (slide) return "prismatic";
  return "fixed";
}

/** A snapshot of stroke transforms, applied on demand from the poses panel. */
export interface Pose {
  id: string;
  name: string;
  /** Downscaled render of the viewport when the pose was saved (data URL). */
  thumbnail: string;
  transforms: Record<string, Transform>;
}
