import type { Joint, JointDofName, Vec3 } from "./types";
import { JOINT_DOF_NAMES, dofUnlocked } from "./types";
import {
  identityRigid,
  quatFromAxisAngle,
  quatMultiply,
  rigidApplyPoint,
  rigidFromAxisAngleAt,
  rigidFromTranslation,
  rigidInvert,
  rigidMultiply,
  rotateVec,
  type RigidTransform,
} from "./rigid";

// Forward/inverse kinematics over the joint edges. The part hierarchy is
// implied by the joints (parentPartId → childPartId); parts themselves stay
// a flat list and strokes keep absolute transforms. A part's "delta" is the
// rigid transform mapping its rest placement to its current placement, i.e.
// the composition of every ancestor joint's motion.

export type JointValues = (joint: Joint, dof: JointDofName) => number;

export const currentValues: JointValues = (joint, dof) => joint.dofs[dof].value;

/** A full assignment of DoF values per joint id (e.g. an IK solution). */
export type JointPose = Map<string, Record<JointDofName, number>>;

export function poseFrom(joints: Joint[]): JointPose {
  const pose: JointPose = new Map();
  for (const joint of joints) {
    pose.set(joint.id, {
      translation: joint.dofs.translation.value,
      twist: joint.dofs.twist.value,
      swingU: joint.dofs.swingU.value,
      swingV: joint.dofs.swingV.value,
    });
  }
  return pose;
}

export function valuesOfPose(pose: JointPose): JointValues {
  return (joint, dof) => pose.get(joint.id)?.[dof] ?? joint.dofs[dof].value;
}

/**
 * The two reference axes perpendicular to a joint's axis, for the swing
 * DoFs. Deterministic in the axis alone, so it never needs to be stored —
 * but it changes if the axis is edited.
 */
export function jointBasis(axis: Vec3): { u: Vec3; v: Vec3 } {
  const ref: Vec3 = Math.abs(axis.y) < 0.9 ? { x: 0, y: 1, z: 0 } : { x: 1, y: 0, z: 0 };
  const u = {
    x: ref.y * axis.z - ref.z * axis.y,
    y: ref.z * axis.x - ref.x * axis.z,
    z: ref.x * axis.y - ref.y * axis.x,
  };
  const uLen = Math.hypot(u.x, u.y, u.z) || 1;
  u.x /= uLen;
  u.y /= uLen;
  u.z /= uLen;
  const v = {
    x: axis.y * u.z - axis.z * u.y,
    y: axis.z * u.x - axis.x * u.z,
    z: axis.x * u.y - axis.y * u.x,
  };
  return { u, v };
}

/**
 * The motion of a single joint, in world rest space. Composition order
 * (innermost first): swing U, swing V, twist about the axis, then slide
 * along it — all rotations about the rest pivot.
 */
export function jointDelta(joint: Joint, values: JointValues): RigidTransform {
  const { u, v } = jointBasis(joint.axis);
  let rigid = identityRigid();
  const swingU = values(joint, "swingU");
  if (swingU !== 0) rigid = rigidFromAxisAngleAt(u, swingU, joint.pivot);
  const swingV = values(joint, "swingV");
  if (swingV !== 0) {
    rigid = rigidMultiply(rigidFromAxisAngleAt(v, swingV, joint.pivot), rigid);
  }
  const twist = values(joint, "twist");
  if (twist !== 0) {
    rigid = rigidMultiply(rigidFromAxisAngleAt(joint.axis, twist, joint.pivot), rigid);
  }
  const slide = values(joint, "translation");
  if (slide !== 0) {
    rigid = rigidMultiply(
      rigidFromTranslation({
        x: joint.axis.x * slide,
        y: joint.axis.y * slide,
        z: joint.axis.z * slide,
      }),
      rigid,
    );
  }
  return rigid;
}

/**
 * Rest→current delta per part, for every part that is some joint's child.
 * Parts not in the map (roots, unjointed parts) have identity delta.
 */
export function computePartDeltas(
  joints: Joint[],
  values: JointValues = currentValues,
): Map<string, RigidTransform> {
  const childIds = new Set(joints.map((j) => j.childPartId));
  const byParent = new Map<string, Joint[]>();
  for (const joint of joints) {
    const list = byParent.get(joint.parentPartId) ?? [];
    list.push(joint);
    byParent.set(joint.parentPartId, list);
  }

  const deltas = new Map<string, RigidTransform>();
  // roots: parents that are nobody's child
  const queue = joints.filter((j) => !childIds.has(j.parentPartId));
  const visited = new Set<string>();
  while (queue.length > 0) {
    const joint = queue.shift()!;
    if (visited.has(joint.childPartId)) continue; // duplicate driver or cycle
    visited.add(joint.childPartId);
    const parentDelta = deltas.get(joint.parentPartId) ?? identityRigid();
    deltas.set(
      joint.childPartId,
      rigidMultiply(parentDelta, jointDelta(joint, values)),
    );
    queue.push(...(byParent.get(joint.childPartId) ?? []));
  }
  return deltas;
}

/** All parts rigidly attached below `partId` (inclusive). */
export function partsInSubtree(joints: Joint[], partId: string): Set<string> {
  const result = new Set([partId]);
  const queue = [partId];
  while (queue.length > 0) {
    const current = queue.shift()!;
    for (const joint of joints) {
      if (joint.parentPartId === current && !result.has(joint.childPartId)) {
        result.add(joint.childPartId);
        queue.push(joint.childPartId);
      }
    }
  }
  return result;
}

/** The joints from the root down to (and including) the one driving
 *  `partId`, root-first. Empty if no joint drives the part. */
export function jointChainTo(joints: Joint[], partId: string): Joint[] {
  const chain: Joint[] = [];
  const seen = new Set<string>();
  let current = partId;
  for (;;) {
    const driver = joints.find((j) => j.childPartId === current);
    if (!driver || seen.has(driver.id)) break;
    seen.add(driver.id);
    chain.unshift(driver);
    current = driver.parentPartId;
  }
  return chain;
}

/** Per-part patch that carries strokes from configuration `before` to
 *  configuration `after`: Δ(after) ∘ Δ(before)⁻¹. Parts whose delta is
 *  unchanged still get an entry (an ~identity patch). */
export function computeArticulationPatch(
  joints: Joint[],
  before: JointValues,
  after: JointValues,
): Map<string, RigidTransform> {
  const beforeDeltas = computePartDeltas(joints, before);
  const afterDeltas = computePartDeltas(joints, after);
  const patch = new Map<string, RigidTransform>();
  for (const [partId, afterDelta] of afterDeltas) {
    const beforeDelta = beforeDeltas.get(partId) ?? identityRigid();
    patch.set(partId, rigidMultiply(afterDelta, rigidInvert(beforeDelta)));
  }
  return patch;
}

const clamp = (v: number, [min, max]: [number, number]) =>
  Math.min(Math.max(v, min), max);

/** The world direction of one DoF's motion under the current pose. Exact
 *  for the composition order in `jointDelta`: a DoF's axis is rotated by
 *  everything applied *outside* it (its own joint's outer components plus
 *  the parent chain), never by itself or inner components. */
function posedDofAxis(
  joint: Joint,
  dof: JointDofName,
  values: JointValues,
  parentDelta: RigidTransform,
): Vec3 {
  const { u, v } = jointBasis(joint.axis);
  let q = parentDelta.q;
  if (dof === "translation" || dof === "twist") {
    return rotateVec(q, joint.axis);
  }
  q = quatMultiply(q, quatFromAxisAngle(joint.axis, values(joint, "twist")));
  if (dof === "swingV") return rotateVec(q, v);
  q = quatMultiply(q, quatFromAxisAngle(v, values(joint, "swingV")));
  return rotateVec(q, u);
}

/**
 * Cyclic-coordinate-descent IK: adjust the chain's DoF values so the point
 * that rests at `restPoint` (world rest space, on the chain's end part)
 * lands as close as possible to `target`. Respects each DoF's range;
 * locked DoFs ([0,0] range) are skipped. Returns a new pose for the chain.
 */
export function solveIK(
  joints: Joint[],
  chain: Joint[],
  start: JointPose,
  restPoint: Vec3,
  target: Vec3,
  iterations = 16,
): JointPose {
  const pose: JointPose = new Map();
  for (const [id, values] of start) pose.set(id, { ...values });
  if (chain.length === 0) return pose;
  const endPartId = chain[chain.length - 1].childPartId;
  const valueOf = valuesOfPose(pose);
  const setValue = (joint: Joint, dof: JointDofName, value: number) => {
    const values = pose.get(joint.id) ?? {
      translation: joint.dofs.translation.value,
      twist: joint.dofs.twist.value,
      swingU: joint.dofs.swingU.value,
      swingV: joint.dofs.swingV.value,
    };
    values[dof] = value;
    pose.set(joint.id, values);
  };
  // full CCD steps; interactive drags warm-start from the previous frame so
  // a handful of iterations tracks the target closely
  const epsilon = 1e-4;

  for (let iteration = 0; iteration < iterations; iteration++) {
    for (let c = chain.length - 1; c >= 0; c--) {
      const joint = chain[c];
      for (const dof of JOINT_DOF_NAMES) {
        if (!dofUnlocked(joint.dofs[dof])) continue;

        const deltas = computePartDeltas(joints, valueOf);
        const end = rigidApplyPoint(
          deltas.get(endPartId) ?? identityRigid(),
          restPoint,
        );
        if (
          Math.hypot(end.x - target.x, end.y - target.y, end.z - target.z) <
          epsilon
        ) {
          return pose;
        }

        const parentDelta = deltas.get(joint.parentPartId) ?? identityRigid();
        const axis = posedDofAxis(joint, dof, valueOf, parentDelta);
        const value = valueOf(joint, dof);

        if (dof === "translation") {
          const step =
            (target.x - end.x) * axis.x +
            (target.y - end.y) * axis.y +
            (target.z - end.z) * axis.z;
          setValue(joint, dof, clamp(value + step, joint.dofs[dof].range));
          continue;
        }

        // rotational DoF: rotate about the posed DoF axis at the posed
        // pivot to swing the end effector toward the target, in the
        // plane ⊥ axis
        const childDelta = deltas.get(joint.childPartId) ?? identityRigid();
        const pivot = rigidApplyPoint(childDelta, joint.pivot);
        const project = (p: Vec3): Vec3 => {
          const d = { x: p.x - pivot.x, y: p.y - pivot.y, z: p.z - pivot.z };
          const along = d.x * axis.x + d.y * axis.y + d.z * axis.z;
          return {
            x: d.x - axis.x * along,
            y: d.y - axis.y * along,
            z: d.z - axis.z * along,
          };
        };
        const v1 = project(end);
        const v2 = project(target);
        const l1 = Math.hypot(v1.x, v1.y, v1.z);
        const l2 = Math.hypot(v2.x, v2.y, v2.z);
        if (l1 < 1e-6 || l2 < 1e-6) continue;
        const cross = {
          x: v1.y * v2.z - v1.z * v2.y,
          y: v1.z * v2.x - v1.x * v2.z,
          z: v1.x * v2.y - v1.y * v2.x,
        };
        const sin = cross.x * axis.x + cross.y * axis.y + cross.z * axis.z;
        const cos = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        const step = Math.atan2(sin, cos);
        setValue(joint, dof, clamp(value + step, joint.dofs[dof].range));
      }
    }
  }
  return pose;
}
