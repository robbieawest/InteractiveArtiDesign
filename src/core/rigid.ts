import type { Quat, Transform, Vec3 } from "./types";

// Minimal rigid-transform (rotation + translation) math for articulation.
// Kept in core (no three.js) so FK/IK are unit-testable in isolation; the
// engine converts at its boundary as usual.

export interface RigidTransform {
  q: Quat;
  t: Vec3;
}

export function identityRigid(): RigidTransform {
  return { q: { x: 0, y: 0, z: 0, w: 1 }, t: { x: 0, y: 0, z: 0 } };
}

export function quatMultiply(a: Quat, b: Quat): Quat {
  return {
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

/** Inverse of a unit quaternion. */
export function quatConjugate(q: Quat): Quat {
  return { x: -q.x, y: -q.y, z: -q.z, w: q.w };
}

export function quatFromAxisAngle(axis: Vec3, angle: number): Quat {
  const half = angle / 2;
  const s = Math.sin(half);
  return {
    x: axis.x * s,
    y: axis.y * s,
    z: axis.z * s,
    w: Math.cos(half),
  };
}

export function rotateVec(q: Quat, v: Vec3): Vec3 {
  // v' = q * (v, 0) * q⁻¹, expanded
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  return {
    x: v.x + q.w * tx + (q.y * tz - q.z * ty),
    y: v.y + q.w * ty + (q.z * tx - q.x * tz),
    z: v.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

/** a ∘ b: apply b first, then a. */
export function rigidMultiply(
  a: RigidTransform,
  b: RigidTransform,
): RigidTransform {
  const rotated = rotateVec(a.q, b.t);
  return {
    q: quatMultiply(a.q, b.q),
    t: { x: rotated.x + a.t.x, y: rotated.y + a.t.y, z: rotated.z + a.t.z },
  };
}

export function rigidInvert(r: RigidTransform): RigidTransform {
  const qInv = quatConjugate(r.q);
  const t = rotateVec(qInv, r.t);
  return { q: qInv, t: { x: -t.x, y: -t.y, z: -t.z } };
}

export function rigidApplyPoint(r: RigidTransform, p: Vec3): Vec3 {
  const rotated = rotateVec(r.q, p);
  return {
    x: rotated.x + r.t.x,
    y: rotated.y + r.t.y,
    z: rotated.z + r.t.z,
  };
}

/** Rotation by `angle` about `axis` anchored at `pivot`:
 *  T(pivot) ∘ R(axis, angle) ∘ T(−pivot). */
export function rigidFromAxisAngleAt(
  axis: Vec3,
  angle: number,
  pivot: Vec3,
): RigidTransform {
  const q = quatFromAxisAngle(axis, angle);
  const rotatedPivot = rotateVec(q, pivot);
  return {
    q,
    t: {
      x: pivot.x - rotatedPivot.x,
      y: pivot.y - rotatedPivot.y,
      z: pivot.z - rotatedPivot.z,
    },
  };
}

export function rigidFromTranslation(t: Vec3): RigidTransform {
  return { q: { x: 0, y: 0, z: 0, w: 1 }, t: { ...t } };
}

/** Apply a rigid patch to a stroke transform (scale is untouched). */
export function applyRigidToTransform(
  r: RigidTransform,
  transform: Transform,
): Transform {
  return {
    position: rigidApplyPoint(r, transform.position),
    quaternion: quatMultiply(r.q, transform.quaternion),
    scale: { ...transform.scale },
  };
}
