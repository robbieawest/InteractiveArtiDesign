import * as THREE from "three";
import type { JointDofName } from "../core/types";

/** Matches the gizmo handle colors: X ring/arrow drives slide+twist,
 *  Y drives swing U, Z drives swing V. */
const DOF_COLORS: Record<JointDofName, number> = {
  translation: 0xdd3333,
  twist: 0xdd3333,
  swingU: 0x33aa33,
  swingV: 0x3355dd,
};

const basis = (x: THREE.Vector3, y: THREE.Vector3, z: THREE.Vector3) =>
  new THREE.Matrix4().makeBasis(x, y, z);

/** CircleGeometry sweeps +X → +Y in its own XY plane; these bases map that
 *  plane into the joint-local plane each rotational DoF sweeps through
 *  (local X = axis, Y = u, Z = v; rotations are right-handed):
 *  twist spins u about the axis, the swings tilt the axis about u / v. */
const SECTOR_FRAMES: Partial<Record<JointDofName, THREE.Matrix4>> = {
  twist: basis(
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(1, 0, 0),
  ),
  swingU: basis(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0, -1),
    new THREE.Vector3(0, 1, 0),
  ),
  swingV: basis(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, 0, 1),
  ),
};

/**
 * A translucent fill describing one DoF's range, in joint-local coordinates
 * (origin at the pivot, X along the axis). Rotational DoFs get a pie sector
 * spanning [min, max] in their sweep plane; slide gets a bar along the axis
 * covering its travel. Returns null when the range is degenerate (locked).
 */
export function dofRangeVisual(
  dof: JointDofName,
  range: [number, number],
  length: number,
): THREE.Mesh | null {
  const span = range[1] - range[0];
  if (span < 1e-6) return null;
  const material = new THREE.MeshBasicMaterial({
    color: DOF_COLORS[dof],
    transparent: true,
    opacity: 0.22,
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
  });
  let mesh: THREE.Mesh;
  if (dof === "translation") {
    const thickness = length * 0.04;
    mesh = new THREE.Mesh(new THREE.BoxGeometry(span, thickness, thickness), material);
    mesh.position.x = (range[0] + range[1]) / 2;
  } else {
    const arc = Math.min(span, Math.PI * 2);
    const segments = Math.max(2, Math.ceil(arc * 16));
    const geometry = new THREE.CircleGeometry(length * 0.8, segments, range[0], arc);
    geometry.applyMatrix4(SECTOR_FRAMES[dof]!);
    mesh = new THREE.Mesh(geometry, material);
  }
  // between the joint lines (1) and the axis lines (2)
  mesh.renderOrder = 1.5;
  return mesh;
}
