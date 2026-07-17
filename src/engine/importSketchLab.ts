import * as THREE from "three";
import type { Joint, Part, Stroke, Vec3 } from "../core/types";
import { lockedDofs } from "../core/types";
import { identityTransform, newStrokeId } from "../core/types";
import { recenterPoints } from "../core/geometry";
import {
  isTubeVertexCount,
  reconstructTubeCenterline,
} from "../core/tubeCurve";

// Importer for SketchLab documents ("Rapid Design of Articulated Objects")
// exported to glTF via Sketchfab. The export preserves the whole document in
// the node tree:
//
//   Part_<id>_NoGeom3D          a part (may nest through joints)
//   Joint_<id>_NoGeom3D         a joint edge between two parts
//   SketchCurve_<id>__Projectable_Polyline_<hash>   one stroke, whose mesh
//       is a triangular tube (see core/tubeCurve.ts)
//
// Strokes are baked to absolute world space (the full ancestor chain,
// including Sketchfab's z-up→y-up root and its 0.01 cm→unit scale) and
// tagged with the nearest ancestor part — matching this app's flat storage
// model.
//
// Joints: a Joint_* node stores only a pivot (its world position) — the
// type, axis, and range are not explicit in the file. They ARE recoverable
// from the embedded animation, which exercises every joint: the delta
// rotations across keyframes give a hinge axis (revolute), translation
// deltas give a slide direction (prismatic), and the observed min/max
// excursion gives the range. Joints with no significant motion import as
// "fixed".

export interface SketchLabImport {
  strokes: Stroke[];
  parts: Part[];
  joints: Joint[];
}

interface GltfNode {
  name?: string;
  children?: number[];
  mesh?: number;
  matrix?: number[];
  translation?: number[];
  rotation?: number[];
  scale?: number[];
}

interface GltfAnimation {
  channels: {
    sampler: number;
    target: { node?: number; path: string };
  }[];
  samplers: { input: number; output: number }[];
}

interface Gltf {
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: GltfNode[];
  meshes?: { primitives: { attributes: Record<string, number> }[] }[];
  materials?: {
    pbrMetallicRoughness?: { baseColorFactor?: number[] };
  }[];
  animations?: GltfAnimation[];
  accessors?: {
    bufferView?: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
  }[];
  bufferViews?: {
    buffer: number;
    byteOffset?: number;
    byteStride?: number;
  }[];
  buffers?: { uri?: string; byteLength: number }[];
}

/** Split a .glb container into its JSON and binary chunks. */
export function parseGlb(data: ArrayBuffer): {
  json: unknown;
  bin?: ArrayBuffer;
} {
  const view = new DataView(data);
  if (view.getUint32(0, true) !== 0x46546c67 /* "glTF" */) {
    throw new Error("not a GLB file");
  }
  let offset = 12;
  let json: unknown;
  let bin: ArrayBuffer | undefined;
  while (offset < data.byteLength) {
    const length = view.getUint32(offset, true);
    const type = view.getUint32(offset + 4, true);
    const chunk = data.slice(offset + 8, offset + 8 + length);
    if (type === 0x4e4f534a /* "JSON" */) {
      json = JSON.parse(new TextDecoder().decode(chunk));
    } else if (type === 0x004e4942 /* "BIN" */) {
      bin = chunk;
    }
    offset += 8 + length;
  }
  if (json === undefined) throw new Error("GLB has no JSON chunk");
  return { json, bin };
}

function resolveBuffer(
  gltf: Gltf,
  index: number,
  bin: ArrayBuffer | undefined,
): ArrayBuffer {
  const buffer = gltf.buffers?.[index];
  if (!buffer) throw new Error(`glTF references missing buffer ${index}`);
  const uri = buffer.uri;
  if (uri === undefined) {
    // GLB-style internal buffer.
    if (!bin) throw new Error("glTF needs its binary chunk, none found");
    return bin;
  }
  if (uri.startsWith("data:")) {
    const base64 = uri.slice(uri.indexOf(",") + 1);
    const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
    return bytes.buffer;
  }
  // External .bin: the caller must supply it (e.g. selected alongside the
  // .gltf in the file picker).
  if (!bin) {
    throw new Error(
      `this .gltf stores its geometry in "${uri}" — select that file together with the .gltf`,
    );
  }
  return bin;
}

const COMPONENT_COUNTS: Record<string, number> = {
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
};

function readFloatAccessor(
  gltf: Gltf,
  accessorIndex: number,
  bin: ArrayBuffer | undefined,
  expectedType: string,
): Float32Array {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) throw new Error(`missing accessor ${accessorIndex}`);
  if (accessor.type !== expectedType || accessor.componentType !== 5126) {
    throw new Error(`accessor ${accessorIndex} is not float ${expectedType}`);
  }
  const components = COMPONENT_COUNTS[expectedType];
  const view = gltf.bufferViews?.[accessor.bufferView ?? -1];
  if (!view) throw new Error(`accessor ${accessorIndex} has no bufferView`);
  const buffer = resolveBuffer(gltf, view.buffer, bin);
  const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const packed = components * 4;
  const stride = view.byteStride ?? packed;

  const out = new Float32Array(accessor.count * components);
  if (stride === packed && start % 4 === 0) {
    out.set(new Float32Array(buffer, start, accessor.count * components));
  } else {
    const data = new DataView(buffer);
    for (let i = 0; i < accessor.count; i++) {
      for (let c = 0; c < components; c++) {
        out[i * components + c] = data.getFloat32(start + i * stride + c * 4, true);
      }
    }
  }
  return out;
}

function nodeLocalMatrix(node: GltfNode): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  if (node.matrix) {
    m.fromArray(node.matrix); // glTF matrices are column-major, like three's
  } else {
    const t = node.translation ?? [0, 0, 0];
    const r = node.rotation ?? [0, 0, 0, 1];
    const s = node.scale ?? [1, 1, 1];
    m.compose(
      new THREE.Vector3(t[0], t[1], t[2]),
      new THREE.Quaternion(r[0], r[1], r[2], r[3]),
      new THREE.Vector3(s[0], s[1], s[2]),
    );
  }
  return m;
}

/** glTF baseColorFactor is linear; CSS colors are sRGB. */
function linearToSrgbHex(rgb: number[]): string {
  const channel = (c: number) => {
    const srgb = c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    return Math.round(Math.min(Math.max(srgb, 0), 1) * 255)
      .toString(16)
      .padStart(2, "0");
  };
  return `#${channel(rgb[0] ?? 0)}${channel(rgb[1] ?? 0)}${channel(rgb[2] ?? 0)}`;
}

function median(values: Float32Array): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

// The ribbon shader renders pressure-0 strokes at 3× the style width
// (computeVertexWidths' base multiplier), so divide the recovered tube
// diameter by 3 to reproduce the original on-screen thickness.
const RIBBON_BASE_WIDTH_MULTIPLIER = 3;

/** A Joint_* node found during traversal, before the animation is mined for
 *  its type/axis/range. */
interface JointRecord {
  nodeIndex: number;
  parentPartId: string;
  childPartId?: string;
  /** World matrix of the joint node at rest. */
  world: THREE.Matrix4;
  /** World matrix of the joint's parent node (translation deltas from the
   *  animation live in that frame). */
  parentWorld: THREE.Matrix4;
  restRotation: THREE.Quaternion;
  restTranslation: THREE.Vector3;
}

export function importSketchLabGltf(
  json: unknown,
  bin?: ArrayBuffer,
): SketchLabImport {
  const gltf = json as Gltf;
  const nodes = gltf.nodes ?? [];
  const roots = gltf.scenes?.[gltf.scene ?? 0]?.nodes ?? [];

  const strokes: Stroke[] = [];
  const parts: Part[] = [];
  const partIds = new Set<string>();
  const jointRecords: JointRecord[] = [];

  const visit = (
    nodeIndex: number,
    parentMatrix: THREE.Matrix4,
    partId: string | undefined,
    pendingJoint: JointRecord | undefined,
  ): void => {
    const node = nodes[nodeIndex];
    if (!node) return;
    const world = new THREE.Matrix4().multiplyMatrices(
      parentMatrix,
      nodeLocalMatrix(node),
    );
    const name = node.name ?? "";

    if (name.startsWith("Part_")) {
      const sketchLabId = name.replace(/^Part_/, "").replace(/_NoGeom3D$/, "");
      partId =
        sketchLabId !== "" && !partIds.has(sketchLabId)
          ? sketchLabId
          : crypto.randomUUID();
      partIds.add(partId);
      parts.push({ id: partId, name: `Part ${parts.length + 1}` });
      // this part is what the enclosing joint drives
      if (pendingJoint && pendingJoint.childPartId === undefined) {
        pendingJoint.childPartId = partId;
      }
      pendingJoint = undefined;
    } else if (name.startsWith("Joint_") && partId !== undefined) {
      const t = node.translation ?? [0, 0, 0];
      const r = node.rotation ?? [0, 0, 0, 1];
      pendingJoint = {
        nodeIndex,
        parentPartId: partId,
        world,
        parentWorld: parentMatrix.clone(),
        restRotation: new THREE.Quaternion(r[0], r[1], r[2], r[3]),
        restTranslation: new THREE.Vector3(t[0], t[1], t[2]),
      };
      jointRecords.push(pendingJoint);
    }

    if (node.mesh !== undefined && name.startsWith("SketchCurve_")) {
      const stroke = meshToStroke(gltf, node.mesh, bin, world, partId);
      if (stroke) strokes.push(stroke);
    }

    for (const child of node.children ?? []) {
      visit(child, world, partId, pendingJoint);
    }
  };

  for (const root of roots) visit(root, new THREE.Matrix4(), undefined, undefined);

  if (strokes.length === 0) {
    throw new Error("no SketchLab strokes (SketchCurve_* nodes) in this file");
  }

  const joints: Joint[] = [];
  for (const record of jointRecords) {
    if (record.childPartId === undefined) continue; // dangling joint node
    joints.push(deriveJoint(gltf, bin, record, joints.length + 1));
  }
  return { strokes, parts, joints };
}

// Motion smaller than these spans is treated as noise, not a degree of
// freedom: ~3° of rotation, 0.01 world units of slide.
const MIN_ROTATION_SPAN = 0.05;
const MIN_TRANSLATION_SPAN = 0.01;

/** Mine the animation channels targeting this joint node for its type,
 *  axis, and range (see the file header). */
function deriveJoint(
  gltf: Gltf,
  bin: ArrayBuffer | undefined,
  record: JointRecord,
  ordinal: number,
): Joint {
  const pivotV = new THREE.Vector3();
  const worldRestQuat = new THREE.Quaternion();
  record.world.decompose(pivotV, worldRestQuat, new THREE.Vector3());
  const pivot: Vec3 = { x: pivotV.x, y: pivotV.y, z: pivotV.z };

  const base = {
    id: crypto.randomUUID(),
    name: `Joint ${ordinal}`,
    parentPartId: record.parentPartId,
    childPartId: record.childPartId!,
    pivot,
  };

  const rotation = analyzeRotation(gltf, bin, record, worldRestQuat);
  const rotates =
    rotation !== undefined &&
    rotation.range[1] - rotation.range[0] > MIN_ROTATION_SPAN;
  const translation = analyzeTranslation(gltf, bin, record);
  const slides =
    translation !== undefined &&
    translation.range[1] - translation.range[0] > MIN_TRANSLATION_SPAN;

  const dofs = lockedDofs();
  if (rotates) {
    dofs.twist = { range: rotation.range, value: 0 };
    // a screw has one axis: keep any translation that runs along it, too
    if (slides) {
      const along =
        translation.axis.x * rotation.axis.x +
        translation.axis.y * rotation.axis.y +
        translation.axis.z * rotation.axis.z;
      if (Math.abs(along) > 0.9) {
        const sign = Math.sign(along);
        dofs.translation = {
          range:
            sign >= 0
              ? translation.range
              : [-translation.range[1], -translation.range[0]],
          value: 0,
        };
      }
    }
    return { ...base, axis: rotation.axis, dofs };
  }
  if (slides) {
    dofs.translation = { range: translation.range, value: 0 };
    return { ...base, axis: translation.axis, dofs };
  }
  return { ...base, axis: { x: 1, y: 0, z: 0 }, dofs };
}

function samplerOutputsFor(
  gltf: Gltf,
  nodeIndex: number,
  path: "rotation" | "translation",
): number | undefined {
  for (const animation of gltf.animations ?? []) {
    for (const channel of animation.channels) {
      if (channel.target.node === nodeIndex && channel.target.path === path) {
        return animation.samplers[channel.sampler]?.output;
      }
    }
  }
  return undefined;
}

function analyzeRotation(
  gltf: Gltf,
  bin: ArrayBuffer | undefined,
  record: JointRecord,
  worldRestQuat: THREE.Quaternion,
): { axis: Vec3; range: [number, number] } | undefined {
  const output = samplerOutputsFor(gltf, record.nodeIndex, "rotation");
  if (output === undefined) return undefined;
  const keys = readFloatAccessor(gltf, output, bin, "VEC4");

  // Delta from rest, in the joint's local rest frame: Δ = r0⁻¹ ⊗ r(t).
  // (Node matrix = T·R(t), so R(t) = r0·Δ acts after the rest frame.)
  const restInv = record.restRotation.clone().invert();
  const deltas: { axis: THREE.Vector3; angle: number }[] = [];
  const q = new THREE.Quaternion();
  for (let i = 0; i * 4 < keys.length; i++) {
    q.set(keys[i * 4], keys[i * 4 + 1], keys[i * 4 + 2], keys[i * 4 + 3]);
    if (q.dot(record.restRotation) < 0) {
      q.set(-q.x, -q.y, -q.z, -q.w); // same rotation, near hemisphere
    }
    const delta = restInv.clone().multiply(q).normalize();
    const angle = 2 * Math.acos(Math.min(Math.max(delta.w, -1), 1));
    if (angle < 1e-3) continue;
    const s = Math.sqrt(1 - delta.w * delta.w);
    deltas.push({
      axis: new THREE.Vector3(delta.x / s, delta.y / s, delta.z / s),
      angle,
    });
  }
  if (deltas.length === 0) return undefined;

  // canonicalize axis sign against the largest excursion, then range the
  // signed angles (rest = 0 is always included)
  const reference = deltas.reduce((a, b) => (a.angle > b.angle ? a : b)).axis;
  let min = 0;
  let max = 0;
  for (const { axis, angle } of deltas) {
    const signed = axis.dot(reference) >= 0 ? angle : -angle;
    min = Math.min(min, signed);
    max = Math.max(max, signed);
  }
  const worldAxis = reference.clone().applyQuaternion(worldRestQuat).normalize();
  return {
    axis: { x: worldAxis.x, y: worldAxis.y, z: worldAxis.z },
    range: [min, max],
  };
}

function analyzeTranslation(
  gltf: Gltf,
  bin: ArrayBuffer | undefined,
  record: JointRecord,
): { axis: Vec3; range: [number, number] } | undefined {
  const output = samplerOutputsFor(gltf, record.nodeIndex, "translation");
  if (output === undefined) return undefined;
  const keys = readFloatAccessor(gltf, output, bin, "VEC3");

  // Deltas from rest, mapped through the parent world matrix's linear part
  // (rotation AND scale) so direction and distance come out in world units.
  const linear = new THREE.Matrix3().setFromMatrix4(record.parentWorld);
  const deltas: THREE.Vector3[] = [];
  for (let i = 0; i * 3 < keys.length; i++) {
    const v = new THREE.Vector3(keys[i * 3], keys[i * 3 + 1], keys[i * 3 + 2])
      .sub(record.restTranslation)
      .applyMatrix3(linear);
    if (v.lengthSq() < 1e-12) continue;
    deltas.push(v);
  }
  if (deltas.length === 0) return undefined;

  const direction = deltas
    .reduce((a, b) => (a.lengthSq() > b.lengthSq() ? a : b))
    .clone()
    .normalize();
  let min = 0;
  let max = 0;
  for (const delta of deltas) {
    const signed = delta.dot(direction);
    min = Math.min(min, signed);
    max = Math.max(max, signed);
  }
  return { axis: { x: direction.x, y: direction.y, z: direction.z }, range: [min, max] };
}

function meshToStroke(
  gltf: Gltf,
  meshIndex: number,
  bin: ArrayBuffer | undefined,
  world: THREE.Matrix4,
  partId: string | undefined,
): Stroke | undefined {
  const primitive = gltf.meshes?.[meshIndex]?.primitives[0];
  const positionAccessor = primitive?.attributes.POSITION;
  if (primitive === undefined || positionAccessor === undefined) {
    return undefined;
  }

  const positions = readFloatAccessor(gltf, positionAccessor, bin, "VEC3");
  if (!isTubeVertexCount(positions.length / 3)) return undefined;

  // World-transform the raw tube verts first so ring radii (→ stroke width)
  // come out in world units too.
  const v = new THREE.Vector3();
  for (let i = 0; i < positions.length; i += 3) {
    v.set(positions[i], positions[i + 1], positions[i + 2]).applyMatrix4(world);
    positions[i] = v.x;
    positions[i + 1] = v.y;
    positions[i + 2] = v.z;
  }

  const { points: worldPoints, radii } = reconstructTubeCenterline(positions);
  const { points, center } = recenterPoints(worldPoints);

  const materialIndex = (primitive as { material?: number }).material;
  const baseColor =
    gltf.materials?.[materialIndex ?? -1]?.pbrMetallicRoughness
      ?.baseColorFactor ?? [0, 0, 0, 1];
  const color = linearToSrgbHex(baseColor);

  // Median radius ignores the tapered ends where SketchLab baked pressure in.
  const width = (2 * median(radii)) / RIBBON_BASE_WIDTH_MULTIPLIER;

  const transform = identityTransform();
  transform.position = center;

  return {
    id: newStrokeId(),
    points,
    pressure: new Float32Array(points.length / 3),
    style: { visible: true, color, width },
    fill: { visible: false, color },
    transform,
    surface: { shape: "plane", transform: identityTransform() },
    ...(partId !== undefined && { partId }),
  };
}
