import * as THREE from "three";
import type { Part, Stroke } from "../core/types";
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
// model. Joints are counted but intentionally NOT imported: the articulation
// framework and its schema don't exist yet. Re-run the import once they do.

export interface SketchLabImport {
  strokes: Stroke[];
  parts: Part[];
  /** Joint nodes seen (and skipped) in the file. */
  jointCount: number;
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

interface Gltf {
  scene?: number;
  scenes?: { nodes?: number[] }[];
  nodes?: GltfNode[];
  meshes?: { primitives: { attributes: Record<string, number> }[] }[];
  materials?: {
    pbrMetallicRoughness?: { baseColorFactor?: number[] };
  }[];
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

function readVec3Accessor(
  gltf: Gltf,
  accessorIndex: number,
  bin: ArrayBuffer | undefined,
): Float32Array {
  const accessor = gltf.accessors?.[accessorIndex];
  if (!accessor) throw new Error(`missing accessor ${accessorIndex}`);
  if (accessor.type !== "VEC3" || accessor.componentType !== 5126) {
    throw new Error(`accessor ${accessorIndex} is not float VEC3`);
  }
  const view = gltf.bufferViews?.[accessor.bufferView ?? -1];
  if (!view) throw new Error(`accessor ${accessorIndex} has no bufferView`);
  const buffer = resolveBuffer(gltf, view.buffer, bin);
  const start = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const stride = view.byteStride ?? 12;

  const out = new Float32Array(accessor.count * 3);
  if (stride === 12) {
    out.set(new Float32Array(buffer, start, accessor.count * 3));
  } else {
    const data = new DataView(buffer);
    for (let i = 0; i < accessor.count; i++) {
      const o = start + i * stride;
      out[i * 3] = data.getFloat32(o, true);
      out[i * 3 + 1] = data.getFloat32(o + 4, true);
      out[i * 3 + 2] = data.getFloat32(o + 8, true);
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
  let jointCount = 0;

  const visit = (
    nodeIndex: number,
    parentMatrix: THREE.Matrix4,
    partId: string | undefined,
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
    } else if (name.startsWith("Joint_")) {
      jointCount++;
    }

    if (node.mesh !== undefined && name.startsWith("SketchCurve_")) {
      const stroke = meshToStroke(gltf, node.mesh, bin, world, partId);
      if (stroke) strokes.push(stroke);
    }

    for (const child of node.children ?? []) visit(child, world, partId);
  };

  for (const root of roots) visit(root, new THREE.Matrix4(), undefined);

  if (strokes.length === 0) {
    throw new Error("no SketchLab strokes (SketchCurve_* nodes) in this file");
  }
  return { strokes, parts, jointCount };
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

  const positions = readVec3Accessor(gltf, positionAccessor, bin);
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
