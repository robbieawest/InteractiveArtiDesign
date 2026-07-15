import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import type { SketchDocument } from "../core/SketchDocument";
import type { Stroke } from "../core/types";
import { computeVertexWidths } from "./ribbon";
import { buildFillGeometry } from "./StrokeRenderer";

// GLB export. Ribbons only exist through our shader, so exported files
// need real geometry: each stroke becomes a tube swept along its centerline
// with per-point radius (a port of Penzil's TubeGeometryWithVariableWidth,
// itself a variable-radius variant of three's TubeGeometry).

export async function exportGlb(doc: SketchDocument): Promise<Blob> {
  const scene = new THREE.Scene();

  for (const stroke of doc.allStrokes()) {
    const group = new THREE.Group();
    applyStrokeTransform(group, stroke);

    if (stroke.style.visible && stroke.points.length >= 6) {
      const tube = buildStrokeTube(stroke);
      if (tube) group.add(tube);
    }
    if (stroke.fill.visible && stroke.points.length >= 9) {
      group.add(
        new THREE.Mesh(
          buildFillGeometry(stroke.points),
          new THREE.MeshStandardMaterial({
            color: stroke.fill.color,
            side: THREE.DoubleSide,
          }),
        ),
      );
    }
    if (group.children.length > 0) scene.add(group);
  }

  const exporter = new GLTFExporter();
  const result = await exporter.parseAsync(scene, { binary: true });
  return new Blob([result as ArrayBuffer], { type: "model/gltf-binary" });
}

function applyStrokeTransform(object: THREE.Object3D, stroke: Stroke): void {
  const t = stroke.transform;
  object.position.set(t.position.x, t.position.y, t.position.z);
  object.quaternion.set(
    t.quaternion.x,
    t.quaternion.y,
    t.quaternion.z,
    t.quaternion.w,
  );
  object.scale.set(t.scale.x, t.scale.y, t.scale.z);
}

function buildStrokeTube(stroke: Stroke): THREE.Mesh | null {
  // Collapse consecutive duplicate points (a still pointer produces them);
  // zero-length segments break Frenet frame computation.
  const centers: THREE.Vector3[] = [];
  const radii: number[] = [];
  const widths = computeVertexWidths(stroke.pressure);
  for (let i = 0; i < stroke.points.length; i += 3) {
    const p = new THREE.Vector3(
      stroke.points[i],
      stroke.points[i + 1],
      stroke.points[i + 2],
    );
    if (centers.length === 0 || p.distanceToSquared(centers[centers.length - 1]) > 1e-12) {
      centers.push(p);
      // ribbon full width is lineWidth × width-multiplier; radius is half
      radii.push(0.5 * stroke.style.width * widths[i / 3]);
    }
  }
  if (centers.length < 2) return null;

  const path = new THREE.CatmullRomCurve3(centers);
  const geometry = buildVariableRadiusTube(path, centers.length - 1, radii, 8);
  return new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({
      color: stroke.style.color,
      flatShading: true,
      roughness: 1,
    }),
  );
}

/**
 * TubeGeometry with a radius per tubular segment instead of one constant.
 * `radii` must have at least `tubularSegments + 1` entries.
 */
export function buildVariableRadiusTube(
  path: THREE.Curve<THREE.Vector3>,
  tubularSegments: number,
  radii: number[],
  radialSegments: number,
): THREE.BufferGeometry {
  const frames = path.computeFrenetFrames(tubularSegments, false);

  const vertices: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const vertex = new THREE.Vector3();
  const normal = new THREE.Vector3();
  const point = new THREE.Vector3();

  for (let i = 0; i <= tubularSegments; i++) {
    path.getPointAt(i / tubularSegments, point);
    const N = frames.normals[i];
    const B = frames.binormals[i];
    const radius = radii[Math.min(i, radii.length - 1)];

    for (let j = 0; j <= radialSegments; j++) {
      const v = (j / radialSegments) * Math.PI * 2;
      const sin = Math.sin(v);
      const cos = -Math.cos(v);

      normal
        .set(
          cos * N.x + sin * B.x,
          cos * N.y + sin * B.y,
          cos * N.z + sin * B.z,
        )
        .normalize();
      normals.push(normal.x, normal.y, normal.z);

      vertex.copy(point).addScaledVector(normal, radius);
      vertices.push(vertex.x, vertex.y, vertex.z);

      uvs.push(i / tubularSegments, j / radialSegments);
    }
  }

  for (let j = 1; j <= tubularSegments; j++) {
    for (let i = 1; i <= radialSegments; i++) {
      const a = (radialSegments + 1) * (j - 1) + (i - 1);
      const b = (radialSegments + 1) * j + (i - 1);
      const c = (radialSegments + 1) * j + i;
      const d = (radialSegments + 1) * (j - 1) + i;
      indices.push(a, b, d, b, c, d);
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setIndex(indices);
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(vertices, 3),
  );
  geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  return geometry;
}
