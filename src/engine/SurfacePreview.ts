import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { Viewport } from "./Viewport";
import type { Joint } from "../core/types";
import {
  computeArticulationPatch,
  currentValues,
  poseFrom,
  valuesOfPose,
  type JointPose,
} from "../core/articulation";
import { identityRigid } from "../core/rigid";
import { getSurfaceMatcap, injectFresnelRim } from "./surfaceMatcap";
import type { SurfacingSketch } from "../surfacing/client";

/** Appearance of the surface overlay, editable from the Surfacer panel. */
export interface SurfaceStyle {
  color: string;
  opacity: number;
}

const DEFAULT_STYLE: SurfaceStyle = { color: "#ffaa3c", opacity: 0.55 };

/** One mesh in the overlay, prepared for rigid skinning. Positions and normals
 *  are cached in world space at bind time; each vertex indexes into this mesh's
 *  small `localParts` table (the distinct parts its vertices attach to), so a
 *  repose only builds one matrix per part and then runs a tight per-vertex
 *  loop. The world⁻¹ matrices map the reposed vertices back into the mesh's own
 *  local space so node transforms still hold. */
interface SkinnedMesh {
  position: THREE.BufferAttribute;
  normal: THREE.BufferAttribute | null;
  /** World-space rest positions / normals, 3 floats per vertex. */
  restPos: Float32Array;
  restNorm: Float32Array | null;
  /** Per vertex: index into `localParts`. */
  boundIndex: Int32Array;
  /** Distinct parts this mesh's vertices bind to (null = static). */
  localParts: (string | null)[];
  worldInverse: THREE.Matrix4;
  worldInverseNormal: THREE.Matrix3;
}

/** Displays the mesh returned by a surfacing job as a scene overlay. The
 *  mesh is derived output, not sketch data: it never enters the document or
 *  the undo stack, and re-running a job simply replaces it.
 *
 *  The overlay is skinned to the joint rig: after `bindSkin`, each vertex
 *  follows the part it sits closest to, so articulating the model deforms the
 *  surface exactly as it deforms the strokes (rigid, one part per vertex). */
export class SurfacePreview {
  private readonly group = new THREE.Group();
  private readonly loader = new GLTFLoader();
  private style: SurfaceStyle = { ...DEFAULT_STYLE };
  /** Our own matcap material, shared across the overlay's meshes and driven by
   *  `style`; replaces whatever the glb shipped so shading + colour are ours. */
  private material: THREE.MeshMatcapMaterial | null = null;
  /** Pieces published by an in-flight job, by name, so a re-published piece
   *  replaces its older self instead of stacking on top of it. */
  private readonly partials = new Map<string, THREE.Object3D>();

  // skinning state
  private skinned: SkinnedMesh[] = [];
  /** Joint values at bind time (the pose the surface was produced in). */
  private bindPose: JointPose = new Map();
  /** Signature of the last pose applied, to skip redundant reposes. */
  private lastPoseKey = "";

  constructor(private readonly viewport: Viewport) {
    this.viewport.scene.add(this.group);
  }

  get hasContent(): boolean {
    return this.group.children.length > 0;
  }

  async show(glb: ArrayBuffer): Promise<void> {
    const gltf = await this.loader.parseAsync(glb, "");
    this.clear();
    this.adopt(gltf.scene);
  }

  /** Add (or replace) one piece published by a job still in flight, so the
   *  overlay fills in as the adapter finishes parts instead of appearing all
   *  at once at the end. Pieces are keyed by the adapter's name for them: the
   *  same name again is a newer snapshot of that piece and supersedes it.
   *
   *  Partial content is unskinned — the final `show` + `bindSkin` is what
   *  seats the overlay on the rig. */
  async showPartial(name: string, glb: ArrayBuffer): Promise<void> {
    const gltf = await this.loader.parseAsync(glb, "");
    const previous = this.partials.get(name);
    if (previous) {
      // keep the shared material: the other pieces are still using it
      disposeSubtree(previous, this.material);
      this.group.remove(previous);
    }
    // whatever was bound belonged to geometry we are now replacing
    this.dropSkin();
    this.partials.set(name, gltf.scene);
    this.adopt(gltf.scene);
  }

  /** Put a freshly parsed glb scene into the overlay under our own material. */
  private adopt(scene: THREE.Object3D): void {
    this.group.add(scene);
    this.group.visible = true; // a fresh surface always shows
    // take over shading: replace the glb's materials (often with baked vertex
    // colours) with one matcap material we fully control
    const material = (this.material ??= buildSurfaceMaterial());
    scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        // surfacing meshes (VNS marching cubes, concatenated parts) export
        // with positions only — no normals, which matcap shading needs. Add
        // them so the surface actually shades (and so bindSkin can cache them)
        if (!obj.geometry.getAttribute("normal")) {
          obj.geometry.computeVertexNormals();
        }
        for (const m of asArray(obj.material)) m.dispose();
        obj.material = material;
      }
    });
    this.applyStyle();
    this.viewport.invalidate();
  }

  private dropSkin(): void {
    this.skinned = [];
    this.bindPose = new Map();
    this.lastPoseKey = "";
  }

  /** Surface color and opacity; applied live to the current overlay. */
  setStyle(style: Partial<SurfaceStyle>): void {
    this.style = { ...this.style, ...style };
    this.applyStyle();
    this.viewport.invalidate();
  }

  getStyle(): SurfaceStyle {
    return { ...this.style };
  }

  setVisible(visible: boolean): void {
    this.group.visible = visible;
    this.viewport.invalidate();
  }

  private applyStyle(): void {
    const material = this.material;
    if (!material) return;
    material.color.set(this.style.color);
    material.opacity = this.style.opacity;
    material.transparent = this.style.opacity < 1;
    // a translucent double-sided surface sorts badly with depth writes on
    material.depthWrite = this.style.opacity >= 1;
  }

  /** Bind the current overlay to the rig: snapshot each vertex's world rest
   *  position and attach it to the nearest part (by the sketch's world-space
   *  stroke points). `joints` records the pose the surface was made in, so a
   *  surface produced while posed still skins correctly. */
  bindSkin(sketch: SurfacingSketch, joints: Joint[]): void {
    this.skinned = [];
    this.bindPose = poseFrom(joints);
    this.lastPoseKey = "";

    // cloud of (world point → part id) from every stroke, indexed by a
    // spatial hash so each vertex's nearest-sample lookup is ~O(1) instead of
    // scanning the whole cloud (which made dense multi-part surfaces crawl)
    const grid = new SampleGrid(sketch);

    this.group.updateWorldMatrix(true, true);
    this.group.traverse((obj) => {
      if (!(obj instanceof THREE.Mesh)) return;
      const position = obj.geometry.getAttribute("position");
      if (!(position instanceof THREE.BufferAttribute)) return;
      // skinning writes positions/normals directly, so bounding volumes go
      // stale during a drag — skip culling instead of recomputing them
      obj.frustumCulled = false;
      const normalRaw = obj.geometry.getAttribute("normal");
      const normal =
        normalRaw instanceof THREE.BufferAttribute ? normalRaw : null;

      const count = position.count;
      const restPos = new Float32Array(count * 3);
      const restNorm = normal ? new Float32Array(count * 3) : null;
      const boundIndex = new Int32Array(count);
      const localParts: (string | null)[] = [];
      const partLookup = new Map<string | null, number>();

      const world = obj.matrixWorld;
      const worldInverse = world.clone().invert();
      const worldNormal = new THREE.Matrix3().getNormalMatrix(world);
      const p = new THREE.Vector3();
      const n = new THREE.Vector3();
      for (let i = 0; i < count; i++) {
        p.fromBufferAttribute(position, i).applyMatrix4(world);
        restPos[i * 3] = p.x;
        restPos[i * 3 + 1] = p.y;
        restPos[i * 3 + 2] = p.z;
        if (restNorm && normal) {
          n.fromBufferAttribute(normal, i).applyMatrix3(worldNormal).normalize();
          restNorm[i * 3] = n.x;
          restNorm[i * 3 + 1] = n.y;
          restNorm[i * 3 + 2] = n.z;
        }
        const part = grid.nearestPart(p.x, p.y, p.z);
        let idx = partLookup.get(part);
        if (idx === undefined) {
          idx = localParts.push(part) - 1;
          partLookup.set(part, idx);
        }
        boundIndex[i] = idx;
      }
      this.skinned.push({
        position,
        normal,
        restPos,
        restNorm,
        boundIndex,
        localParts,
        worldInverse,
        worldInverseNormal: new THREE.Matrix3().getNormalMatrix(worldInverse),
      });
    });

    // seat it at whatever pose the joints are in now
    this.repose(joints);
  }

  /** Deform the overlay to the joints' current pose. Cheap and idempotent:
   *  a no-op when nothing is bound or the pose is unchanged. Normals are
   *  transformed rigidly (per-part rotation) rather than recomputed from the
   *  faces, so a drag stays interactive on dense meshes. */
  repose(joints: Joint[]): void {
    if (this.skinned.length === 0) return;
    const key = poseKey(joints);
    if (key === this.lastPoseKey) return;
    this.lastPoseKey = key;

    const patch = computeArticulationPatch(
      joints,
      valuesOfPose(this.bindPose),
      currentValues,
    );
    const identity = identityRigid();
    const t = new THREE.Vector3();
    const q = new THREE.Quaternion();
    const one = new THREE.Vector3(1, 1, 1);
    const rigidMat = new THREE.Matrix4();
    const rotMat = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const n = new THREE.Vector3();

    for (const mesh of this.skinned) {
      // one position matrix (world⁻¹ ∘ rigid) and normal matrix per part
      const posMats = mesh.localParts.map((pid) => {
        const r = (pid && patch.get(pid)) || identity;
        rigidMat.compose(
          t.set(r.t.x, r.t.y, r.t.z),
          q.set(r.q.x, r.q.y, r.q.z, r.q.w),
          one,
        );
        return new THREE.Matrix4().multiplyMatrices(mesh.worldInverse, rigidMat);
      });
      const normMats = mesh.restNorm
        ? mesh.localParts.map((pid) => {
            const r = (pid && patch.get(pid)) || identity;
            rotMat.makeRotationFromQuaternion(
              q.set(r.q.x, r.q.y, r.q.z, r.q.w),
            );
            return new THREE.Matrix3()
              .setFromMatrix4(rotMat)
              .premultiply(mesh.worldInverseNormal);
          })
        : null;

      const position = mesh.position;
      const restPos = mesh.restPos;
      const boundIndex = mesh.boundIndex;
      for (let i = 0; i < position.count; i++) {
        const idx = boundIndex[i];
        p.set(restPos[i * 3], restPos[i * 3 + 1], restPos[i * 3 + 2]);
        p.applyMatrix4(posMats[idx]);
        position.setXYZ(i, p.x, p.y, p.z);
      }
      position.needsUpdate = true;

      if (normMats && mesh.restNorm && mesh.normal) {
        const restNorm = mesh.restNorm;
        const normal = mesh.normal;
        for (let i = 0; i < normal.count; i++) {
          n.set(restNorm[i * 3], restNorm[i * 3 + 1], restNorm[i * 3 + 2]);
          n.applyMatrix3(normMats[boundIndex[i]]).normalize();
          normal.setXYZ(i, n.x, n.y, n.z);
        }
        normal.needsUpdate = true;
      }
    }
    this.viewport.invalidate();
  }

  clear(): void {
    this.dropSkin();
    this.partials.clear();
    for (const child of [...this.group.children]) {
      disposeSubtree(child);
      this.group.remove(child);
    }
    this.material?.dispose();
    this.material = null;
    this.viewport.invalidate();
  }

  dispose(): void {
    this.clear();
    this.viewport.scene.remove(this.group);
  }
}

/** A uniform spatial hash over the sketch's stroke points, so binding each
 *  surface vertex to its nearest part is ~O(1) rather than a scan of the whole
 *  point cloud. Cells hold packed sample coords + a part index; the part
 *  strings are interned so per-vertex lookups compare integers. */
class SampleGrid {
  private readonly cells = new Map<number, number[]>();
  private readonly px: number[] = [];
  private readonly py: number[] = [];
  private readonly pz: number[] = [];
  private readonly partOf: number[] = [];
  private readonly parts: (string | null)[] = [];
  private readonly cell: number;
  private readonly minX: number;
  private readonly minY: number;
  private readonly minZ: number;
  private readonly count: number;

  constructor(sketch: SurfacingSketch) {
    const partIndex = new Map<string | null, number>();
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (const stroke of sketch.strokes) {
      let pi = partIndex.get(stroke.partId);
      if (pi === undefined) {
        pi = this.parts.push(stroke.partId) - 1;
        partIndex.set(stroke.partId, pi);
      }
      for (const [x, y, z] of stroke.points) {
        this.px.push(x);
        this.py.push(y);
        this.pz.push(z);
        this.partOf.push(pi);
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
      }
    }
    this.count = this.px.length;
    this.minX = minX;
    this.minY = minY;
    this.minZ = minZ;
    // ~1 sample per cell on average: cell = diag / cbrt(n)
    const diag =
      this.count > 0
        ? Math.hypot(maxX - minX, maxY - minY, maxZ - minZ)
        : 0;
    this.cell = diag > 0 ? diag / Math.cbrt(this.count) || 1 : 1;
    for (let i = 0; i < this.count; i++) {
      const key = this.hash(this.px[i], this.py[i], this.pz[i]);
      const bucket = this.cells.get(key);
      if (bucket) bucket.push(i);
      else this.cells.set(key, [i]);
    }
  }

  private cellIndex(v: number, min: number): number {
    return Math.floor((v - min) / this.cell);
  }

  // pack three smallish cell indices into one number key (grids are shallow)
  private key(ix: number, iy: number, iz: number): number {
    return (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791);
  }

  private hash(x: number, y: number, z: number): number {
    return this.key(
      this.cellIndex(x, this.minX),
      this.cellIndex(y, this.minY),
      this.cellIndex(z, this.minZ),
    );
  }

  /** Part id of the sample nearest (x,y,z); null if empty or that sample is
   *  itself unassigned. Searches outward by cell shells, stopping one shell
   *  past the first hit (enough to beat any diagonal near-miss). */
  nearestPart(x: number, y: number, z: number): string | null {
    if (this.count === 0) return null;
    const cx = this.cellIndex(x, this.minX);
    const cy = this.cellIndex(y, this.minY);
    const cz = this.cellIndex(z, this.minZ);
    let best = Infinity;
    let bestPart: number | null = null;
    let hitRadius = -1;
    const maxRadius = 64;
    for (let r = 0; r <= maxRadius; r++) {
      for (let ix = cx - r; ix <= cx + r; ix++) {
        for (let iy = cy - r; iy <= cy + r; iy++) {
          for (let iz = cz - r; iz <= cz + r; iz++) {
            // shell only: skip cells already covered by smaller radii
            const onShell =
              Math.max(
                Math.abs(ix - cx),
                Math.abs(iy - cy),
                Math.abs(iz - cz),
              ) === r;
            if (!onShell) continue;
            const bucket = this.cells.get(this.key(ix, iy, iz));
            if (!bucket) continue;
            for (const i of bucket) {
              const dx = x - this.px[i];
              const dy = y - this.py[i];
              const dz = z - this.pz[i];
              const d = dx * dx + dy * dy + dz * dz;
              if (d < best) {
                best = d;
                bestPart = this.partOf[i];
              }
            }
          }
        }
      }
      if (bestPart !== null) {
        if (hitRadius === -1) hitRadius = r;
        if (r >= hitRadius + 1) break; // one extra shell, then done
      }
    }
    return bestPart === null ? null : this.parts[bestPart];
  }
}

/** A cheap fingerprint of the joints' current DoF values, so repose can skip
 *  work when the pose hasn't actually moved. */
function poseKey(joints: Joint[]): string {
  let key = "";
  for (const j of joints) {
    const d = j.dofs;
    key += `${d.translation.value},${d.twist.value},${d.swingU.value},${d.swingV.value};`;
  }
  return key;
}

/** Free the GPU resources of a subtree leaving the overlay. `keep` is the
 *  material shared with whatever stays behind, so it survives. */
function disposeSubtree(root: THREE.Object3D, keep?: THREE.Material | null): void {
  root.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      obj.geometry.dispose();
      for (const material of asArray(obj.material)) {
        if (material !== keep) material.dispose();
      }
    }
  });
}

function asArray(m: THREE.Material | THREE.Material[]): THREE.Material[] {
  return Array.isArray(m) ? m : [m];
}

/** A matcap material for the surface: reads form via a baked sphere image
 *  (no scene lights, stable under orbit), tinted by the style colour, with a
 *  fresnel rim injected so the silhouette reads even at low opacity. */
function buildSurfaceMaterial(): THREE.MeshMatcapMaterial {
  const material = new THREE.MeshMatcapMaterial({
    matcap: getSurfaceMatcap(),
    side: THREE.DoubleSide,
    transparent: true,
  });
  injectFresnelRim(material, new THREE.Color(0xffffff), 0.5, 2.2, "add");
  return material;
}

