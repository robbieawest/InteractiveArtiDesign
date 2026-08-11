// Shared offscreen sketch rendering. Two callers sit on top of this:
// benchmarkThumbnail.ts (one small dark view for the benchmark grid) and
// strokeViews.ts (several large light views to condition an image-to-3D
// model). Everything they have in common — the shared WebGL context, stroke
// geometry, glb materials, framing, teardown — lives here; they differ only
// in the style they pass and the directions they ask for.
//
// One renderer is shared by every caller and rendered synchronously on
// demand: a live canvas per view would mean one WebGL context each, and
// browsers cap those around 16.

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { SurfacingSketch } from "../surfacing/client";

export interface SketchRenderStyle {
  /** Output is always square, `size` x `size` pixels. */
  size: number;
  strokeColor: number;
  /** Tube radius as a fraction of the sketch's bounding radius. 0 falls back
   *  to 1px LineSegments — `linewidth` is a no-op on almost every driver, so
   *  any stroke thicker than a hairline has to be real geometry. */
  strokeThickness: number;
  surfaceColor: number;
  surfaceOpacity: number;
  /** Camera pullback as a multiple of the bounding radius. ~1.0 fills the
   *  frame; >1 leaves margin around the subject. */
  margin: number;
  /** Park the key light at the camera instead of a fixed world position, so
   *  every direction is lit the same. Fixed lighting reads better for a
   *  single canonical view; a headlight is what multi-view wants, or the
   *  back views come out black. */
  headlight: boolean;
  ambientIntensity: number;
  keyIntensity: number;
}

const TUBE_RADIAL_SEGMENTS = 6;
const MAX_TUBE_SEGMENTS = 256;

let renderer: THREE.WebGLRenderer | null = null;
const loader = new GLTFLoader();

function getRenderer(size: number): THREE.WebGLRenderer {
  if (!renderer) {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true, // required for toDataURL
    });
    renderer.setClearColor(0x000000, 0);
  }
  // callers render at different resolutions through the same context
  renderer.setSize(size, size, false);
  return renderer;
}

/** Frees the shared context. Call when nothing will render for a while; the
 *  next render transparently makes a new one. */
export function disposeSketchRenderer(): void {
  renderer?.dispose();
  renderer = null;
}

function asArray(material: THREE.Material | THREE.Material[] | undefined) {
  if (!material) return [];
  return Array.isArray(material) ? material : [material];
}

/** A material and whatever textures it holds. Surfacing glbs are usually
 *  untextured, but an imported one that is not would otherwise keep its
 *  images alive for as long as the page. */
function disposeMaterial(material: THREE.Material): void {
  for (const value of Object.values(material)) {
    if (value instanceof THREE.Texture) value.dispose();
  }
  material.dispose();
}

/** Bounding radius of the stroke centerlines alone. Tube thickness is
 *  relative to this, so a sketch keeps the same apparent line weight
 *  whatever units it was drawn in. */
function strokeRadius(sketch: SurfacingSketch): number {
  const box = new THREE.Box3();
  const point = new THREE.Vector3();
  for (const stroke of sketch.strokes) {
    for (const p of stroke.points) box.expandByPoint(point.fromArray(p));
  }
  if (box.isEmpty()) return 1;
  return box.getSize(new THREE.Vector3()).length() / 2 || 1;
}

function lineStrokes(
  sketch: SurfacingSketch,
  style: SketchRenderStyle,
): THREE.Object3D {
  const positions: number[] = [];
  for (const stroke of sketch.strokes) {
    for (let i = 0; i + 1 < stroke.points.length; i++) {
      positions.push(...stroke.points[i], ...stroke.points[i + 1]);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  return new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: style.strokeColor }),
  );
}

function tubeStrokes(
  sketch: SurfacingSketch,
  style: SketchRenderStyle,
  radius: number,
): THREE.Object3D {
  const group = new THREE.Group();
  // one material for every tube: they are identical, and sharing keeps the
  // teardown below to a single dispose
  const material = new THREE.MeshStandardMaterial({
    color: style.strokeColor,
    roughness: 0.5,
    metalness: 0.0,
  });
  const tubeRadius = radius * style.strokeThickness;

  for (const stroke of sketch.strokes) {
    const points: THREE.Vector3[] = [];
    for (const p of stroke.points) {
      const v = new THREE.Vector3().fromArray(p);
      // CatmullRomCurve3 produces NaN tangents on repeated points, and a
      // stroke held still for a moment records plenty of them
      if (points.length && points[points.length - 1].distanceToSquared(v) < 1e-18) {
        continue;
      }
      points.push(v);
    }
    if (points.length < 2) continue;

    const curve = new THREE.CatmullRomCurve3(points);
    const segments = Math.min((points.length - 1) * 2, MAX_TUBE_SEGMENTS);
    const geometry = new THREE.TubeGeometry(
      curve,
      segments,
      tubeRadius,
      TUBE_RADIAL_SEGMENTS,
      false,
    );
    group.add(new THREE.Mesh(geometry, material));
  }
  return group;
}

/** Strokes as drawable geometry: hairlines when the style asks for no
 *  thickness, swept tubes otherwise. */
function buildStrokes(
  sketch: SurfacingSketch,
  style: SketchRenderStyle,
): THREE.Object3D {
  if (style.strokeThickness <= 0) return lineStrokes(sketch, style);
  return tubeStrokes(sketch, style, strokeRadius(sketch));
}

/** The sketch plus whatever surface geometry has arrived so far (per-part
 *  partials, or the finished object), as one disposable group. */
async function buildContent(
  sketch: SurfacingSketch,
  surfaces: ArrayBuffer[],
  style: SketchRenderStyle,
): Promise<THREE.Group> {
  const content = new THREE.Group();
  content.add(buildStrokes(sketch, style));

  for (const glb of surfaces) {
    // parseAsync wants its own copy; a detached buffer would poison a re-render
    const gltf = await loader.parseAsync(glb.slice(0), "");
    gltf.scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (mesh.isMesh) {
        // the glb's own material (and any texture hanging off it) is replaced
        // wholesale, so let go of it here rather than leaving it to the GC
        for (const original of asArray(mesh.material)) disposeMaterial(original);
        mesh.material = new THREE.MeshStandardMaterial({
          color: style.surfaceColor,
          roughness: 0.6,
          metalness: 0.0,
          transparent: style.surfaceOpacity < 1,
          opacity: style.surfaceOpacity,
          side: THREE.DoubleSide,
        });
      }
    });
    content.add(gltf.scene);
  }
  return content;
}

/** Point the camera at `box` from `direction`, framed to the style's margin.
 *  The pose is whatever the document stores — nothing here poses anything. */
function frameCamera(
  camera: THREE.PerspectiveCamera,
  box: THREE.Box3,
  direction: THREE.Vector3,
  margin: number,
): void {
  if (box.isEmpty()) {
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2 || 1;
  const distance = (radius * margin) / Math.tan((camera.fov * Math.PI) / 360);
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = Math.max(distance / 100, 0.01);
  camera.far = distance + radius * 4;
  camera.updateProjectionMatrix();
  // A view from directly above or below has its direction parallel to the
  // default up, which leaves lookAt with no way to choose a roll: the basis
  // is degenerate and the image comes out spun by an arbitrary angle. Fall
  // back to -Z there, so a top view is framed with the front of the sketch
  // upright rather than at random.
  camera.up.set(0, 1, 0);
  if (Math.abs(direction.y) > 0.999) camera.up.set(0, 0, -Math.sign(direction.y));
  camera.lookAt(center);
}

function disposeContent(content: THREE.Object3D): void {
  const seen = new Set<THREE.Material>();
  content.traverse((object) => {
    const mesh = object as THREE.Mesh;
    mesh.geometry?.dispose();
    for (const material of asArray(mesh.material)) {
      // tubes share one material across every stroke
      if (seen.has(material)) continue;
      seen.add(material);
      disposeMaterial(material);
    }
  });
}

/** Render one sketch from each of `directions`, returning a PNG data URL per
 *  direction. Directions need not be normalized. */
export async function renderSketchViews(
  sketch: SurfacingSketch,
  surfaces: ArrayBuffer[],
  style: SketchRenderStyle,
  directions: THREE.Vector3[],
): Promise<string[]> {
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, style.ambientIntensity));
  const key = new THREE.DirectionalLight(0xffffff, style.keyIntensity);
  key.position.set(1, 2, 3);
  scene.add(key);

  const content = await buildContent(sketch, surfaces, style);
  scene.add(content);
  const box = new THREE.Box3().setFromObject(content);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  const gl = getRenderer(style.size);
  const urls: string[] = [];

  for (const direction of directions) {
    frameCamera(camera, box, direction.clone().normalize(), style.margin);
    if (style.headlight) key.position.copy(camera.position);
    gl.render(scene, camera);
    urls.push(gl.domElement.toDataURL("image/png"));
  }

  // nothing here outlives the call: the parsed glbs are one-shot, and holding
  // their buffers on the GPU across a batch of sketches is exactly the leak
  // this renderer exists to avoid
  disposeContent(content);
  return urls;
}
