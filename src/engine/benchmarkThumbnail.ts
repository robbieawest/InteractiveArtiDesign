// Offscreen thumbnail renderer for the benchmark window.
//
// Thumbnails are cheap on purpose: strokes draw as plain line segments rather
// than the screen-space ribbons the editor uses, because at 200px the
// difference is invisible and a ribbon pass per sketch is not. One renderer is
// shared by every thumbnail and rendered synchronously on demand — a grid of
// live canvases would mean one WebGL context each, and browsers cap those
// around 16.

import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { SurfacingSketch } from "../surfacing/client";

const SIZE = 256;
const STROKE_COLOR = 0x333333;
const SURFACE_COLOR = 0xff9c3c;

let renderer: THREE.WebGLRenderer | null = null;
const loader = new GLTFLoader();

function getRenderer(): THREE.WebGLRenderer {
  if (!renderer) {
    renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true, // required for toDataURL
    });
    renderer.setSize(SIZE, SIZE, false);
    renderer.setClearColor(0x000000, 0);
  }
  return renderer;
}

/** Frees the shared context. Call when the benchmark window closes for good;
 *  the next thumbnail transparently makes a new one. */
export function disposeThumbnailRenderer(): void {
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

function strokeLines(sketch: SurfacingSketch): THREE.Object3D {
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
    new THREE.LineBasicMaterial({ color: STROKE_COLOR }),
  );
}

/** Frame the whole scene from a fixed three-quarter view. Every thumbnail
 *  uses the same angle so the grid reads as a set, and the pose is
 *  whatever the document stores — nothing here poses anything. */
function frame(camera: THREE.PerspectiveCamera, target: THREE.Object3D): void {
  const box = new THREE.Box3().setFromObject(target);
  if (box.isEmpty()) {
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    return;
  }
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2 || 1;
  // 1.02 rather than a comfortable margin: at 256px the sketch needs to fill
  // the frame to be recognisable in a grid
  const distance = (radius * 1.02) / Math.tan((camera.fov * Math.PI) / 360);
  const direction = new THREE.Vector3(0.6, 0.45, 1).normalize();
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = Math.max(distance / 100, 0.01);
  camera.far = distance + radius * 4;
  camera.updateProjectionMatrix();
  camera.lookAt(center);
}

/** Render one thumbnail: the sketch's stored pose, plus whatever surface
 *  geometry has arrived so far (per-part partials, or the finished object).
 *  Returns a data URL; the caller caches it and only re-renders when the
 *  geometry set changes. */
export async function renderThumbnail(
  sketch: SurfacingSketch,
  surfaces: ArrayBuffer[] = [],
): Promise<string> {
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(1, 2, 3);
  scene.add(key);

  const content = new THREE.Group();
  content.add(strokeLines(sketch));

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
          color: SURFACE_COLOR,
          roughness: 0.6,
          metalness: 0.0,
          transparent: true,
          opacity: 0.85,
          side: THREE.DoubleSide,
        });
      }
    });
    content.add(gltf.scene);
  }
  scene.add(content);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  frame(camera, content);

  const gl = getRenderer();
  gl.render(scene, camera);
  const url = gl.domElement.toDataURL("image/png");

  // nothing here outlives the call: the parsed glbs are one-shot, and holding
  // their buffers on the GPU across a grid of sketches is exactly the leak
  // this renderer exists to avoid
  content.traverse((object) => {
    const mesh = object as THREE.Mesh;
    mesh.geometry?.dispose();
    for (const material of asArray(mesh.material)) disposeMaterial(material);
  });
  return url;
}
