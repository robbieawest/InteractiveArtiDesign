import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { Viewport } from "./Viewport";

/** Displays the mesh returned by a surfacing job as a scene overlay. The
 *  mesh is derived output, not sketch data: it never enters the document or
 *  the undo stack, and re-running a job simply replaces it. */
export class SurfacePreview {
  private readonly group = new THREE.Group();
  private readonly loader = new GLTFLoader();

  constructor(private readonly viewport: Viewport) {
    this.viewport.scene.add(this.group);
  }

  get hasContent(): boolean {
    return this.group.children.length > 0;
  }

  async show(glb: ArrayBuffer): Promise<void> {
    const gltf = await this.loader.parseAsync(glb, "");
    // keep the strokes readable through the surface
    gltf.scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        for (const material of asArray(obj.material)) {
          material.transparent = true;
          material.opacity = Math.min(material.opacity, 0.55);
          material.side = THREE.DoubleSide;
        }
      }
    });
    this.clear();
    this.group.add(gltf.scene);
    this.viewport.invalidate();
  }

  clear(): void {
    for (const child of [...this.group.children]) {
      child.traverse((obj) => {
        if (obj instanceof THREE.Mesh) {
          obj.geometry.dispose();
          for (const material of asArray(obj.material)) material.dispose();
        }
      });
      this.group.remove(child);
    }
    this.viewport.invalidate();
  }

  dispose(): void {
    this.clear();
    this.viewport.scene.remove(this.group);
  }
}

function asArray(m: THREE.Material | THREE.Material[]): THREE.Material[] {
  return Array.isArray(m) ? m : [m];
}
