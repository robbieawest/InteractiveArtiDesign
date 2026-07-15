import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import type { SurfaceShape, Transform } from "../core/types";
import type { Viewport } from "./Viewport";

export type GizmoMode = "translate" | "rotate" | "scale";

/**
 * The drawing surface ("canvas" in Penzil terms): a translucent mesh the
 * draw tool raycasts against, positioned by a TransformControls gizmo.
 * Drawing still works while the surface is hidden — raycasting ignores
 * visibility — matching Penzil's behavior.
 */
export class CanvasSurface {
  readonly mesh: THREE.Mesh;
  readonly controls: TransformControls;
  shape: SurfaceShape = "plane";

  constructor(private readonly viewport: Viewport) {
    const material = new THREE.MeshToonMaterial({
      color: 0xe6edf5,
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide,
      emissive: new THREE.Color(0xffffff),
      emissiveIntensity: 0.3,
      polygonOffset: true,
      polygonOffsetFactor: 2.5,
      polygonOffsetUnits: -1,
    });
    this.mesh = new THREE.Mesh(makeGeometry("plane"), material);
    viewport.scene.add(this.mesh);

    this.controls = new TransformControls(
      viewport.camera,
      viewport.renderer.domElement,
    );
    this.controls.addEventListener("change", () => viewport.invalidate());
    viewport.scene.add(this.controls.getHelper());
    this.controls.attach(this.mesh);
    viewport.invalidate();
  }

  /** True while the pointer is over or dragging the gizmo — the draw tool
   *  must not start a stroke then. */
  get gizmoActive(): boolean {
    return (
      this.controls.enabled &&
      (this.controls.dragging || this.controls.axis !== null)
    );
  }

  setShape(shape: SurfaceShape): void {
    if (shape === this.shape) return;
    this.shape = shape;
    this.mesh.geometry.dispose();
    this.mesh.geometry = makeGeometry(shape);
    this.viewport.invalidate();
  }

  setGizmoMode(mode: GizmoMode): void {
    this.controls.setMode(mode);
    this.viewport.invalidate();
  }

  /** Return the surface to the origin with no rotation and unit scale. */
  resetTransform(): void {
    this.mesh.position.set(0, 0, 0);
    this.mesh.quaternion.identity();
    this.mesh.scale.set(1, 1, 1);
    this.viewport.invalidate();
  }

  setVisible(visible: boolean): void {
    this.mesh.visible = visible;
    this.controls.enabled = visible;
    this.controls.getHelper().visible = visible;
    this.viewport.invalidate();
  }

  /** Shape + world transform, recorded into each stroke drawn on it. */
  snapshot(): { shape: SurfaceShape; transform: Transform } {
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    this.mesh.getWorldPosition(position);
    this.mesh.getWorldQuaternion(quaternion);
    this.mesh.getWorldScale(scale);
    return {
      shape: this.shape,
      transform: {
        position: { x: position.x, y: position.y, z: position.z },
        quaternion: {
          x: quaternion.x,
          y: quaternion.y,
          z: quaternion.z,
          w: quaternion.w,
        },
        scale: { x: scale.x, y: scale.y, z: scale.z },
      },
    };
  }

  dispose(): void {
    this.controls.dispose();
    this.viewport.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

function makeGeometry(shape: SurfaceShape): THREE.BufferGeometry {
  switch (shape) {
    case "plane":
      return new THREE.PlaneGeometry(5, 5);
    case "cube":
      return new THREE.BoxGeometry(5, 5, 5);
    case "cylinder":
      return new THREE.CylinderGeometry(2.5, 2.5, 5, 24);
    case "sphere":
      return new THREE.SphereGeometry(2.5, 24, 24);
  }
}
