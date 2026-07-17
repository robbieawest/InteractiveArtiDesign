import * as THREE from "three";
import type { SketchDocument } from "../core/SketchDocument";
import type { Vec3 } from "../core/types";
import type { Viewport } from "./Viewport";

/**
 * While the document is exploded, draws one line per joint between the
 * (current) centroids of the parent and child parts, so the articulation
 * structure stays readable with the parts pushed apart. Rebuilds are
 * coalesced to one per frame — explode drags emit a stroke event per stroke
 * per frame.
 */
export class JointLines {
  private readonly lines: THREE.LineSegments;
  private readonly unsubscribe: () => void;
  private dirty = false;

  constructor(
    private readonly doc: SketchDocument,
    private readonly viewport: Viewport,
  ) {
    this.lines = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: 0x555555,
        transparent: true,
        opacity: 0.8,
        depthTest: false,
      }),
    );
    this.lines.renderOrder = 1;
    this.lines.visible = false;
    this.viewport.scene.add(this.lines);
    this.unsubscribe = doc.subscribe(() => this.scheduleRebuild());
    this.rebuild();
  }

  dispose(): void {
    this.unsubscribe();
    this.viewport.scene.remove(this.lines);
    this.lines.geometry.dispose();
    (this.lines.material as THREE.Material).dispose();
  }

  private scheduleRebuild(): void {
    if (this.dirty) return;
    this.dirty = true;
    requestAnimationFrame(() => {
      this.dirty = false;
      this.rebuild();
      this.viewport.invalidate();
    });
  }

  private rebuild(): void {
    const joints = this.doc.allJoints();
    if (!this.doc.exploded || joints.length === 0) {
      this.lines.visible = false;
      return;
    }
    const positions: number[] = [];
    for (const joint of joints) {
      const parent = this.partCentroid(joint.parentPartId);
      const child = this.partCentroid(joint.childPartId);
      if (!parent || !child) continue;
      positions.push(parent.x, parent.y, parent.z, child.x, child.y, child.z);
    }
    this.lines.geometry.dispose();
    this.lines.geometry = new THREE.BufferGeometry();
    this.lines.geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(positions, 3),
    );
    this.lines.visible = positions.length > 0;
  }

  private partCentroid(partId: string): Vec3 | undefined {
    const strokes = this.doc.strokesInPart(partId);
    if (strokes.length === 0) return undefined;
    const c = { x: 0, y: 0, z: 0 };
    for (const s of strokes) {
      c.x += s.transform.position.x;
      c.y += s.transform.position.y;
      c.z += s.transform.position.z;
    }
    return {
      x: c.x / strokes.length,
      y: c.y / strokes.length,
      z: c.z / strokes.length,
    };
  }
}
